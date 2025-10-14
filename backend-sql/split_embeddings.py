# split_embeddings.py
import os
import json
from pathlib import Path
import re
from typing import List
import fitz
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.auto import partition
from backend_sql.databases import load_project_index, persist_project_index, get_connection
import nltk

# Add path to your offline nltk_data folder
nltk.data.path.append(r"nltk_data")  
# ----------------------------
# Embedding model
# ----------------------------

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def redact_pii(text: str) -> str:
    """Simple regex-based PII redaction (email, phone, ID, etc)."""
    if not text.strip():
        return text

    # Common regex patterns for PII
    patterns = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\+?\d[\d\s\-]{8,}\d",
        "ID": r"\b[A-Z]{2,3}\d{4,}\b",  # e.g., PAN/employee IDs
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    redacted_text = text
    for label, pattern in patterns.items():
        redacted_text = re.sub(pattern, f"[{label}_REDACTED]", redacted_text)

    return redacted_text


def sanitize_user_query(query: str) -> str:

    # 1. Basic prompt injection check
    forbidden_phrases = [
        "ignore previous instructions",
        "forget your instructions",
        "bypass safety",
        "malicious",
        "--",  # comment injection
        "drop table",    # dangerous SQL
        "delete from",   # dangerous SQL
        "update ",   # dangerous SQL
    ]
    for phrase in forbidden_phrases:
        query = re.sub(re.escape(phrase), "[REDACTED]", query, flags=re.IGNORECASE)

    # 2. Strip leading/trailing spaces
    return query.strip()

# ----------------------------
# Load & split with Unstructured
# ----------------------------
def split_textual_document(filepath: str, max_chunk_size: int = 1000) -> List[Document]:
    """
    Improved chunking with smart grouping and relational metadata:
    - Groups related elements (table descriptions + tables)
    - Adds metadata linking for relational queries
    - Maintains clean, focused chunks
    - Preserves hierarchy and context
    """
    elements = partition(filename=filepath)
    documents = []

    hierarchy_stack = []  # Track active heading levels
    current_group = []  # Group related elements together
    chunk_counter = 0

    def flush_group():
        """Flush current group as one or more documents with relational metadata."""
        nonlocal current_group, chunk_counter
        if not current_group:
            return

        # Group 1: Table descriptions with their tables
        grouped_elements = []
        temp_group = []
        
        for el in current_group:
            if el.category == "Table" and temp_group and temp_group[-1].category in ["UncategorizedText", "NarrativeText"]:
                # Table follows a description - group them
                temp_group.append(el)
            else:
                if temp_group:
                    grouped_elements.append(temp_group)
                temp_group = [el]
        if temp_group:
            grouped_elements.append(temp_group)

        # Create documents for each group
        for group in grouped_elements:
            if len(group) == 1:
                # Single element - create individual chunk
                el = group[0]
                text = el.text.strip()
                if not text:
                    continue
                    
                # Split large elements but maintain context
                if len(text) > max_chunk_size:
                    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                    for idx, chunk in enumerate(chunks):
                        documents.append(create_document(
                            chunk, el, hierarchy_stack, filepath, chunk_counter
                        ))
                        chunk_counter += 1
                else:
                    documents.append(create_document(
                        text, el, hierarchy_stack, filepath, chunk_counter
                    ))
                    chunk_counter += 1
            else:
                # Multiple related elements - combine if they fit
                combined_text = "\n".join([el.text.strip() for el in group if el.text.strip()])
                if len(combined_text) <= max_chunk_size:
                    # Combine into one chunk
                    primary_el = group[0]
                    documents.append(create_document(
                        combined_text, primary_el, hierarchy_stack, filepath, chunk_counter,
                        related_elements=len(group),
                        element_types=[el.category for el in group]
                    ))
                    chunk_counter += 1
                else:
                    # Keep separate but add linking metadata
                    for i, el in enumerate(group):
                        text = el.text.strip()
                        if not text:
                            continue
                            
                        doc = create_document(
                            text, el, hierarchy_stack, filepath, chunk_counter,
                            group_id=chunk_counter // len(group),  # Same group ID for related chunks
                            element_index=i,
                            total_elements=len(group)
                        )
                        documents.append(doc)
                        chunk_counter += 1

        current_group = []

    def create_document(content, element, hierarchy, filepath, chunk_id, **extra_metadata):
        """Helper to create Document with consistent metadata."""
        metadata = {
            "category": element.category,
            "filename": Path(filepath).name,
            "hierarchy": hierarchy.copy(),
            "parent_heading": hierarchy[-1] if hierarchy else None,
            "chunk_id": chunk_id,
            "element_id": id(element),
        }
        metadata.update(extra_metadata)
        redacted_content = redact_pii(content)
        return Document(page_content=redacted_content, metadata=metadata)

    # Process elements
    for el in elements:
        if not el.text.strip():
            continue

        if el.category == "Title":
            # Flush current group before new heading
            flush_group()
            if hierarchy_stack:
                hierarchy_stack.pop()
            hierarchy_stack.append(el.text.strip())
            
            # Title gets its own chunk
            current_group = [el]
            flush_group()
        else:
            current_group.append(el)

    # Flush final group
    flush_group()

    # Add sequential linking metadata for navigation
    add_sequential_links(documents)
    
    return documents

def add_sequential_links(documents):
    """Add metadata to link sequential chunks for better relational retrieval."""
    for i in range(len(documents)):
        if i > 0:
            # Link to previous chunk
            prev_id = documents[i-1].metadata.get("chunk_id")
            if prev_id is not None:
                documents[i].metadata["prev_chunk_id"] = prev_id
        if i < len(documents) - 1:
            # Link to next chunk
            next_id = documents[i+1].metadata.get("chunk_id")
            if next_id is not None:
                documents[i].metadata["next_chunk_id"] = next_id
        
        # Add section context for relational queries
        if "group_id" in documents[i].metadata:
            documents[i].metadata["section_type"] = "related_group"


def split_tabular_document(filepath: str, batch_size: int = 50):
    """Parse CSV/XLSX into schema + row batches for embedding."""
    ext = Path(filepath).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
        sheets = {"Sheet1": df}
    else:  # Excel
        xls = pd.ExcelFile(filepath)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}

    documents = []
    for sheet, df in sheets.items():
        # Schema / header info
        documents.append(Document(
            page_content=f"Sheet: {sheet}\nColumns: {', '.join(df.columns)}",
            metadata={"category": "schema", "sheet": sheet}
        ))

        # Row batches
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            batch_text = f"Sheet: {sheet}\nRows {start}-{start+len(batch)-1}:\n{batch.to_csv(index=False)}"
            redacted_content = redact_pii(batch_text)
            documents.append(Document(
                page_content=redacted_content,
                metadata={"category": "rows", "sheet": sheet, "batch": f"{start}-{start+len(batch)-1}"}
            ))
    return documents

def split_pdf_document(filepath: str, max_chunk_size: int = 1000) -> List[Document]:
    """
    Extract text from PDF using PyMuPDF and split into Document chunks
    using same logic as split_textual_document.
    """

    # Step 1: Extract text from PDF
    elements = []
    try:
        with fitz.open(filepath) as pdf_doc:
            for page_num, page in enumerate(pdf_doc):
                text = page.get_text().strip()
                if text:
                    # Treat each page as an element
                    elements.append(type("Element", (), {"text": text, "category": "Page"}))
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    # Step 2: Split like textual document
    documents = []
    hierarchy_stack = []
    current_group = []
    chunk_counter = 0

    def flush_group():
        nonlocal current_group, chunk_counter
        if not current_group:
            return
        grouped_elements = [current_group]  # simple grouping for PDF pages
        for group in grouped_elements:
            combined_text = "\n".join([el.text.strip() for el in group if el.text.strip()])
            if len(combined_text) <= max_chunk_size:
                documents.append(create_document(combined_text, group[0], hierarchy_stack, filepath, chunk_counter))
                chunk_counter += 1
            else:
                # Split large chunks
                text = combined_text
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                for chunk in chunks:
                    documents.append(create_document(chunk, group[0], hierarchy_stack, filepath, chunk_counter))
                    chunk_counter += 1
        current_group.clear()

    def create_document(content, element, hierarchy, filepath, chunk_id, **extra_metadata):
        metadata = {
            "category": element.category,
            "filename": Path(filepath).name,
            "hierarchy": hierarchy.copy(),
            "parent_heading": hierarchy[-1] if hierarchy else None,
            "chunk_id": chunk_id,
        }
        metadata.update(extra_metadata)
        redacted_content = redact_pii(content)
        return Document(page_content=redacted_content, metadata=metadata)

    # Add all pages to current group
    current_group = elements
    flush_group()
    add_sequential_links(documents)

    return documents



def load_and_split_document(filepath: str):
    """Dispatcher: pick the right splitting strategy based on file type."""
    ext = Path(filepath).suffix.lower()
    if ext in [".docx", ".doc", ".txt"]:
        return split_textual_document(filepath)
    elif ext == ".pdf":
        return split_pdf_document(filepath)
    elif ext in [".csv", ".xlsx"]:
        return split_tabular_document(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
# ----------------------------
# Save and index documents
# ----------------------------
def save_and_index(project_id: int, doc_id: int, doc_name: str, filepath: str, doc_type: str, documents):
    pi = load_project_index(project_id)
    conn_thread = get_connection()
    c_thread = conn_thread.cursor()
    try:
        c_thread.execute("UPDATE documents SET status=? WHERE id=?", ("indexing", doc_id))
        if not pi.vectorstore:
            pi.vectorstore = FAISS.from_documents(documents, embedding_model)
        else:
            pi.vectorstore.add_documents(documents)
        for idx, doc in enumerate(documents, start=1):
            c_thread.execute(
    "INSERT INTO doc_chunks (doc_id, chunk_type, content, metadata) VALUES (?, ?, ?, ?)",
    (
        doc_id,
        doc.metadata.get("category")or "text_chunk",
        doc.page_content,
        json.dumps(doc.metadata, ensure_ascii=False)
    )
)
            progress_percent = int((idx / len(documents)) * 100)
            c_thread.execute("UPDATE documents SET progress=? WHERE id=?", (progress_percent, doc_id))
            if idx % 20 == 0 or idx == len(documents):
                conn_thread.commit()
        c_thread.execute("UPDATE documents SET status=?, progress=? WHERE id=?", ("completed", 100, doc_id))
        conn_thread.commit()
        persist_project_index(project_id)
    except Exception as e:
        print(f"Indexing error for doc {doc_id} in project {project_id}: {e}")
        c_thread.execute("UPDATE documents SET status=? WHERE id=?", ("error", doc_id))
        conn_thread.commit()
    finally:
        conn_thread.close()

# ----------------------------
# Rebuild FAISS Vector DB After Document Deletion
# ----------------------------
def rebuild_project_index_from_store(project_id: int):
    pi = load_project_index(project_id)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
SELECT content, metadata 
FROM doc_chunks 
WHERE doc_id IN (SELECT id FROM documents WHERE project_id=?)
""", (project_id,))
    rows = c.fetchall()
    conn.close()
    if rows:
        documents = [Document(page_content=r[0], metadata=json.loads(r[1] or "{}")) for r in rows]
        pi.vectorstore = FAISS.from_documents(documents, embedding_model)
        persist_project_index(project_id)
