# split_embeddings.py

# Chunking and Saving the documents in vector DB

import os
import json
from pathlib import Path
from typing import List
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.auto import partition
from databases import load_project_index, persist_project_index, get_connection

# ----------------------------
# Embedding model
# ----------------------------

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------------
# Load & split with Unstructured
# ----------------------------
def split_textual_document(filepath: str, max_chunk_size: int = 1000) -> List[Document]:
    """
    Parse Word/PDF/TXT using unstructured and preserve hierarchy:
    - Headings (Title) define new sections.
    - Paragraphs, tables, lists are attached under the nearest heading.
    - Each Document stores a 'hierarchy' breadcrumb in metadata.
    - Large elements (long text or tables) are split into smaller chunks.
    """
    elements = partition(filename=filepath)
    documents = []

    hierarchy_stack = []  # Track active heading levels
    current_section = []  # Collect elements under the current heading

    def flush_section():
        """Flush accumulated section into documents with hierarchy metadata."""
        nonlocal current_section
        if not current_section:
            return

        hierarchy = [el for el in hierarchy_stack]
        for el in current_section:
            text = el.text.strip()
            if not text:
                continue

            # Split if element is too large
            if len(text) > max_chunk_size:
                # Sentence-based splitting for long paragraphs
                subchunks = [
                    text[i:i+max_chunk_size]
                    for i in range(0, len(text), max_chunk_size)
                ]
            else:
                subchunks = [text]

            for idx, chunk in enumerate(subchunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "category": el.category,
                        "filename": Path(filepath).name,
                        "hierarchy": hierarchy,
                        "parent_heading": hierarchy_stack[-1] if hierarchy_stack else None,
                        "chunk_index": idx
                    }
                ))
        current_section = []

    for el in elements:
        if not el.text.strip():
            continue

        if el.category == "Title":
            # Flush previous section before starting new one
            flush_section()
            if hierarchy_stack:
                hierarchy_stack.pop()
            hierarchy_stack.append(el.text.strip())
        else:
            # Add this element under current heading
            current_section.append(el)

    # Flush last section
    flush_section()

    return documents

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
            documents.append(Document(
                page_content=f"Sheet: {sheet}\nRows {start}-{start+len(batch)-1}:\n{batch.to_csv(index=False)}",
                metadata={"category": "rows", "sheet": sheet, "batch": f"{start}-{start+len(batch)-1}"}
            ))
    return documents


def load_and_split_document(filepath: str):
    """Dispatcher: pick the right splitting strategy based on file type."""
    ext = Path(filepath).suffix.lower()
    if ext in [".pdf", ".docx", ".doc", ".txt"]:
        return split_textual_document(filepath)
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
