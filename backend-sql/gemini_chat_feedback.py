# gemini_chat_feedback.py
import os
import json
from typing import Any, List, Literal, Optional, Union
import uuid
import time
import random
import logging
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel
from backend_sql.databases import get_connection, load_project_index
from backend_sql.split_embeddings import rebuild_project_index_from_store, redact_pii, sanitize_user_query
from langchain.schema import Document
from google import genai
from backend_sql.auth import get_current_user
from rank_bm25 import BM25Okapi
import sqlglot
from sqlglot.expressions import Table, Column, Join
from google.genai.types import GenerateContentConfig
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Last 20 Queries Given as Memory to LLM
conversation_memory: dict[str, list[dict]] = {}
MAX_MEMORY_LENGTH = 20

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "data", "feedback")
os.makedirs(FEEDBACK_DIR, exist_ok=True)
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")

router = APIRouter()

# ----------------------------
# Gemini API call
# ----------------------------
def call_gemini(prompt: str, retries: int = 3, base_delay: int = 2, stream: bool = False):
    """
    Call Gemini API with fallback models and optional streaming.
    If stream=True → yields partial chunks instead of returning full text.
    """
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    attempt_count = 0
    is_fallback = False
    for model in models:
        for attempt in range(1, retries + 2):
            attempt_count += 1
            try:
                # --- STREAMING MODE ---
                if stream:
                    response_stream = genai_client.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=GenerateContentConfig(
                        temperature=0.3,
                    ),
                    )
                    for chunk in response_stream:
                        if hasattr(chunk, "text") and chunk.text:
                            yield {
                                "text": chunk.text,
                                "attempt_count": attempt_count,
                                "is_fallback": is_fallback
                            }  # Send to frontend
                    return  # stop after yielding stream

                # --- NON-STREAMING (original path) ---
                response = genai_client.models.generate_content(model=model, contents=prompt)
                return (
                    getattr(response, "text", None)
                    or getattr(response, "content", None)
                    or str(response),
                    attempt_count,
                    False,
                )

            except Exception as e:
                err_msg = str(e)
                logging.error(f"[Gemini API Error] Model={model} Attempt={attempt}: {err_msg}")
                if "503" in err_msg or "overloaded" in err_msg.lower():
                    if attempt <= retries:
                        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        time.sleep(delay)
                        continue
                    else:
                        is_fallback = True
                        break
                else:
                    is_fallback = True
                    if stream:
                        yield {
                            "text": "❌ Something went wrong while processing your request.",
                            "attempt_count": attempt_count,
                            "is_fallback": True
                        }
                        return
                    return "❌ Something went wrong while processing your request.", attempt_count, True

    if stream:
        yield {
            "text": "⚠️ All Gemini models are busy. Please try again later.",
            "attempt_count": attempt_count,
            "is_fallback": True
        }
        return
    return "⚠️ All Gemini models are busy. Please try again later.", attempt_count, True

# ---------------------------------------
# Document Name Detection in the User's Query
# ---------------------------------------

def extract_doc_names_from_query(project_id: int,query: str) -> List[str]:
    """
    Detect document names mentioned in the user query.
    Simple heuristic: look for words matching existing document names in the project.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM documents WHERE project_id=?", (project_id,))
    doc_names = [r[0] for r in c.fetchall()]
    conn.close()
    
    # Case-insensitive match of any doc name in the query
    mentioned = [name for name in doc_names if re.search(rf"\b{name}\b", query, re.IGNORECASE)]
    return mentioned

# ---------------------------------------------
# Retrieval based on Document mentioned in the User Query
# ---------------------------------------------

def retrieve_from_doc_by_name(project_id: int, doc_name: str) -> List[Document]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM documents WHERE project_id=? AND LOWER(name)=?", (project_id, doc_name.lower()))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    doc_id = row[0]
    c.execute("SELECT content, metadata FROM doc_chunks WHERE doc_id=?", (doc_id,))
    rows = c.fetchall()
    conn.close()

    docs = []
    for content, metadata_json in rows:
        metadata = json.loads(metadata_json) if metadata_json else {}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

# ----------------------------
# Retrieval based on User's Query
# ----------------------------

def retrieve_relevant_chunks(project_id: int, query: str, threshold: float = 0.7, max_candidates: int = 50, alpha: float = 0.5) -> List[Document]:
    """
    Hybrid retrieval using FAISS (semantic) + BM25 (keyword) with relational awareness:
    - Pull top-N candidates from FAISS.
    - Compute BM25 scores on all docs.
    - Fuse scores.
    - Include related chunks using group_id and sequential links.
    - Deduplicate and return ranked results.
    """

    def normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()
    
    pi = load_project_index(project_id)
    if not pi.vectorstore:
        return []

    # ----------------------------
    # Step 1: Embed query for FAISS
    # ----------------------------
    query_embedding = pi.vectorstore.embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    # ----------------------------
    # Step 2: FAISS retrieval
    # ----------------------------
    faiss_index = pi.vectorstore.index
    scores, indices = faiss_index.search(query_embedding, max_candidates)

    faiss_docs = []
    faiss_scores = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc_id = pi.vectorstore.index_to_docstore_id[idx]
        doc = pi.vectorstore.docstore.search(doc_id)
        if not doc or not getattr(doc, "page_content", "").strip():
            continue
        faiss_docs.append(doc)
        faiss_scores.append(score)

    # ----------------------------
    # Step 3: BM25 retrieval
    # ----------------------------
    all_docs = [d for d in pi.vectorstore.docstore._dict.values() if getattr(d, "page_content", "").strip()]
    tokenized_corpus = [d.page_content.split() for d in all_docs]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    bm25_scores = bm25.get_scores(query.split()) if bm25 else [0] * len(all_docs)

    # ----------------------------
    # Step 4: Normalize scores
    # ----------------------------
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    faiss_scores_norm = normalize(np.array(faiss_scores)) if faiss_scores else np.array([])
    bm25_scores_norm = normalize(np.array(bm25_scores)) if bm25_scores is not None and bm25_scores.size > 0 else np.array([])


    # ----------------------------
    # Step 5: Fuse scores with relational awareness
    # ----------------------------
    final_scores = {}

    def add_doc(doc, score):
        key = normalize_key(doc.page_content)
        if key in final_scores:
            final_scores[key] += score
        else:
            final_scores[key] = score

    # FAISS top docs
    for doc, f_score in zip(faiss_docs, faiss_scores_norm):
        add_doc(doc, alpha * f_score)

        # Add related group chunks
        if "group_id" in doc.metadata:
            group_id = doc.metadata["group_id"]
            for d in all_docs:
                if d.metadata.get("group_id") == group_id and d.page_content != doc.page_content:
                    add_doc(d, alpha * f_score * 0.8)  # slightly lower weight for related

        # Add sequential neighbors
        prev_id = doc.metadata.get("prev_chunk_id")
        next_id = doc.metadata.get("next_chunk_id")
        for neighbor in all_docs:
            if neighbor.metadata.get("chunk_id") in [prev_id, next_id]:
                add_doc(neighbor, alpha * f_score * 0.6)

    # BM25 scores
    for doc, b_score in zip(all_docs, bm25_scores_norm):
        add_doc(doc, (1 - alpha) * b_score)

    # ----------------------------
    # Step 6: Rank & Deduplicate
    # ----------------------------
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    unique_docs, seen = [], set()
    for content, _ in ranked:
        key = normalize_key(content)
        if key not in seen:
            for d in all_docs:
                if normalize_key(d.page_content) == key:
                    unique_docs.append(d)
                    break
            seen.add(key)
            
    return unique_docs[:max_candidates]

# ----------------------------
# Query WorkFlow
# ----------------------------

def query_workflow(project_id: int, user_query: str) -> str:
    """
    Dynamic query-driven workflow:
    - Detects mentioned documents in the user query.
    - Retrieves relevant chunks from those documents and the overall project.
    - Sends structured context to Gemini LLM to perform the requested actions.
    """

    # Step 1: Detect document names mentioned in the query
    mentioned_docs = extract_doc_names_from_query(project_id,user_query)

    # Step 2: Retrieve chunks for mentioned documents first
    contexts = []
    for doc in mentioned_docs:
        doc_chunks = retrieve_from_doc_by_name(project_id, doc)
        contexts.extend(doc_chunks)

    # Step 3: Retrieve additional project-wide relevant chunks via FAISS
    project_chunks = retrieve_relevant_chunks(project_id, user_query)
    contexts.extend(project_chunks)
    
    # Remove Duplicate contents
    seen_texts = set()
    unique_contexts = []
    for doc in contexts:
        if doc.page_content not in seen_texts:
            unique_contexts.append(doc)
            seen_texts.add(doc.page_content)
    
    context_json = []
    for chunk in unique_contexts:
        context_json.append({
            "content": chunk.page_content,
            "source": chunk.metadata.get("filename", "unknown"),
            "section": chunk.metadata.get("hierarchy", ""),
            "chunk_index": chunk.metadata.get("chunk_index", 0)
        })
        
    final_json = json.dumps(context_json, separators=(",", ":"))
            
    # Step 4: Construct a single structured prompt for Gemini
    structured_prompt = f"""
    # ROLE: Expert SQL Assistant
    # CONTEXT: You are analyzing documentation to generate precise SQL queries.

    # AVAILABLE DOCUMENTATION:
    {final_json}

    # USER QUERY TO ANSWER:
    {user_query}

    # CRITICAL INSTRUCTIONS (MUST FOLLOW):

    1. Generate valid SQL queries that answer the user question exactly.
    2. Use proper table and column names as given in the contexts.
    3. ***CRITICAL: Carefully read and interpret the content and their relationships. This point must be strictly followed.***
    - Extract all tables, columns, datatypes, PKs, FKs, and relationships
    - Infer business rules or constraints when mentioned
    4. Provide a complete, coherent, and accurate response based only on the above contexts.
    5. SQL queries should be in copy-friendly format within "```sql ```".
    6. When explaining the query:
    - generate an ER diagram-like ASCII structure in separate copy-friendly format:
    - Each table in its own box (┌─┐ │ └─┘)
    - PKs with "PK:" prefix, FKs with "[FK]"
    - Show joins with ◄────► arrows
    - Place aggregation/filter/GROUP BY/HAVING/ORDER BY/other conditions boxes separately and join them.
    - Keep it compact and aligned, like a schema diagram.
    7. Use proper indentation for readability.
    8. If no document or context is provided or needed, still generate accurate and syntactically correct SQL queries for general or conceptual questions.
    9. Recognize common conversational messages like greetings, expressions of gratitude, or farewells, and respond appropriately in a polite, context-aware, and engaging manner, matching the tone and formality of the user while optionally including a brief follow-up or acknowledgment.
"""

    # Step 5: Call Gemini API and return the answer
    last_attempt_count = 0
    last_is_fallback = False
    for chunk in call_gemini(structured_prompt, stream=True):
        last_attempt_count = chunk.get("attempt_count", last_attempt_count)
        last_is_fallback = chunk.get("is_fallback", last_is_fallback)
        yield {"text": chunk.get("text", ""), "attempt_count": last_attempt_count, "is_fallback": last_is_fallback, "contexts": final_json}

    # Step 5: Return contexts safely as attribute in workflow dict (not function)
    return {"contexts": final_json, "attempt_count": last_attempt_count, "is_fallback": last_is_fallback}




def evaluate_query_rag(project_id, user_query, sql_gen, docs):
    import sqlglot
    import re
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Ensure docs is a list
    if not isinstance(docs, list):
        docs = [docs]

    # Normalize all docs to dicts with 'content'
    normalized_docs = []
    for d in docs:
        if hasattr(d, "page_content"):  # Document object
            normalized_docs.append({"content": d.page_content})
        elif isinstance(d, dict):
            if "page_content" in d:
                normalized_docs.append({"content": d["page_content"]})
            elif "content" in d:
                normalized_docs.append({"content": d["content"]})
            else:
                normalized_docs.append({"content": str(d)})
        elif isinstance(d, str):
            normalized_docs.append({"content": d})
        else:
            normalized_docs.append({"content": str(d)})
            
    if not normalized_docs:
        normalized_docs = [{"content": ""}]


    # --- Extract SQL from sql blocks ---
    def extract_sql_blocks(text):
        pattern = r"```sql(.*?)```"
        blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not blocks:
            # fallback: extract lines starting with SQL keywords
            blocks = re.findall(r"(SELECT .*?;)", text, flags=re.DOTALL | re.IGNORECASE)
        return [b.strip() for b in blocks]

    # --- Extract SQL claims ---
    def extract_sql_claims(sql_text):
        import re
        import sqlglot
        from sqlglot.expressions import Table, Column, Join

        claims = []
        trees = []

        # Split into statements by semicolon
        statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

        for stmt in statements:
            try:
                parsed_trees = sqlglot.parse(stmt)
            except Exception as e:
                claims.append(f"parse_error:{e}")
                continue

            for tree in parsed_trees:
                if tree is None:   # 🔥 Skip empty parse results
                    continue
                trees.append(tree)

                # Extract tables
                for t in tree.find_all(Table):
                    if t.name:
                        claims.append(f"table:{t.name.lower()}")

                # Extract columns
                for c in tree.find_all(Column):
                    if c.name:
                        col_name = c.name.lower().split(".")[-1]
                        claims.append(f"column:{col_name}")

                # Extract joins
                for j in tree.find_all(Join):
                    try:
                        on_expr = j.on() if callable(j.on) else j.on
                        if on_expr:
                            join_clause = re.sub(r"\s+", " ", on_expr.sql().lower()).strip()
                            claims.append(f"join:{join_clause}")
                    except Exception:
                        # safely ignore joins that fail
                        continue

        return claims, trees


    # --- Combine all SQL blocks ---
    sql_blocks = extract_sql_blocks(sql_gen)
    if not sql_blocks:
        print("[RAG DEBUG] No SQL blocks found in answer.")
    else:
        print("[RAG DEBUG] SQL Blocks Found:", sql_blocks)
    all_claims = []
    all_trees = []
    for block in sql_blocks:
        claims, trees = extract_sql_claims(block)
        all_claims.extend(claims)
        all_trees.extend(trees)

    # --- Claim support rate ---
    def claim_support_rate(claims, normalized_docs):
        if not claims or not normalized_docs:
            return 0.0
        supported = 0
        for c in claims:
            if ":" not in c:
                continue
            claim_text = c.split(":", 1)[-1].lower()
            found = any(claim_text in d["content"].lower() for d in normalized_docs)
            if found:
                supported += 1
        return supported / len(claims)

    # --- Precision @ K ---
    def precision_at_k(normalized_docs, claims, k=5):
        if not claims or not normalized_docs:
            return 0.0
        claim_names = [c.split(":", 1)[-1].lower() for c in claims if ":" in c]
        relevant_count = 0
        top_docs = normalized_docs[:k]
        for d in top_docs:
            content_lower = d["content"].lower()
            if any(cn in content_lower for cn in claim_names):
                relevant_count += 1
        return relevant_count / k

    # --- Static SQL execution score ---
    from sqlglot.expressions import Table, Column

    def static_sql_score(sql_trees, normalized_docs):
        if not sql_trees:
            return 0.0

        scores = []

        for tree in sql_trees:
            if tree is None:   # 🔥 Guard here too
                continue
            try:
                score_parts = []

                # Columns
                cols = [c.name.lower().split(".")[-1] for c in tree.find_all(Column)]
                col_hits = sum(any(col in d["content"].lower() for d in normalized_docs) for col in cols)
                score_parts.append(col_hits / len(cols) if cols else 1.0)

                # Tables
                tables = [t.name.lower() for t in tree.find_all(Table)]
                table_hits = sum(any(t in d["content"].lower() for d in normalized_docs) for t in tables)
                score_parts.append(table_hits / len(tables) if tables else 1.0)

                scores.append(sum(score_parts) / len(score_parts) if score_parts else 1.0)
                
            except Exception as e:
                print(f"⚠️ Error scoring tree: {e}")
                continue

        return sum(scores) / len(scores)

    # --- Query coverage using user query dynamically ---
    def query_coverage(user_query, normalized_docs):
        if not user_query or not normalized_docs:
            return 0.0
        query_terms = [w.lower() for w in re.findall(r"\b\w+\b", user_query) if len(w) > 2]
        if not query_terms:
            return 0.0
        covered = 0
        for term in query_terms:
            found = any(term in d["content"].lower() for d in normalized_docs)
            if found:
                covered += 1
        return covered / len(query_terms)

    # --- Debug ---
    print("Top 3 chunk texts:", [d["content"][:100] for d in normalized_docs[:3]])
    print("Extracted SQL claims:", all_claims)

    if all_claims and all_trees:
        precision1 = precision_at_k(normalized_docs, all_claims, 1)
        precision3 = precision_at_k(normalized_docs, all_claims, 3)
        precision5 = precision_at_k(normalized_docs, all_claims, 5)
        claim_support = claim_support_rate(all_claims, normalized_docs)
        static_exec = static_sql_score(all_trees, normalized_docs)
    else:
        # No SQL found, don't penalize
        precision1 = precision3 = precision5 = None
        claim_support = static_exec = None

    # Query coverage can still be computed, because it's based on the user query
    query_cov = query_coverage(user_query, normalized_docs)

    return {
        "Precision@1": precision1,
        "Precision@3": precision3,
        "Precision@5": precision5,
        "ClaimSupportRate": claim_support,
        "StaticExecScore": static_exec,
        "QueryCoverage": query_cov,
        "NumClaims": len(all_claims)
    }



# ----------------------------
# Chat Model
# ----------------------------

class ChatRequest(BaseModel):
    project_id: int
    query: str
    conversation_id: Optional[str] = None  # optional
    top_k: int = 5

@router.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conversation_id = req.conversation_id or str(uuid.uuid4())
    safe_query = redact_pii(req.query)
    safe_query = sanitize_user_query(safe_query)
    start_time = time.time()
    
    # Initialize memory if missing
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []
        # Recover past conversation from DB
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT query, answer FROM queries
            WHERE conversation_id=? AND project_id=? AND user_id=?
            ORDER BY created_at
        """, (conversation_id, req.project_id, user_id))
        rows = c.fetchall()
        conn.close()

        for q, a in rows:
            conversation_memory[conversation_id].append({"role": "user", "content": q})
            conversation_memory[conversation_id].append({"role": "assistant", "content": a})

    memory = conversation_memory[conversation_id]
    memory.append({"role": "user", "content": safe_query})

    # Build prompt for LLM
    chat_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in memory])
    prompt = f"{chat_history_str}\nAssistant:"

    # --- Initialize a placeholder in DB if needed ---
    query_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO queries (id, project_id, query, answer, user_id, conversation_id, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (query_id, req.project_id, safe_query, "", user_id, conversation_id, 0))
    conn.commit()
    conn.close()

    # --- Streaming generator ---
    async def stream_and_save():
        full_answer = ""
        attempt_count = 0
        is_fallback = False
        contexts = ""
        for chunk in query_workflow(req.project_id, prompt):
            text = chunk.get("text", "")
            full_answer += text

            # Update attempt/fallback
            attempt_count = chunk.get("attempt_count", attempt_count)
            is_fallback = chunk.get("is_fallback", is_fallback)
            ctx = chunk.get("contexts", contexts)

            # ✅ Normalize context type
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except:
                    ctx = []
            contexts = ctx

            yield text
    
        # Response Time of LLM
        response_time = time.time() - start_time
        
        # Append assistant response to memory
        memory.append({"role": "assistant", "content": full_answer})

        # Truncate memory
        if len(memory) > MAX_MEMORY_LENGTH:
            conversation_memory[conversation_id] = memory[-MAX_MEMORY_LENGTH:]


    # --- Compute metrics (optional) ---
        pi = load_project_index(req.project_id)
        print(contexts)
        metrics = evaluate_query_rag(req.project_id, safe_query, full_answer, contexts)
        print(f"[RAG Metrics] Query: {safe_query}\n{metrics}\n")

        # --- Update DB with final answer and metrics ---
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE queries
            SET answer = ?, response_time = ?, attempts = ?, is_fallback = ?, 
                precision1 = ?, precision3 = ?, precision5 = ?, 
                claim_support = ?, static_exec = ?, query_coverage = ?, num_claims = ?
            WHERE id = ?
        """, (
            full_answer,
            response_time,
            attempt_count,  # or attempt_count if available
            is_fallback,  # or is_fallback if available
            metrics.get("Precision@1"),
            metrics.get("Precision@3"),
            metrics.get("Precision@5"),
            metrics.get("ClaimSupportRate"),
            metrics.get("StaticExecScore"),
            metrics.get("QueryCoverage"),
            metrics.get("NumClaims"),
            query_id
        ))
        conn.commit()
        conn.close()
        
        yield f"\n[[[META]]]{json.dumps({'contexts': contexts, 'query_id': query_id, 'conversation_id': conversation_id})}"

    return StreamingResponse(stream_and_save(), media_type="text/plain")


# ----------------------------
# Feedback model
# ----------------------------
class Feedback(BaseModel):
    query_id: str
    project_id: int
    query: str = ""
    answer: str
    contexts: Any = []
    feedback: Literal["up", "down"]   # "up" or "down"
    comment: str = ""
    user_id: str

# ----------------------------
# Load existing feedback
# ----------------------------
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r") as f:
        feedback_store = json.load(f)
else:
    feedback_store = []


# ----------------------------
# Feedback endpoint
# ----------------------------
@router.post("/feedback")
async def submit_feedback(item: Feedback, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        contexts = item.contexts
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)  # parse JSON string
            except json.JSONDecodeError:
                contexts = [contexts]  # wrap raw string in list
        elif contexts is None:
            contexts = []
        elif not isinstance(contexts, list):
            contexts = [contexts]  # wrap anything else in list

        item.contexts = contexts
        # --- Load existing feedback ---
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback_store = json.load(f)
        else:
            feedback_store = []

        # --- Save new feedback entry ---
        fb_entry = {
            "id": str(uuid.uuid4()),
            "query_id": item.query_id,
            "project_id": item.project_id,
            "query": item.query,
            "answer": item.answer,
            "contexts": item.contexts,
            "feedback": item.feedback,
            "comment": item.comment,
            "user_id": user_id
        }
        feedback_store.append(fb_entry)

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_store, f, indent=2)

        # --- Save feedback to SQLite ---
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO feedback (id, query_id, project_id, answer, contexts, feedback, comment, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fb_entry["id"],
            fb_entry["query_id"],
            fb_entry["project_id"],
            fb_entry["answer"],
            json.dumps(fb_entry["contexts"], separators=(",", ":")),
            fb_entry["feedback"],
            fb_entry["comment"],
            fb_entry["user_id"]
        ))
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": "Feedback stored",
            "id": fb_entry["id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {e}")
    
# ----------------------------
# Get all conversations/queries
# ----------------------------
@router.get("/queryhistory/{project_id}")
def query_history(project_id: str, current_user: dict = Depends(get_current_user)):
    """
    Fetch all queries (conversation history) for the admin/viewing purpose.
    This allows users to see past conversations.
    """
    conn = get_connection()
    c = conn.cursor()
    user_id = current_user["user_id"]
    try:
        # Fetch all queries history from the database for the specific user
        c.execute(
           """
            SELECT id, conversation_id, project_id, query, answer, user_id, response_time, created_at
            FROM queries
            WHERE user_id = ? AND project_id = ?
            ORDER BY created_at DESC
            """,
            (user_id, project_id),
        )
        rows = c.fetchall()

        conversations = [
            {
                "id": row[0],
                "conversation_id": row[1],
                "project_id": row[2],
                "query": row[3],
                "answer": row[4],
                "user_id": row[5],
                "response_time": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

        return conversations

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching conversations: {e}"
        )
    finally:
        conn.close()
