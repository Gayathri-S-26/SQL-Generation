# gemini_chat_feedback.py

# Query and Retrieval

import os
import json
from typing import List, Literal, Optional, Union
import uuid
import time
import random
import logging
import re
from fastapi import APIRouter, Depends, HTTPException
import numpy as np
from pydantic import BaseModel
from databases import get_connection, load_project_index
from split_embeddings import rebuild_project_index_from_store
from langchain.schema import Document
from google import genai
from auth import get_current_user
from rank_bm25 import BM25Okapi
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
def call_gemini(prompt: str, retries: int = 3, base_delay: int = 2) -> str:
    """
    Call Gemini API with automatic retries on transient errors (e.g., 503).
    Retries use exponential backoff with jitter to avoid hammering the API.
    """
    attempt_count = 0
    for attempt in range(1, retries + 2):  # retries + initial attempt
        attempt_count += 1
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return getattr(response, "text", None) or getattr(response, "content", None) or str(response), attempt_count, False

        except Exception as e:
            err_msg = str(e)
            logging.error(f"[Gemini API Error] Attempt {attempt}: {err_msg}")

            # Retry only on transient errors
            if "503" in err_msg or "overloaded" in err_msg.lower():
                if attempt <= retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logging.warning(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    return "The assistant is busy right now. Please try again shortly.", attempt_count, True
            else:
                # Other errors → no retry, just return safe message
                return "Sorry, something went wrong while processing your request.", attempt_count, True

    return "The assistant is unavailable at the moment. Please try again later.", attempt_count, True

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
        if doc:
            faiss_docs.append(doc)
            faiss_scores.append(score)

    # ----------------------------
    # Step 3: BM25 retrieval
    # ----------------------------
    all_docs = list(pi.vectorstore.docstore._dict.values())
    tokenized_corpus = [d.page_content.split() for d in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # ----------------------------
    # Step 4: Normalize scores
    # ----------------------------
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    faiss_scores_norm = normalize(faiss_scores)
    bm25_scores_norm = normalize(bm25_scores)

    # ----------------------------
    # Step 5: Fuse scores with relational awareness
    # ----------------------------
    final_scores = {}

    def add_doc(doc, score):
        key = doc.page_content
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

    unique_docs = []
    seen = set()
    for content, _ in ranked:
        if content not in seen:
            for d in all_docs:
                if d.page_content == content:
                    unique_docs.append(d)
                    break
            seen.add(content)

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
    5. SQL queries should be in copy-friendly format.
    6. When explaining the query:
    - generate an ER diagram-like ASCII structure:
    - Each table in its own box (┌─┐ │ └─┘)
    - PKs with "PK:" prefix, FKs with "[FK]"
    - Show joins with ◄────► arrows
    - Place aggregation/filter/GROUP BY/HAVING/ORDER BY/other conditions boxes separately and join them.
    - Keep it compact and aligned, like a schema diagram.
    7. Use proper indentation for readability.
    """

    # Step 5: Call Gemini API and return the answer
    final_answer, attempt_count, is_fallback= call_gemini(structured_prompt)
    return final_answer, contexts, attempt_count, is_fallback

    

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
    memory.append({"role": "user", "content": req.query})

    # Build prompt for LLM
    chat_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in memory])
    prompt = f"{chat_history_str}\nAssistant:"

    # Call Query Workflow for retrieval and calling GEMINI LLM
    final_answer, contexts, attempt_count, is_fallback = query_workflow(req.project_id, prompt)
    
    # Response Time of LLM
    response_time = time.time() - start_time
    
    # Append assistant response to memory
    memory.append({"role": "assistant", "content": final_answer})

    # Truncate memory
    if len(memory) > MAX_MEMORY_LENGTH:
        conversation_memory[conversation_id] = memory[-MAX_MEMORY_LENGTH:]

    # Persist query
    query_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO queries (id, project_id, query, answer, user_id, conversation_id, response_time, attempts, is_fallback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (query_id, req.project_id, req.query, final_answer, user_id, conversation_id, response_time, attempt_count, is_fallback))
    conn.commit()
    conn.close()
    
    return {
        "answer": final_answer,
        "contexts": contexts,
        "query_id": query_id,
        "conversation_id": conversation_id,
        "history": memory
    }


# ----------------------------
# Feedback model
# ----------------------------
class Feedback(BaseModel):
    query_id: str
    project_id: int
    query: str = ""
    answer: str
    contexts: list[Union[str, dict]] = []
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
