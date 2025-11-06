# gemini_chat_feedback.py
import os
import json
from typing import Any, List, Literal, Optional, Union, Dict, TypedDict, Annotated
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

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

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
# State Definition for LangGraph
# ----------------------------
class AgentState(TypedDict, total=False):
    project_id: int
    user_query: str
    conversation_id: str
    user_id: str
    memory: List[Dict[str, str]]
    query_intent: str
    retrieval_mode: str
    required_documents: List[str]
    next_steps: List[str]
    complexity: str
    keyword_search_words: List[str]
    contexts: List[Dict]
    generated_response: str
    attempt_count: int
    is_fallback: bool
    current_step: str
    response_type: str  
    generation_prompt: str
    rag_metrics: Dict[str, Any]
    validation_result: str
    needs_improvement: bool
    retry_target: str  # "retrieval", "generation", or "none"
    improvement_suggestions: List[str]
    reflection_count: int
    stream_chunks: Optional[Any]
    current_status: str

# ----------------------------
# Gemini API call (Unchanged)
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

def call_llm(prompt: str, retries: int = 3, base_delay: int = 2) -> str:
    """
    Simple synchronous Gemini API call for planning and retrieval agents.
    Returns only the text response.
    """
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    
    for model in models:
        for attempt in range(1, retries + 2):
            try:
                response = genai_client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=GenerateContentConfig(temperature=0.1)  # Lower temp for planning/retrieval
                )
                
                # Extract text from response
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                
                # Fallback to string conversion
                return str(response)
                
            except Exception as e:
                err_msg = str(e)
                logging.error(f"[Gemini API Error] Model={model} Attempt={attempt}: {err_msg}")
                if "503" in err_msg or "overloaded" in err_msg.lower():
                    if attempt <= retries:
                        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        time.sleep(delay)
                        continue
                    else:
                        break
                else:
                    break

    return "⚠️ All Gemini models are busy. Please try again later."

def extract_doc_names_from_query(project_id: int, query: str) -> List[str]:
    """
    Detect document names mentioned in the user query with flexible matching.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM documents WHERE project_id=?", (project_id,))
    doc_names = [r[0] for r in c.fetchall()]
    conn.close()
    
    if not doc_names:
        return []
    
    # Normalize query for better matching
    query_lower = query.lower()
    
    mentioned = []
    for doc_name in doc_names:
        doc_name_lower = doc_name.lower()
        
        # Multiple matching strategies:
        # 1. Exact word match (original behavior)
        if re.search(rf"\b{re.escape(doc_name_lower)}\b", query_lower):
            mentioned.append(doc_name)
            continue
            
        # 2. Partial match (handles extensions, spaces, etc.)
        # Remove file extensions for matching
        doc_base = re.sub(r'\.(pdf|docx?|txt|csv|xlsx?)$', '', doc_name_lower)
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)  # Replace punctuation with spaces
        
        if doc_base in query_clean.split():
            mentioned.append(doc_name)
            continue
            
        # 3. Handle cases like "sample.pdf" → "sample"
        if re.search(rf"\b{re.escape(doc_base)}\b", query_lower):
            mentioned.append(doc_name)
    
    print(f"🔍 Document extraction: query='{query}' → found={mentioned}")
    return mentioned


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

def retrieve_relevant_chunks(project_id: int, query: str, keyword_terms: List[str], threshold: float = 0.7, max_candidates: int = 50, alpha: float = 0.5) -> List[Document]:
    """
    Hybrid retrieval using FAISS (semantic) + BM25 (keyword) with relational awareness.
    """
    def normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()
    
    pi = load_project_index(project_id)
    if not pi.vectorstore:
        return []

    # Step 1: Embed query for FAISS
    query_embedding = pi.vectorstore.embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    # Step 2: FAISS retrieval
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

    # Step 3: BM25 retrieval
    all_docs = [d for d in pi.vectorstore.docstore._dict.values() if getattr(d, "page_content", "").strip()]
    if keyword_terms:
        # Use only the extracted keywords for BM25
        keyword_query = " ".join(keyword_terms)
        print(f"🔍 BM25 using ONLY keywords: {keyword_terms}")
        
        tokenized_corpus = [d.page_content.split() for d in all_docs]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        bm25_scores = bm25.get_scores(keyword_query.split()) if bm25 else [0] * len(all_docs)
    else:
        tokenized_corpus = [d.page_content.split() for d in all_docs]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        bm25_scores = bm25.get_scores(query.split()) if bm25 else [0] * len(all_docs)

    # Step 4: Normalize scores
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    faiss_scores_norm = normalize(np.array(faiss_scores)) if faiss_scores else np.array([])
    bm25_scores_norm = normalize(np.array(bm25_scores)) if bm25_scores is not None and bm25_scores.size > 0 else np.array([])

    # Step 5: Fuse scores with relational awareness
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
                    add_doc(d, alpha * f_score * 0.8)

        # Add sequential neighbors
        prev_id = doc.metadata.get("prev_chunk_id")
        next_id = doc.metadata.get("next_chunk_id")
        for neighbor in all_docs:
            if neighbor.metadata.get("chunk_id") in [prev_id, next_id]:
                add_doc(neighbor, alpha * f_score * 0.6)

    # BM25 scores
    for doc, b_score in zip(all_docs, bm25_scores_norm):
        add_doc(doc, (1 - alpha) * b_score)

    # Step 6: Rank & Deduplicate
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

def evaluate_query_rag(project_id, user_query, sql_gen, docs):
    import sqlglot
    import re

    # Ensure docs is a list
    if not isinstance(docs, list):
        docs = [docs]

    # Normalize all docs to dicts with 'content'
    normalized_docs = []
    for d in docs:
        if hasattr(d, "page_content"):
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
            blocks = re.findall(r"(SELECT .*?;)", text, flags=re.DOTALL | re.IGNORECASE)
        return [b.strip() for b in blocks]

    # --- Extract SQL claims ---
    def extract_sql_claims(sql_text):
        import re
        import sqlglot
        from sqlglot.expressions import Table, Column, Join

        claims = []
        trees = []

        statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

        for stmt in statements:
            try:
                parsed_trees = sqlglot.parse(stmt)
            except Exception as e:
                claims.append(f"parse_error:{e}")
                continue

            for tree in parsed_trees:
                if tree is None:
                    continue
                trees.append(tree)

                for t in tree.find_all(Table):
                    if t.name:
                        claims.append(f"table:{t.name.lower()}")

                for c in tree.find_all(Column):
                    if c.name:
                        col_name = c.name.lower().split(".")[-1]
                        claims.append(f"column:{col_name}")

                for j in tree.find_all(Join):
                    try:
                        on_expr = j.on() if callable(j.on) else j.on
                        if on_expr:
                            join_clause = re.sub(r"\s+", " ", on_expr.sql().lower()).strip()
                            claims.append(f"join:{join_clause}")
                    except Exception:
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


    # --- Debug ---
    print("Top 3 chunk texts:", [d["content"][:100] for d in normalized_docs[:3]])
    print("Extracted SQL claims:", all_claims)

    if all_claims and all_trees:
        precision1 = precision_at_k(normalized_docs, all_claims, 1)
        precision3 = precision_at_k(normalized_docs, all_claims, 3)
        precision5 = precision_at_k(normalized_docs, all_claims, 5)
    else:
        precision1 = precision3 = precision5 = None

    return {
        "Precision@1": precision1,
        "Precision@3": precision3,
        "Precision@5": precision5,
        "NumClaims": len(all_claims)
    }
    
def get_project_documents(project_id: int) -> list:
    """Return a list of document names for the given project_id."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT name FROM documents WHERE project_id=?", (project_id,))
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows] if rows else []
    except Exception as e:
        print(f"⚠️ Error fetching project documents: {e}")
        return []

# ----------------------------
# Agent Functions
# ----------------------------
def planning_agent(state: AgentState) -> AgentState:
    """Analyze user query and plan the approach."""
    print("🔍 Planning Agent: Analyzing query intent...")
    prompt = f"""
    Analyze the user query and provide structured reasoning about how to handle it.

    USER QUERY: {state['user_query']}
    CONVERSATION HISTORY: {state['memory'][-20:] if state['memory'] else 'No history'}

    Analyze and return ONLY a JSON with these exact fields:
    - query_intent: "conversational", "sql_query"
    
    CRITICAL GUIDELINES:
    - If the query asks for data, information, analysis, reports, or anything that would require looking up data → "sql_query"
    - If the query mentions tables, columns, databases, schemas, SQL, queries, data, records → "sql_query"  
    - If the query asks "show me", "list", "find", "get", "how many", "what is the" followed by data-related terms which need retrieval → "sql_query"
    - Only use "conversational" for greetings, thanks, casual chat, or questions about the assistant itself which does not need retrieval.

    For greetings/conversational (like "hello", "thanks", "how are you"), use:
    {{
        "query_intent": "conversational",
        "next_steps": ["generation"],
    }}
    For SQL Query (Like "Generate query to validate sales data"), use:
    {{
        "query_intent": "sql_query",
        "next_steps": ["retrieval"],
    }}

    Return ONLY valid JSON, no other text.
    """
    
    try:
        # Use non-streaming call for planning
        result = call_llm(prompt)
        
        # Handle the return format properly
        if isinstance(result, tuple) and len(result) == 3:
            response_text = result[0]  # Extract the text from tuple (text, attempt_count, is_fallback)
        else:
            # Fallback if call_gemini returns something unexpected
            response_text = str(result) if result else '{"query_intent": "unknown", "next_steps": ["retrieval"]}'
        
        # Clean the response - remove any markdown formatting
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            planning_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Extract JSON from malformed response or use fallback
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                planning_data = json.loads(json_match.group())
            else:
                planning_data = {"query_intent": "unknown"}
        
        state.update({
            "query_intent": planning_data.get("query_intent", "unknown"),
            "next_steps": planning_data.get("next_steps", []),
            "current_step": "planning_complete",
        })
        
        print(f"📋 Planning Result: {state['query_intent']}, Steps: {state['next_steps']}")
        
    except Exception as e:
        print(f"❌ Planning agent error: {e}")
        print(f"Raw response was: {response_text if 'response_text' in locals() else 'No response'}")
        
        # Default fallback based on query content
        user_query_lower = state['user_query'].lower()
        conversational_keywords = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye', 'how are you']
        
        if any(keyword in user_query_lower for keyword in conversational_keywords):
            state.update({
                "query_intent": "conversational",
                "next_steps": ["generation"],
                "current_step": "planning_complete",
            })
        else:
            state.update({
                "query_intent": "sql_query", 
                "next_steps": ["retrieval"],
                "current_step": "planning_complete",
            })
        
        print(f"🔄 Using fallback planning: {state['query_intent']}")
    
    return state

# ----------------------------
# Retrieval Agent
# ----------------------------
def retrieval_agent(state: AgentState) -> AgentState:
    """Use LLM to analyze retrieval needs and fetch relevant documents."""
    print("🔍 Retrieval Agent: Analyzing retrieval needs...")
    
    # 🧩 Feedback-aware logic
    validation_feedback = state.get("rag_metrics", {}).get("LLMReasoning", "")
    validation_suggestions = state.get("improvement_suggestions", [])
    retry_target = state.get("retry_target", "none")
    reflection_count = state.get("reflection_count", 0)

    # Include validation insights if this is a retry after validation
    feedback_section = ""
    if retry_target == "retrieval" and (validation_feedback or validation_suggestions):
        feedback_section = f"""
        The previous validation step suggested retrieval improvement.
        REASONING: {validation_feedback}
        SUGGESTIONS: {', '.join(validation_suggestions) if validation_suggestions else 'None'}
        
        Based on this feedback, refine your retrieval analysis to better target the relevant information.
        """
        print("🧠 Applying validation feedback to refine retrieval...")
    
    
    prompt = f"""
    Analyze the user query for document retrieval needs. Return ONLY JSON with these exact fields:
    - retrieval_mode: "document_level" if user requests SQL or analysis for entire document, otherwise "chunk_level"
    - required_documents: list of specific document names mentioned or implied
    - keyword_search_words: list of key search terms for semantic search  
    - complexity: "low", "medium", or "high" based on query complexity

    USER QUERY: {state['user_query']}
    
    {feedback_section}

    Guidelines:
    
🔹 RETRIEVAL_MODE:
- "document_level" → when the query explicitly mentions full document processing or any document name mentioned with extension
   e.g., "generate SQL for whole document X", "analyze complete file Y", "use entire data model".
- "chunk_level" → when query targets specific info, e.g., "find Q4 sales", "get revenue of last year".

🔹 REQUIRED_DOCUMENTS:
- Extract ONLY explicit document names directly mentioned like pdf,doc,docx,csv,txt,xlsx file names.
- Must match exact document names from the database
- If no specific documents mentioned → empty list []
- Examples: 
  "Generate sql from document.csv" → ["document.csv"]
  "Analyze the document sample.docx" → ["sample.docx"]

🔹 KEYWORD_SEARCH_WORDS:
- Extract all MOST SPECIFIC nouns/entities from the query
- Focus on concrete business entities, not generic terms
- Skip verbs, adjectives, and generic words like "data", "information", "details"
- Examples:
  "info about colleges" → ["colleges"]
  "sales report for Q4" → ["sales", "Q4"]
  "customer orders last month" → ["customers", "orders"]
  "analyze revenue trends" → ["revenue", "trends"]

 🔹Complexity: "low" for simple lookups, "medium" for joins, "high" for complex analytics

    Return ONLY valid JSON, no other text.
    """
    
    try:
        actual_doc_names = get_project_documents(state['project_id'])
        if not actual_doc_names:
            print("⚠️ No documents found in this project. Skipping retrieval.")
            
            state.update({
                "contexts": [],
                "retrieval_mode": "none",
                "required_documents": [],
                "keyword_search_words": [],
                "complexity": "none",
                "retrieval_metrics": {
                    "documents_found": 0,
                    "keywords_used": 0,
                    "chunks_retrieved": 0,
                    "complexity_level": "none"
                },
                "current_step": "retrieval_complete",
                "retry_target": None
            })
            
            return state
        # Get LLM analysis for retrieval parameters
        result = call_llm(prompt)
        
        # Handle response format
        if isinstance(result, tuple) and len(result) == 3:
            response_text = result[0]
        else:
            response_text = str(result) if result else '{"required_documents": [], "keyword_search_words": [], "complexity": "medium"}'
        
        # Clean the response
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        retrieval_plan = json.loads(response_text)
        
        # Extract the LLM reasoning results
        retrieval_mode = retrieval_plan.get("retrieval_mode","chunk_level")
        required_documents = retrieval_plan.get("required_documents", [])
        keyword_search_words = retrieval_plan.get("keyword_search_words", [])
        complexity = retrieval_plan.get("complexity", "medium")
        
        actual_doc_names = extract_doc_names_from_query(state['project_id'], state['user_query'])
        required_documents = [doc for doc in required_documents if doc in actual_doc_names]
        
        print(f"📋 Retrieval Plan:  Mode={retrieval_mode}, Docs={required_documents}, Keywords={keyword_search_words}, Complexity={complexity}")
        
        # Use existing logic to retrieve documents based on LLM analysis
        contexts = []
        
        # 1. Retrieve the whole document if user mentioned specific documents for the entire content
        if retrieval_mode == "document_level" and required_documents:
            print(f"📄 Full-document retrieval mode selected for: {required_documents}")
            for doc_name in required_documents:
                doc_chunks = retrieve_from_doc_by_name(state['project_id'], doc_name)
                contexts.extend(doc_chunks)
        
        
        # 2. Adjust retrieval parameters based on complexity
        if complexity == "high":
            threshold = 0.6  # Stricter similarity for complex queries
            max_candidates = 100  # More candidates for complex queries
        elif complexity == "low":
            threshold = 0.8  # Higher threshold for simple queries
            max_candidates = 20   # Fewer candidates for simple queries
        else:  # medium
            threshold = 0.7
            max_candidates = 50
        
        # 3. Retrieve project-wide relevant chunks using existing logic
        project_chunks = retrieve_relevant_chunks(
            state['project_id'], 
            state['user_query'],
            keyword_terms=keyword_search_words, 
            threshold=threshold, 
            max_candidates=max_candidates
        )
        contexts.extend(project_chunks)
        
        if not contexts:
            print("⚠️ Retrieval completed but no matching chunks found.")
            state.update({
                "contexts": [],
                "retrieval_mode": retrieval_mode,
                "required_documents": required_documents,
                "keyword_search_words": keyword_search_words,
                "complexity": complexity,
                "retrieval_metrics": {
                    "documents_found": len(required_documents),
                    "keywords_used": len(keyword_search_words),
                    "chunks_retrieved": 0,
                    "complexity_level": complexity
                },
                "current_step": "retrieval_complete"
            })
            return state
        
        # Remove duplicates using existing logic
        seen_texts = set()
        unique_contexts = []
        for doc in contexts:
            if doc.page_content not in seen_texts:
                unique_contexts.append(doc)
                seen_texts.add(doc.page_content)
        
        # Convert to context format
        context_json = []
        for chunk in unique_contexts:
            context_json.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("filename", "unknown"),
                "section": chunk.metadata.get("hierarchy", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0)
            })
        
        # Update state with retrieval results
        state.update({
            "retrieval_mode": retrieval_mode,
            "required_documents": required_documents,
            "keyword_search_words": keyword_search_words, 
            "complexity": complexity,
            "contexts": context_json,
            "retrieval_metrics": {
                "documents_found": len(required_documents),
                "keywords_used": len(keyword_search_words),
                "chunks_retrieved": len(context_json),
                "complexity_level": complexity
            },
            "current_step": "retrieval_complete"
        })
        
        print(f"✅ Retrieval complete: {len(context_json)} chunks retrieved")
        
    except Exception as e:
        print(f"❌ Retrieval agent error: {e}")
        # Fallback to basic retrieval without LLM analysis
        print("🔄 Using fallback retrieval...")
        
        # Use existing basic retrieval as fallback
        project_chunks = retrieve_relevant_chunks(state['project_id'], state['user_query'])
        
        # Convert to context format
        context_json = []
        for chunk in project_chunks:
            context_json.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("filename", "unknown"),
                "section": chunk.metadata.get("hierarchy", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0)
            })
        
        state.update({
            "retrieval_mode": "chunk_level",
            "required_documents": [],
            "keyword_search_words": [],
            "complexity": "medium", 
            "contexts": context_json,
            "current_step": "retrieval_complete"
        })
    
    return state

# ----------------------------
# Generation Agent
# ----------------------------

# ----------------------------
# Generation Agent
# ----------------------------

def generation_agent(state: AgentState) -> AgentState:
    """Generate response based on query intent and available contexts."""
    print("🤖 Generation Agent: Generating response...")
    
    try:
        validation_feedback = state.get("rag_metrics", {}).get("LLMReasoning", "")
        validation_suggestions = state.get("improvement_suggestions", [])
        retry_target = state.get("retry_target", "none")
        reflection_count = state.get("reflection_count", 0)
        
        retrieval_mode = state.get("retrieval_mode", "unknown")
        retrieval_metrics = state.get("retrieval_metrics", {})
        documents_found = retrieval_metrics.get("documents_found", 0)
        no_docs_present = (retrieval_mode == "none" or documents_found == 0)

        # Include validation insights if this is a retry after validation
        feedback_section = ""
        if retry_target == "generation" and (validation_feedback or validation_suggestions):
            feedback_section = f"""
            The previous validation step suggested generation improvement.
            REASONING: {validation_feedback}
            SUGGESTIONS: {', '.join(validation_suggestions) if validation_suggestions else 'None'}
            
            Based on this feedback, refine your generation.
            """
            print("🧠 Applying validation feedback to refine generation...")
                
                
        if state.get("query_intent") == "conversational":
            # Handle conversational queries (non-streaming for simplicity)
            print("💬 Generating conversational response...")
            conversational_prompt = f"""
            You are a friendly AI assistant. Respond naturally to the user's message.
            Always stay within the conversation topic or SQL/data-related context.
            Never fabricate information, speculate, or respond to out-of-scope topics (e.g., personal opinions, politics, weather, jokes).
            If the user asks something out of scope, politely decline and redirect back to SQL or data-related help.
            
            USER MESSAGE: {state['user_query']}
            CONVERSATION HISTORY: {state['memory'][-20:] if state['memory'] else 'No recent history'}
            
            {feedback_section}
            Respond in a friendly, helpful manner. Keep it concise and natural.
            """
            
            result = call_llm(conversational_prompt)
            if isinstance(result, tuple) and len(result) == 3:
                response_text = result[0]
            else:
                response_text = str(result) if result else "I'm here to help! How can I assist you today?"
            
            state.update({
                "generated_response": response_text,
                "response_type": "conversational",
                "current_step": "generation_complete"
            })
            
        else:
            # Handle SQL queries with contexts - prepare for streaming
            print("📊 Generating SQL response with contexts...")
            
            # Prepare contexts for the prompt
            contexts = state.get("contexts", [])
            final_json = json.dumps(contexts, separators=(",", ":")) if contexts else "[]"
            
            # Use your existing structured prompt
            structured_prompt = f"""
            # ROLE: Expert SQL Assistant
            # CONTEXT: You are analyzing documentation to generate precise SQL queries.
            ---
            
            {feedback_section}
            
            CONVERSATION HISTORY: {state['memory'][-20:] if state['memory'] else 'No recent history'}

            # AVAILABLE DOCUMENTATION:
            {final_json}

            ---

            # USER QUERY:
            "{state['user_query']}"
            
            # PROJECT DOCUMENT STATUS:
            {"No documents are available for this project." if no_docs_present else "Some documents were available for retrieval."}

            ---

            # INSTRUCTIONS:

            ## 1. QUERY ANALYSIS & INTENT
            - Analyze the user's question to understand what data is being requested. 
               -- **SQL_QUERY** → The user is asking for a data query, report, metric, aggregation, or data relationship. 
               -- **GENERAL_QUESTION** → The user is asking for an explanation, concept, or descriptive answer — not a SQL request.
               -- **OUT_OF_SCOPE** → The question is unrelated to data, SQL, or the provided documentation (e.g., weather, jokes, small talk). 
            - Identify the key **entities**, **metrics**, **filters**, and **relationships** involved.  
            - Determine whether this is a **simple lookup**, **aggregation**, **join**, or **complex analytical query**.
            ---
            ## 2. SCHEMA UNDERSTANDING (From Documentation)
            **CRITICAL:** Carefully extract and verify the following information from the provided context:
            - Tables available and their purposes.  
            - Columns with exact names and data types.  
            - Primary Keys (`PK:`) and Foreign Keys (`FK:`).  
            - Relationships between tables.  
            - Business rules or constraints mentioned in the documentation.  
            - If the context is insufficient, apply standard SQL design patterns reasonably.

            ---

            ## 3. SQL GENERATION REQUIREMENTS
            -- Generate a syntactically correct SQL query.
            -- Use exact table and column names from the documentation.
            -- Include proper JOIN conditions using PK/FK relationships.
            -- Apply relevant WHERE clauses, aggregations, and ordering.
            -- **CRITICAL:** Always generate a performance-optimized and cost-efficient SQL query.

            -- Formatting Guidelines:
            --   • Use clear indentation and line breaks.
            --   • Add comments for any complex logic.
            --   • Ensure the query directly answers the user's question.

            ## 4. EXPLANATION & SCHEMA DIAGRAM
            - Generate an ER diagram-like ASCII structure in separate copy-friendly format:
            - Each table in its own box (┌─┐ │ └─┘)
            - PKs with "PK:" prefix, FKs with "[FK]"
            - Show joins with ◄────► arrows
            - Place aggregation/filter/GROUP BY/HAVING/ORDER BY/other conditions boxes separately and join them.
            - Keep it compact and aligned, like a schema diagram.

            ## 5. OUTPUT FORMAT — ADAPTIVE RESPONSE GENERATION
            
            ### FORMATTING REQUIREMENTS:
            - Use CLEAR SECTION BREAKS with blank lines between major sections
            - Separate different points with line breaks for better readability
            - Use bullet points with proper spacing between each item
            - Add blank lines before and after code blocks
            - Ensure paragraphs have proper spacing for easy reading
            
            ### CASE 1: If intent == "SQL_QUERY"
            Follow this full, structured output:
            Provide a brief natural-language summary that directly answers the user's question.

            **SQL Query**
            SQL queries should be in copy-friendly format within "```sql ```".
            ```sql SELECT * FROM ... ```

            **Explanation Diagram**
            ASCII diagram showing tables, relationships, and key operations

            **Notes**
            * List any assumptions made.
            * Mention limitations or potential ambiguities.
            * Suggest further analysis if applicable.
            
            ### CASE 2: If intent == "GENERAL_QUESTION"

            * Do NOT include SQL, diagrams, or “Notes” placeholders.
            * Provide a concise and factual explanation directly answering the question.
            * If relevant, refer to documentation or general SQL best practices.
            * If some info is missing, politely state what’s unavailable and suggest next steps.
            
            ### CASE 3: If intent == "OUT_OF_SCOPE"

            * Acknowledge politely that the question is outside SQL or documentation scope.
            * Do NOT generate any SQL or technical placeholders.
            * Respond conversationally, in a friendly and helpful tone.
            * Optionally offer to switch back to SQL/data assistance.

            ## 6. CONVERSATIONAL HANDLING

            - If the input is a greeting, thank-you, or general conversation:

            - Respond naturally and politely.

            - Maintain a professional yet friendly tone.

            - Optionally offer SQL or data assistance if relevant.

            - Keep responses concise and engaging.
            """
            
            ## Stream the response and collect chunks
            print("🔄 Streaming response from LLM...")
            stream_chunks = []
            full_response = ""
            attempt_count = 0
            is_fallback = False
            for chunk in call_gemini(structured_prompt, stream=True):
                if isinstance(chunk, dict) and "text" in chunk:
                    text_chunk = chunk["text"]
                    state["attempt_count"] = chunk["attempt_count"]
                    state["is_fallback"] = chunk["is_fallback"]
                    full_response += text_chunk
                    stream_chunks.append(text_chunk)
                    print(f"📦 Collected chunk: {text_chunk[:50]}...")
                else:
                    print(f"⚠️ Unexpected chunk format: {type(chunk)}")
            
            print(f"✅ Collected {len(stream_chunks)} chunks, total length: {len(full_response)}")
            
            # Store everything in state
            state.update({
                "generation_prompt": structured_prompt,
                "generated_response": full_response,
                "stream_chunks": stream_chunks,
                "response_type": "sql",
                "contexts_used": contexts,
                "current_step": "generation_complete"
            })
        
        print(f"✅ Generation complete. Type: {state.get('response_type', 'unknown')}")
        return state
        
        
    except Exception as e:
        print(f"❌ Generation agent error: {e}")
        # Fallback response
        state.update({
            "generated_response": "I apologize, but I encountered an error while generating the response. Please try again.",
            "response_type": "error",
            "current_step": "generation_complete"
        })
    
    return state


# ----------------------------
# Validation Agent
# ----------------------------
def validation_agent(state: AgentState) -> AgentState:
    """Evaluate the RAG response quality and provide improvement suggestions."""
    print("🔍 Validation Agent: Evaluating response quality...")
    
    try:
        # Only validate SQL responses that have been generated
        if state.get("response_type") == "sql" and state.get("generated_response"):
            retrieval_mode = state.get("retrieval_mode", "unknown")
            retrieval_metrics = state.get("retrieval_metrics", {})
            documents_found = retrieval_metrics.get("documents_found", 0)
            no_docs_present = (retrieval_mode == "none" or documents_found == 0)
            generated_response = state["generated_response"]
            contexts = state.get("contexts", [])
            user_query = state["user_query"]
            project_id = state["project_id"]
            
            # Run basic RAG evaluation (only reliable rule-based metrics)
            rag_metrics = evaluate_query_rag(project_id, user_query, generated_response, contexts)
            
            # Add LLM reasoning for problematic metrics
            llm_eval_prompt = f"""
            As a SQL and data analysis expert, evaluate this RAG system output.

            USER QUERY: "{user_query}"
            
            GENERATED RESPONSE:
            {generated_response}
            
            AVAILABLE CONTEXTS:
            {(contexts)}
            
            PROJECT DOCUMENT STATUS:
            {"No documents are available for this project. Retrieval was skipped." if no_docs_present else "Some documents were available for retrieval."}
            
            RETRIEVAL METRICS (Rule-based):
            - Precision@1: {rag_metrics.get('Precision@1', 'N/A')}
            - Precision@3: {rag_metrics.get('Precision@3', 'N/A')}
            - Precision@5: {rag_metrics.get('Precision@5', 'N/A')}
            - SQL Elements Found: {rag_metrics.get('NumClaims', 'N/A')}
            
            Evaluate these specific aspects:

            1. QUERY COVERAGE (0.0-1.0):
            - Does the generated response fully address the user's original query intent?
            - Consider both explicit requirements and implicit needs
            - Score: 1.0 = perfectly addresses, 0.0 = completely misses

            2. CLAIM SUPPORT RATE (0.0-1.0):
            - What percentage of SQL elements (tables, columns, joins, conditions) are supported by the available contexts?
            - Consider semantic support, not just exact string matches
            - Include synonyms and related terms
            - Score: 1.0 = fully supported, 0.0 = completely unsupported

            3. STATIC EXECUTION SCORE (0.0-1.0):
            - How correct and executable is the generated SQL?
            - Consider syntax, logic, and potential execution success
            - Score: 1.0 = perfectly executable, 0.0 = completely invalid
            
            4. RETRY DECISION ANALYSIS:
            - Analyze if the main issue is RETRIEVAL (bad contexts) or GENERATION (bad SQL)
            - RETRIEVAL issues: contexts missing key info, low precision scores
            - GENERATION issues: good contexts but poor SQL, syntax errors
            - If both bad but retrieval worse → RETRIEVAL
            - Default to RETRIEVAL if unclear
            - If NO DOCUMENTS are available for the Project, DO NOT suggest retrieval retry.

            Return ONLY JSON with these exact fields:
            - "QueryCoverage": number between 0.0-1.0
            - "ClaimSupportRate": number between 0.0-1.0  
            - "StaticExecScore": number between 0.0-1.0
            - "needs_improvement": true or false
            - "retry_target": "retrieval" or "generation" or "none"
            - "improvement_suggestions": ["list", "of", "specific", "suggestions"]
            - "reasoning": "Detailed analysis of scores and retry decision"

            Return ONLY valid JSON, no other text.
            """
            
            try:
                result = call_llm(llm_eval_prompt)
                response_text = result[0] if isinstance(result, tuple) else str(result)
                
                # Clean response
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                llm_analysis = json.loads(response_text)
                
                # Add LLM metrics to rag_metrics with same names
                rag_metrics.update({
                    "QueryCoverage": llm_analysis.get("QueryCoverage", 0.5),
                    "ClaimSupportRate": llm_analysis.get("ClaimSupportRate", 0.5),
                    "StaticExecScore": llm_analysis.get("StaticExecScore", 0.5),
                    "LLMReasoning": llm_analysis.get("reasoning", "No detailed reasoning"),
                    "needs_improvement": llm_analysis.get("needs_improvement", False),
                    "retry_target": llm_analysis.get("retry_target", "none"),
                    "improvement_suggestions": llm_analysis.get("improvement_suggestions", [])
                })
                
                print(f"🤖 LLM Analysis: Needs improvement = {rag_metrics['needs_improvement']}, Retry target = {rag_metrics['retry_target']}")
                
            except Exception as e:
                print(f"❌ LLM evaluation error: {e}")
                # Fallback scores for LLM metrics
                rag_metrics.update({
                    "QueryCoverage": None,
                    "ClaimSupportRate": None,
                    "StaticExecScore": None,
                    "LLMReasoning": f"LLM evaluation failed: {str(e)}",
                    "needs_improvement": True,  # Default to improvement needed on error
                    "retry_target": "retrieval",  # Default to retrieval retry
                    "improvement_suggestions": ["Check system configuration and try again"]
                })
            
            # Generate validation summary
            validation_summary = f"""
            ## RAG Quality Assessment

            ### 📊 Metrics Summary:
            - **Precision@1**: {rag_metrics['Precision@1'] or 'N/A'} (Retrieval Quality)
            - **Precision@3**: {rag_metrics['Precision@3'] or 'N/A'} (Retrieval Quality) 
            - **Precision@5**: {rag_metrics['Precision@5'] or 'N/A'} (Retrieval Quality)
            - **Claim Support Rate**: {rag_metrics['ClaimSupportRate'] or 'N/A'} (LLM: Context Support)
            - **Static Execution Score**: {rag_metrics['StaticExecScore'] or 'N/A'} (LLM: SQL Quality)
            - **Query Coverage**: {rag_metrics['QueryCoverage'] or 'N/A'} (LLM: Intent Match)
            - **SQL Elements Found**: {rag_metrics['NumClaims']} (Rule-based)

            ### 🧠 LLM Analysis:
            {rag_metrics.get('LLMReasoning', 'No analysis available')}

            ### 🔄 Retry Decision:
            - **Needs Improvement**: {rag_metrics.get('needs_improvement', False)}
            - **Retry Target**: {rag_metrics.get('retry_target', 'none')}
            - **Suggestions**: {', '.join(rag_metrics.get('improvement_suggestions', ['No specific suggestions']))}

            ### 🎯 Status: {'⚠️ Needs Improvement' if rag_metrics.get('needs_improvement', False) else '✅ Quality Acceptable'}
            """
            current_count = state.get("reflection_count")
            if state["needs_improvement"] and current_count < 2:
               current_count = current_count + 1
               state["reflection_count"]=current_count
               print(f"✅ Final reflection count: {current_count}")
            else:
                state["needs_improvement"] = False
                state["retry_target"] = None
                print(f"✅ Final reflection count: {current_count}")
            
            state.update({
                "rag_metrics": rag_metrics,
                "validation_result": validation_summary,
                "needs_improvement": rag_metrics.get('needs_improvement', False),
                "retry_target": rag_metrics.get('retry_target', 'none'),
                "improvement_suggestions": rag_metrics.get('improvement_suggestions', []),
                "current_step": "validation_complete"
            })
            
            print(f"✅ Hybrid validation complete. Needs improvement: {rag_metrics.get('needs_improvement', False)}")
            print(f"🎯 Retry target: {rag_metrics.get('retry_target', 'none')}")
            
        else:
            state.update({
                "rag_metrics": {},
                "validation_result": "Validation skipped - not a SQL response",
                "needs_improvement": False,
                "retry_target": "none",
                "improvement_suggestions": [],
                "current_step": "validation_complete"
            })
            print("⏭️  Validation skipped (not a SQL response)")
            
    except Exception as e:
        print(f"❌ Validation agent error: {e}")
        state.update({
            "rag_metrics": {},
            "validation_result": f"Validation failed: {str(e)}",
            "needs_improvement": True,
            "retry_target": "retrieval",
            "improvement_suggestions": ["System error occurred - retry recommended"],
            "current_step": "validation_complete"
        })
    
    return state


def update_status_before_node(state: AgentState, node_name: str):
    status_messages = {
        "planning": "🔍 Analyzing query...",
        "retrieval": "📋 Retrieving information...", 
        "generation": "🤖 Generating response...",
        "validation": "✅ Validating answer...",
        "retry_retrieval": "🔄 Improving retrieval...",
        "retry_generation": "🔄 Improving response..."
    }
    state["current_status"] = status_messages.get(node_name, "Processing...")
    return state

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# ----------------------------
# LangGraph Workflow Construction
# ----------------------------
def create_agent_workflow():
    """Create the LangGraph workflow with agentic decision making."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", lambda state: planning_agent(update_status_before_node(state, "planning")))
    workflow.add_node("retrieval", lambda state: retrieval_agent(update_status_before_node(state, "retrieval")))
    workflow.add_node("generation", lambda state: generation_agent(update_status_before_node(state, "generation")))
    workflow.add_node("validation", lambda state: validation_agent(update_status_before_node(state, "validation")))
    
    # Define workflow edges
    workflow.set_entry_point("planning")
    workflow.add_conditional_edges(
        "planning",
        # Lambda function that routes based on query_intent
        lambda state: "generation" if state.get("query_intent") == "conversational" else "retrieval",
        {
            "generation": "generation",  
            "retrieval": "retrieval",
        }
    )
    workflow.add_edge("retrieval", "generation")
    workflow.add_conditional_edges(
        "generation",
        lambda state: "validation" if state.get("response_type") == "sql" and state.get("current_step") == "generation_complete" else END,
        {
            "validation": "validation",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "validation",
        lambda state: (
            # Increment count and return retry target if conditions met
            (state.update({"reflection_count": state.get("reflection_count")}) or 
            print(f"🔄 Reflection count: {state.get('reflection_count')}/2") or
            state.get("retry_target")) 
            if (
                state.get("needs_improvement", False) and 
                state.get("retry_target") in ["retrieval", "generation"] and 
                state.get("reflection_count", 0) < 2
            ) 
            else END
        ),
        {
            "retrieval": "retrieval",
            "generation": "generation", 
            END: END
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)

# Create the workflow
agent_workflow = create_agent_workflow()




# ----------------------------
# Chat Model (Updated to use LangGraph)
# ----------------------------
class ChatRequest(BaseModel):
    project_id: int
    query: str
    conversation_id: Optional[str] = None
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
    
    # ✅ ADD DEBUG: Show full conversation history
    print(f"🧠 FULL Conversation History for {conversation_id}: {len(memory)} messages")
    for i, msg in enumerate(memory):
        print(f"  {i}: {msg['role']}: {msg['content'][:80]}...")

    memory.append({"role": "user", "content": safe_query})
    
    # Initialize state for LangGraph
    initial_state = AgentState(
        user_query=safe_query,
        project_id=req.project_id,
        conversation_id=conversation_id,
        user_id=user_id,  
        memory=memory,
        query_intent="",
        next_steps=[],
        retrieval_mode="",
        required_documents=[],
        complexity="",
        keyword_search_words=[],
        generated_response="",
        contexts=[],
        attempt_count=0,
        is_fallback=False,
        current_step="start",
        rag_metrics={},
        validation_result="",
        needs_improvement=False,
        retry_target="none",
        improvement_suggestions=[],
        reflection_count=0
    )
    config = {
        "configurable": {
            "thread_id": conversation_id  # ✅ Only needed
        }
    }
    
        
    
    # Store query in database initially
    query_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO queries (id, project_id, query, answer, user_id, conversation_id, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (query_id, req.project_id, safe_query, "", user_id, conversation_id, 0))
    conn.commit()
    conn.close()
    
    # Streaming response generator
    async def stream_and_save():
        # Send initial status
        yield "[[[STATUS]]]🚀 Starting processing..."
        
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        # Track previous status to avoid duplicates
        previous_status = ""
        
        # Stream status updates during workflow execution
        for current_state in agent_workflow.stream(initial_state, config=config):
            for node_name, state in current_state.items():
                if "current_status" in state and state["current_status"] != previous_status:
                    yield f"[[[STATUS]]]{state['current_status']}"
                    previous_status = state["current_status"]
                final_state = state
        
        workflow_state = final_state if final_state else agent_workflow.invoke(initial_state, config=config)
        
        # Get the final response
        full_answer = workflow_state.get("generated_response", "")
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Store assistant response in memory
        memory.append({"role": "assistant", "content": full_answer})
        
        # Truncate memory if needed
        if len(memory) > MAX_MEMORY_LENGTH:
            conversation_memory[conversation_id] = memory[-MAX_MEMORY_LENGTH:]
            
        rag_metrics = workflow_state.get("rag_metrics", {})
        
        # Update database with final answer
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE queries
            SET answer = ?, response_time = ?, attempts = ?, is_fallback = ?, 
                precision1 = ?, precision3 = ?, precision5 = ?, 
                claim_support = ?, static_exec = ?, query_coverage = ?, num_claims = ?
            WHERE id = ?
        """, (full_answer,
            response_time,
            workflow_state.get("attempt_count", 1),
            workflow_state.get("is_fallback", False),
            rag_metrics.get("Precision@1"),
            rag_metrics.get("Precision@3"),
            rag_metrics.get("Precision@5"),
            rag_metrics.get("ClaimSupportRate"),
            rag_metrics.get("StaticExecScore"),
            rag_metrics.get("QueryCoverage"),
            rag_metrics.get("NumClaims"),
            query_id))
        conn.commit()
        conn.close()
        
        # Send the COMPLETE response at the end (not in chunks)
        yield f"{full_answer}"
        
        # Send metadata at the end
        yield f"[[[META]]]{json.dumps({'contexts': workflow_state.get('contexts', []), 'query_id': query_id, 'conversation_id': conversation_id})}"
    
    return StreamingResponse(stream_and_save(), media_type="text/plain")

# ----------------------------
# Feedback model (Unchanged)
# ----------------------------
class Feedback(BaseModel):
    query_id: str
    project_id: int
    query: str = ""
    answer: str
    contexts: Any = []
    feedback: Literal["up", "down"]
    comment: str = ""
    user_id: str

# Load existing feedback
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r") as f:
        feedback_store = json.load(f)
else:
    feedback_store = []

@router.post("/feedback")
async def submit_feedback(item: Feedback, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        contexts = item.contexts
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)
            except json.JSONDecodeError:
                contexts = [contexts]
        elif contexts is None:
            contexts = []
        elif not isinstance(contexts, list):
            contexts = [contexts]

        item.contexts = contexts
        
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback_store = json.load(f)
        else:
            feedback_store = []

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
# Get all conversations/queries (Unchanged)
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
