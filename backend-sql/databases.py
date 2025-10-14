# databases.py
import os
import sqlite3
import pickle
from pathlib import Path
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "app_data.db")
STORE_ROOT = os.path.join(BASE_DIR, "stores")
os.makedirs(STORE_ROOT, exist_ok=True)

# ----------------------------
# SQLite utilities
# ----------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id TEXT PRIMARY KEY,
        conversation_id TEXT,
        project_id INTEGER,
        query TEXT,
        answer TEXT,
        user_id TEXT,
        response_time FLOAT,
        attempts INTEGER DEFAULT 1,
        is_fallback BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        precision1 REAL,
        precision3 REAL,
        precision5 REAL,
        claim_support REAL,
        query_coverage REAL,
        num_claims INTEGER
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_by TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS user_project (
        user_id TEXT,
        project_id INTEGER,
        role TEXT DEFAULT 'user',
        PRIMARY KEY(user_id, project_id)
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        name TEXT,
        path TEXT,
        type TEXT,
        status TEXT DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        uploaded_by TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS doc_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER,
        chunk_type TEXT,
        content TEXT,
        metadata TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        query_id TEXT,
        project_id INTEGER,
        query TEXT,
        answer TEXT,
        contexts TEXT,
        feedback TEXT,
        comment TEXT,
        user_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

# ----------------------------
# FAISS per-project cache
# ----------------------------
class ProjectIndex:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

project_cache: dict[int, ProjectIndex] = {}

def project_dir(project_id: int) -> str:
    d = os.path.join(STORE_ROOT, str(project_id))
    os.makedirs(d, exist_ok=True)
    return d

def load_project_index(project_id: int) -> ProjectIndex:
    if project_id in project_cache:
        return project_cache[project_id]
    d = project_dir(project_id)
    vs_path = os.path.join(d, "faiss_store.pkl")
    vectorstore = None
    if os.path.exists(vs_path):
        try:
            vectorstore = pickle.load(open(vs_path, "rb"))
        except Exception as e:
            print(f"Failed to load persisted vectorstore: {e}")
    pi = ProjectIndex(vectorstore=vectorstore)
    project_cache[project_id] = pi
    return pi

def persist_project_index(project_id: int):
    pi = project_cache.get(project_id)
    if not pi or not pi.vectorstore:
        return
    d = project_dir(project_id)
    try:
        pickle.dump(pi.vectorstore, open(os.path.join(d, "faiss_store.pkl"), "wb"))
    except Exception as e:
        print(f"Error persisting vectorstore {project_id}: {e}")
