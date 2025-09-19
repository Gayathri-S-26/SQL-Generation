# projects_documents.py
import os
import shutil
import json
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from databases import get_connection, load_project_index, persist_project_index, project_cache
from langchain.schema import Document
from split_embeddings import load_and_split_document, save_and_index, rebuild_project_index_from_store
from auth import get_current_user

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

router = APIRouter()

class ProjectRequest(BaseModel):
    project_name: str

@router.post("/project")
def create_or_get_project(req: ProjectRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM projects WHERE name=?", (req.project_name,))
    row = c.fetchone()
    if row:
        pid = row[0]
    else:
        c.execute("INSERT INTO projects (name, created_by) VALUES (?, ?)", (req.project_name, user_id))
        pid = c.lastrowid
        conn.commit()
        # Get the user's role from users table
        c.execute("SELECT role FROM users WHERE id=?", (user_id,))
        user_role_row = c.fetchone()
        user_role = user_role_row[0] if user_role_row else "user"

        # Add entry to user_project table
        c.execute(
            "INSERT OR IGNORE INTO user_project (user_id, project_id, role) VALUES (?, ?, ?)",
            (user_id, pid, user_role)
        )
        conn.commit()
    conn.close()
    load_project_index(pid)
    return {"project_id": pid, "project_name": req.project_name, "user_id": user_id}

@router.get("/projects")
def list_projects(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    if current_user["role"] == "admin":
        c.execute("SELECT id, name, created_at FROM projects ORDER BY created_at DESC")
    else:
        c.execute("""
            SELECT p.id, p.name, p.created_at
            FROM projects p
            JOIN user_project up ON p.id = up.project_id
            WHERE up.user_id = ?
            ORDER BY p.created_at DESC
        """, (current_user["user_id"],))
    rows = [{"project_id": r[0], "project_name": r[1], "created_at": r[2]} for r in c.fetchall()]
    conn.close()
    return rows

@router.get("/project/{project_id}")
def project_details(project_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, path, type, status, progress FROM documents WHERE project_id=?", (project_id,))
    docs = [{"doc_id": r[0], "doc_name": r[1], "path": r[2], "type": r[3], "status": r[4], "progress": r[5]} for r in c.fetchall()]
    conn.close()
    return {"project_id": project_id, "documents": docs}

@router.post("/upload")
async def upload_files(project_id: int = Query(...), files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn_main = get_connection()
    c_main = conn_main.cursor()
    c_main.execute("SELECT id FROM projects WHERE id=?", (project_id,))
    if not c_main.fetchone():
        conn_main.close()
        return JSONResponse(status_code=404, content={"status": "error", "message": "Project not found"})
    conn_main.close()

    results = []
    for file in files:
        filename = file.filename
        proj_dir = os.path.join(RAW_DIR, str(project_id))
        os.makedirs(proj_dir, exist_ok=True)
        filepath = os.path.join(proj_dir, filename)

        try:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(file.file, f)

            conn_thread = get_connection()
            c_thread = conn_thread.cursor()
            c_thread.execute(
    "INSERT INTO documents (project_id, name, path, type, status, uploaded_by) VALUES (?, ?, ?, ?, ?, ?)",
    (project_id, filename, filepath, "pending", "pending", user_id)
)
            doc_id = c_thread.lastrowid
            conn_thread.commit()
            conn_thread.close()

            documents = load_and_split_document(filepath)

            if background_tasks:
                background_tasks.add_task(save_and_index, project_id, doc_id, filename, filepath, "doc", documents)
            else:
                save_and_index(project_id, doc_id, filename, filepath, "doc", documents)

            results.append({
                "status": "processing",
                "doc_id": doc_id,
                "doc_name": filename,
                "chunks": len(documents)
            })

        except Exception as e:
            results.append({"status": "error", "doc_name": filename, "message": str(e)})

    return results

# ----------------------------
# Document Upload Status endpoint
# ----------------------------
@router.get("/status/{doc_id}")
async def check_status(doc_id: int):
    conn_thread = get_connection()
    c_thread = conn_thread.cursor()
    c_thread.execute("SELECT status, progress FROM documents WHERE id=?", (doc_id,))
    row = c_thread.fetchone()
    conn_thread.close()
    if row:
        return {"doc_id": doc_id, "status": row[0], "progress": row[1]}
    else:
        return {"doc_id": doc_id, "status": "not_found", "progress": 0}

# ----------------------------
# List documents for a project
# ----------------------------
@router.get("/project/{project_id}/documents")
def list_project_documents(project_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, path, type, status, progress FROM documents WHERE project_id=?", (project_id,))
    items = [{"doc_id": r[0], "doc_name": r[1], "path": r[2], "type": r[3], "status": r[4], "progress": r[5]} for r in c.fetchall()]
    conn.close()
    return {"project_id": project_id, "documents": items}

# ----------------------------
# Delete a document from project
# ----------------------------
@router.delete("/project/{project_id}/documents/{doc_id}")
def delete_document(project_id: int, doc_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, path FROM documents WHERE id=? AND project_id=?", (doc_id, project_id))
    r = c.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found in project")

    try:
        c.execute("DELETE FROM doc_chunks WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete document rows: {e}")
    conn.close()

    try:
        if os.path.exists(r[1]):
            os.remove(r[1])
    except Exception:
        pass

    pi = load_project_index(project_id)
    rebuild_project_index_from_store(project_id)
    return {"status": "deleted", "doc_id": doc_id}
