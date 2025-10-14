#analytics_dashboard.py
import sqlite3
from fastapi import APIRouter, Depends, HTTPException
import numpy as np
import pandas as pd
from pathlib import Path
from backend_sql.databases import DB_PATH
from backend_sql.auth import get_current_user

router = APIRouter()


def load_data():
    """
    Load data from SQLite database for analytics dashboard
    """
    conn = sqlite3.connect(DB_PATH)
    queries = pd.read_sql("SELECT * FROM queries", conn)
    feedback = pd.read_sql("SELECT * FROM feedback", conn)
    projects = pd.read_sql("SELECT * FROM projects", conn)
    documents = pd.read_sql("SELECT * FROM documents", conn)
    users = pd.read_sql("SELECT * FROM users", conn)
    conn.close()
    return queries, feedback, projects, documents, users


@router.get("/analytics-data")
def get_analytics_data(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access forbidden: Admins only.")
 
    try:
        queries, feedback, projects, documents, users = load_data()
 
        # --- THIS IS THE FIX ---
        # Replace any NaN values in the queries DataFrame with None (which becomes 'null' in JSON)
        queries = queries.replace({np.nan: None})
       
        return {
            "queries": queries.to_dict(orient="records"),
            "feedback": feedback.to_dict(orient="records"),
            "projects": projects.to_dict(orient="records"),
            "documents": documents.to_dict(orient="records"),
            "users": users[["id", "username", "role"]].to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load analytics data: {str(e)}")
