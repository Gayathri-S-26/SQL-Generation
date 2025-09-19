# app_backend.py
import uvicorn
from fastapi import FastAPI
from databases import init_db,get_connection
from projects_documents import router as projects_router
from gemini_chat_feedback import router as chat_router
from auth import router as auth_router, get_current_user
from fastapi.middleware.cors import CORSMiddleware
# ----------------------------
# Initialize DB
# ----------------------------
init_db()

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(title="SQL Generator", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Include routers
# ----------------------------
app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(chat_router)

# ----------------------------
# Root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "SQL Gen AI Backend is running."}

# ----------------------------
# Run with uvicorn
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("app_backend:app", host="0.0.0.0", port=8000, reload=True)
