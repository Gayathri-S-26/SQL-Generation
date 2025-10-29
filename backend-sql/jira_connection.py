#jira_connection.py
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from jira import JIRA
from pydantic import BaseModel

from backend_sql.projects_documents import RAW_DIR
from backend_sql.auth import get_connection, get_current_user
from backend_sql.split_embeddings import load_and_split_document, save_and_index


router = APIRouter()

class JiraConnectionRequest(BaseModel):
    jira_url: str
    username: str
    api_token: str
    is_default: bool = False

class JiraImportRequest(BaseModel):
    issue_key: str
    target_project_id: Optional[int] = None
    new_project_name: Optional[str] = None

# Jira service class
class JiraService:
    def __init__(self, jira_url: str, username: str, api_token: str):
        self.jira_url = jira_url
        self.username = username
        self.api_token = api_token
        self.client = None
        
    def connect(self):
        try:
            self.client = JIRA(
                server=self.jira_url,
                basic_auth=(self.username, self.api_token),
                options={"verify": False}
            )
            return True
        except Exception as e:
            print(f"Jira connection failed: {e}")
            return False
    
    def get_projects(self):
        if not self.client:
            return []
        return [{
            'id': project.id,
            'key': project.key,
            'name': project.name,
            'description': getattr(project, 'description', '')
        } for project in self.client.projects()]
    
    def get_issues(self, project_key: str, max_results: int = 100):
        if not self.client:
            return []
        
        issues = self.client.search_issues(
            f'project={project_key}',
            maxResults=max_results,
            expand='attachment,renderedFields'
        )
        
        issue_list = []
        for issue in issues:
            # Get attachment information
            attachments = []
            for attachment in issue.fields.attachment:
                attachments.append({
                    'filename': attachment.filename,
                    'url': attachment.content,
                    'size': attachment.size,
                    'mimeType': getattr(attachment, 'mimeType', '')
                })
            
            issue_list.append({
                'key': issue.key,
                'summary': issue.fields.summary,
                'description': getattr(issue.fields, 'description', ''),
                'issue_type': issue.fields.issuetype.name,
                'status': issue.fields.status.name,
                'assignee': getattr(issue.fields.assignee, 'displayName', 'Unassigned') if issue.fields.assignee else 'Unassigned',
                'reporter': getattr(issue.fields.reporter, 'displayName', '') if issue.fields.reporter else '',
                'priority': getattr(issue.fields.priority, 'name', '') if issue.fields.priority else '',
                'created': issue.fields.created,
                'updated': issue.fields.updated,
                'attachments': attachments
            })
        
        return issue_list
    
    def download_attachment(self, attachment_url: str, filename: str, download_path: str):
        """✅ Stable Jira attachment downloader using authenticated Jira session"""
        try:
            print(f"📎 Downloading attachment: {filename}")
            os.makedirs(download_path, exist_ok=True)

            # Ensure we have a connected Jira client
            if not self.client:
                print("⚠️ Jira client not connected — connecting now...")
                if not self.connect():
                    print("❌ Failed to connect to Jira for download.")
                    return None

            # Use Jira's internal authenticated session
            response = self.client._session.get(
                attachment_url,
                stream=True,
                verify=False
            )

            if response.status_code == 200:
                filepath = os.path.join(download_path, filename)
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Successfully downloaded: {filepath}")
                return filepath
            else:
                print(f"❌ Jira download failed with HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return None


# Add these new API endpoints after your existing endpoints

@router.post("/jira/connect")
async def connect_jira(req: JiraConnectionRequest, current_user: dict = Depends(get_current_user)):
    """Connect to Jira and save connection"""
    user_id = current_user["user_id"]
    
    # Test connection
    jira_service = JiraService(req.jira_url, req.username, req.api_token)
    if not jira_service.connect():
        raise HTTPException(status_code=400, detail="Failed to connect to Jira. Check your credentials.")
    
    # Save connection to database
    conn = get_connection()
    c = conn.cursor()
    
    # If this is set as default, remove default from other connections
    if req.is_default:
        c.execute("UPDATE jira_connections SET is_default=0 WHERE user_id=?", (user_id,))
    
    c.execute("""
        INSERT INTO jira_connections (user_id, jira_url, username, api_token, is_default)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, req.jira_url, req.username, req.api_token, req.is_default))
    
    conn.commit()
    conn.close()
    
    return {"status": "connected", "message": "Successfully connected to Jira"}

@router.get("/jira/connections")
async def get_jira_connections(current_user: dict = Depends(get_current_user)):
    """Get user's Jira connections"""
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, jira_url, username, is_default, created_at FROM jira_connections WHERE user_id=?", (user_id,))
    connections = [{
        'id': row[0],
        'jira_url': row[1],
        'username': row[2],
        'is_default': bool(row[3]),
        'created_at': row[4]
    } for row in c.fetchall()]
    
    conn.close()
    return connections

@router.get("/jira/projects")
async def get_jira_projects(connection_id: int = Query(...), current_user: dict = Depends(get_current_user)):
    """Get projects from Jira"""
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    
    # Get connection details
    c.execute("SELECT jira_url, username, api_token FROM jira_connections WHERE id=? AND user_id=?", 
              (connection_id, user_id))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Jira connection not found")
    
    jira_url, username, api_token = row
    conn.close()
    
    # Connect and get projects
    jira_service = JiraService(jira_url, username, api_token)
    if not jira_service.connect():
        raise HTTPException(status_code=400, detail="Failed to connect to Jira")
    
    projects = jira_service.get_projects()
    return projects

@router.get("/jira/issues")
async def get_jira_issues(connection_id: int = Query(...), project_key: str = Query(...), current_user: dict = Depends(get_current_user)):
    """Get issues from a Jira project"""
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    
    # Get connection details
    c.execute("SELECT jira_url, username, api_token FROM jira_connections WHERE id=? AND user_id=?", 
              (connection_id, user_id))
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Jira connection not found")
    
    jira_url, username, api_token = row
    conn.close()
    
    # Connect and get issues
    jira_service = JiraService(jira_url, username, api_token)
    if not jira_service.connect():
        raise HTTPException(status_code=400, detail="Failed to connect to Jira")
    
    issues = jira_service.get_issues(project_key)
    return issues

@router.post("/jira/import-issue")
async def import_jira_issue(req: JiraImportRequest, connection_id: int = Query(...), current_user: dict = Depends(get_current_user)):
    """Import a Jira issue into the application"""
    user_id = current_user["user_id"]
    
    # Get Jira connection
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT jira_url, username, api_token FROM jira_connections WHERE id=? AND user_id=?", (connection_id, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Jira connection not found")
    
    jira_url, username, api_token = row
    
    # Connect to Jira
    jira_service = JiraService(jira_url, username, api_token)
    if not jira_service.connect():
        conn.close()
        raise HTTPException(status_code=400, detail="Failed to connect to Jira")
    
    # Get issue details
    try:
        issue = jira_service.client.issue(req.issue_key, expand='attachment,renderedFields')
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Issue {req.issue_key} not found: {e}")
    
    # Determine target project
    target_project_id = req.target_project_id
    if not target_project_id and req.new_project_name:
        # Create new project
        c.execute("INSERT INTO projects (name, created_by) VALUES (?, ?)", (req.new_project_name, user_id))
        target_project_id = c.lastrowid
        # Add user to project
        c.execute("INSERT INTO user_project (user_id, project_id, role) VALUES (?, ?, ?)", 
                 (user_id, target_project_id, 'user'))
        conn.commit()
    
    if not target_project_id:
        conn.close()
        raise HTTPException(status_code=400, detail="Either target_project_id or new_project_name must be provided")
    
    # Prepare issue data
    attachments = []
    for attachment in issue.fields.attachment:
        attachments.append({
            'filename': attachment.filename,
            'url': attachment.content,
            'size': attachment.size,
            'mimeType': getattr(attachment, 'mimeType', '')
        })
    
    # Save imported issue
    c.execute("""
        INSERT INTO jira_imported_issues 
        (project_id, jira_issue_key, jira_issue_summary, jira_issue_description, 
         jira_issue_type, jira_issue_status, jira_issue_assignee, jira_issue_attachments)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        target_project_id,
        issue.key,
        issue.fields.summary,
        getattr(issue.fields, 'description', ''),
        issue.fields.issuetype.name,
        issue.fields.status.name,
        getattr(issue.fields.assignee, 'displayName', 'Unassigned') if issue.fields.assignee else 'Unassigned',
        json.dumps(attachments)
    ))
    imported_issue_id = c.lastrowid
    
    # Download and process attachments
    successful_attachments = 0
    failed_attachments = []
    
    if attachments:
        proj_dir = os.path.join(RAW_DIR, str(target_project_id))
        os.makedirs(proj_dir, exist_ok=True)
        
        for attachment in attachments:
            print(f"🔄 Processing attachment: {attachment['filename']}")
            filepath = jira_service.download_attachment(
                attachment['url'], 
                attachment['filename'],
                proj_dir
            )
            
            if filepath:
                try:
                    # Add to documents table
                    c.execute("""
                        INSERT INTO documents (project_id, name, path, type, status, uploaded_by)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (target_project_id, attachment['filename'], filepath, "jira_attachment", "processing", user_id))
                    doc_id = c.lastrowid
                    conn.commit()
                    
                    # Process the document
                    try:
                        print(f"🔄 Loading and splitting document: {attachment['filename']}")
                        documents = load_and_split_document(filepath)
                        print(f"✅ Successfully split into {len(documents)} chunks")
                        
                        # Save and index in background to avoid timeout
                        import threading
                        def background_index():
                            try:
                                save_and_index(target_project_id, doc_id, attachment['filename'], filepath, "jira_attachment", documents)
                                print(f"✅ Successfully indexed: {attachment['filename']}")
                            except Exception as e:
                                print(f"❌ Indexing failed for {attachment['filename']}: {e}")
                                conn_thread = get_connection()
                                c_thread = conn_thread.cursor()
                                c_thread.execute("UPDATE documents SET status=? WHERE id=?", ("error", doc_id))
                                conn_thread.commit()
                                conn_thread.close()
                        
                        # Start background indexing
                        thread = threading.Thread(target=background_index)
                        thread.daemon = True
                        thread.start()
                        
                        successful_attachments += 1
                        
                    except Exception as e:
                        print(f"❌ Failed to process Jira attachment {attachment['filename']}: {e}")
                        c.execute("UPDATE documents SET status=? WHERE id=?", ("error", doc_id))
                        failed_attachments.append(attachment['filename'])
                        conn.commit()
                        
                except Exception as e:
                    print(f"❌ Failed to add attachment to database {attachment['filename']}: {e}")
                    failed_attachments.append(attachment['filename'])
            else:
                print(f"❌ Download failed for: {attachment['filename']}")
                failed_attachments.append(attachment['filename'])
                
                # Even if download fails, create a placeholder document with the issue context
                try:
                    placeholder_content = f"""
                    Jira Issue: {issue.key}
                    Summary: {issue.fields.summary}
                    Description: {issue.fields.description or 'No description'}
                    
                    Original Attachment: {attachment['filename']}
                    Status: Download failed - file could not be retrieved from Jira
                    URL: {attachment['url']}
                    """
                    
                    placeholder_path = os.path.join(proj_dir, f"{attachment['filename']}.txt")
                    with open(placeholder_path, 'w', encoding='utf-8') as f:
                        f.write(placeholder_content)
                    
                    c.execute("""
                        INSERT INTO documents (project_id, name, path, type, status, uploaded_by)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (target_project_id, f"{attachment['filename']} (placeholder)", placeholder_path, "jira_attachment", "error", user_id))
                    conn.commit()
                    print(f"📝 Created placeholder for failed attachment: {attachment['filename']}")
                    
                except Exception as placeholder_error:
                    print(f"❌ Failed to create placeholder: {placeholder_error}")
    
    conn.commit()
    conn.close()
    
    # Create query context from issue
    issue_context = f"""
    Jira Issue: {issue.key}
    Summary: {issue.fields.summary}
    Description: {issue.fields.description or 'No description'}
    Type: {issue.fields.issuetype.name}
    Status: {issue.fields.status.name}
    Assignee: {getattr(issue.fields.assignee, 'displayName', 'Unassigned') if issue.fields.assignee else 'Unassigned'}
    """
    
    return {
        "status": "imported",
        "project_id": target_project_id,
        "imported_issue_id": imported_issue_id,
        "issue_context": issue_context,
        "successful_attachments": successful_attachments,
        "failed_attachments": failed_attachments,
        "total_attachments": len(attachments)
    }
