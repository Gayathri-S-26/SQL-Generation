import streamlit as st
import requests
import uuid
from io import BytesIO
from streamlit_cookies_manager import EncryptedCookieManager
from analytics_dashboard import analytics_dashboard_page

API_URL = "http://127.0.0.1:8000"

# ----------------------------
# Initialize cookie manager
# ----------------------------
cookies = EncryptedCookieManager(
    prefix="myapp_",  # cookie prefix
    password="super-secret-password-123!",  # secret key to encrypt cookie
)
if not cookies.ready():
    st.stop()
    
# ----------------------------
# CSS Styling
# ----------------------------

st.markdown("""
<style>
body {
    background-color: #E3D6EF;
}
.appview-container .main .block-container {
        padding-top: 1rem;  /* default is ~5rem, reduce as needed */
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
    }
/* Make all sidebar buttons borderless */
.stSidebar button {
    border: none;
    background-color: transparent;  /* optional: make background transparent too */
    box-shadow: none;  /* remove shadow if any */
}

/* Optional: remove hover effects */
.stSidebar button:hover {
    background-color: #3F0D66; 
    color:white; /* or keep transparent */
}

.project-button button {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

.project-button button:hover {
    background-color: #3F0D66 !important;
    color: white !important;
}

button:not(.project-list-btn):not(.stSidebar button) {
    background-color: #3F0D66 !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}

button:not(.project-list-btn):not(.stSidebar button):hover {
    opacity: 0.9;
    color:white;
    cursor: pointer;
}

.stChatInput { 
   border: 2px solid #3F0D66 !important;
   border-radius: 10px !important;
   padding: 5px !important;
}

.chat-container { max-height: 500px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
.user-bubble { background-color: #3F0D66; padding: 8px 12px; border-radius: 15px; margin: 5px; max-width: 100%; word-wrap: break-word; color:white; }
.user-message { justify-content: flex-end; }
.chat-message { display: flex; }

</style>
""", unsafe_allow_html=True)


# ----------------------------
# Session State Initialization
# ----------------------------
if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "role" not in st.session_state:
    st.session_state.role = None
    
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    
if "username" not in st.session_state:
    st.session_state.username = None
    
if "page_history" not in st.session_state:
    st.session_state.page_history = []

if "page" not in st.session_state:
    st.session_state.page = "login"
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "project_name" not in st.session_state:
    st.session_state.project_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "doc_progress" not in st.session_state:
    st.session_state.doc_progress = {}
if "new_uploads" not in st.session_state:
    st.session_state.new_uploads = []


def render_header_with_back():
    """Render global header and back button in same line."""
    cols = st.columns([1, 9])  # Adjust ratio for back button vs header

    with cols[0]:
        # Back button logic
        if st.session_state.page != "login" and st.session_state.page_history:
            if st.button("← Back"):
                while st.session_state.page_history:
                    prev_page = st.session_state.page_history.pop()
                    if prev_page != "login":
                        st.session_state.page = prev_page
                        st.rerun()

    with cols[1]:
        # Global header
        st.markdown(
            """
            <h3 style='
                text-align: center;  /* align left to stay next to back button */
                color: #3F0D66; 
                margin-top: 5px; 
                margin-bottom: 5px;
            '>🤖 AI-Powered SQL Automation</h3>
            <hr style='margin-top: 5px; margin-bottom: 10px;'>
            """,
            unsafe_allow_html=True
        )
    
# ----------------------------
# Helper: Show User Info Banner
# ----------------------------
def show_user_banner():
    st.markdown(
        f"""
        <div style="position: absolute; top: 15px; right: 25px; font-size: 12px; color: gray;">
            🆔 User ID: <code>{st.session_state.user_id}</code>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ----------------------------
# Go to Page
# ----------------------------
def go_to_page(page_name, save_cookie=False):
    """
    Navigate to a page and optionally update last_page in cookies.
    - save_cookie: Only True when you really want to persist last_page.
    """
    if st.session_state.page != page_name:
        st.session_state.page_history.append(st.session_state.page)
    st.session_state.page = page_name
    st.rerun()
    
    if save_cookie:
        cookies["last_page"] = page_name  # save to cookie


# ----------------------------
# Helper: Set token cookie
# ----------------------------
def set_access_token(access_token, user_id, username):
    st.session_state.access_token = access_token
    st.session_state.user_id = user_id
    st.session_state.username = username
    cookies["access_token"] = access_token
    cookies["user_id"] = user_id
    cookies["username"] = username
    cookies["last_page"] = ""
    cookies.save()

# ----------------------------
# Helper: Clear token (logout)
# ----------------------------
def clear_access_token():
    st.session_state.access_token = None
    st.session_state.user_id = None
    st.session_state.username = None
    cookies["access_token"] = ""
    cookies["user_id"] = ""
    cookies["username"] = ""
    cookies.save()

# ----------------------------
# Attempt persistent login from cookie
# ----------------------------
if not st.session_state.access_token:
    access_token = cookies.get("access_token")
    if access_token:
        try:
            resp = requests.get(f"{API_URL}/verify_access_token",
                                headers={"Authorization": f"Bearer {access_token}"})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.user_id = data["user_id"]
                st.session_state.username = data.get("username")
                st.session_state.role = data.get("role", "user")
                st.session_state.access_token = access_token
                st.session_state.token_verified = True
                last_page = cookies.get("last_page") or "projects"
                st.session_state.page = last_page
                st.rerun()
            else:
                clear_access_token()
        except Exception:
            clear_access_token()

# ----------------------------
# Login page
# ----------------------------
def login_page():
    render_header_with_back()
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            try:
                resp = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    set_access_token(data["access_token"], data["user_id"], data["username"])
                    st.session_state.role = data.get("role", "user")
                    go_to_page("projects")
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            except Exception as e:
                st.error(f"❌ Login failed: {e}")
        else:
            st.error("❌ Enter username and password")

# ----------------------------
# Page: Projects
# ----------------------------
def projects_page():
    render_header_with_back()
    st.title("📁 Projects")

    # Create Project button (regular Streamlit button)
    if st.button("➕ Create Project"):
        go_to_page("create_project")

    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        projects_resp = requests.get(f"{API_URL}/projects", headers=headers)
        projects = projects_resp.json() if projects_resp.status_code == 200 else []
    except Exception:
        projects = []

    st.subheader("Your Projects")

    if projects:
        st.markdown(
            """
        <style>
        .project-name {
            font-size: 16px;
            padding: 6px 10px;
            margin: 2px 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        for p in projects:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"<div class='project-name'>📂 {p['project_name']}</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Open", key=f"open_{p['project_id']}"):
                    st.session_state.project_id = p["project_id"]
                    st.session_state.project_name = p["project_name"]
                    go_to_page("project_detail")
                    st.rerun()
    else:
        st.info("No projects found. Use ➕ Create Project above.")

# ----------------------------
# Page: Create Project
# ----------------------------
def create_project_page():
    render_header_with_back()
    st.title("➕ Create New Project")

    proj_name = st.text_input("Project Name")
    if st.button("Create"):
        if proj_name.strip():
            try:
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                resp = requests.post(
                    f"{API_URL}/project",
                    json={"project_name": proj_name},
                    headers=headers,
                )
                data = resp.json()
                st.session_state.project_id = data["project_id"]
                st.session_state.project_name = data["project_name"]
                st.success(f"✅ Project '{data['project_name']}' created")
                go_to_page("projects")
            except Exception as e:
                st.error(f"❌ Failed to create project: {e}")
        else:
            st.warning("Please enter a project name")

# ----------------------------
# Page: Project Detail
# ----------------------------
def project_detail_page():
    render_header_with_back()
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

    st.title(f"📂 Documents: {st.session_state.project_name}")

    if st.button("➕ Add Documents"):
        go_to_page( "upload")

    try:
        docs_resp = requests.get(
            f"{API_URL}/project/{st.session_state.project_id}/documents",
            headers=headers,
        )
        documents = docs_resp.json().get("documents", []) if docs_resp.status_code == 200 else []
    except Exception:
        documents = []

    if documents:
        for doc in documents:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📑 {doc['doc_name']}")
            if col2.button("🗑️ Delete", key=f"del_{doc['doc_id']}"):
                try:
                    resp = requests.delete(
                        f"{API_URL}/project/{st.session_state.project_id}/documents/{doc['doc_id']}",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        st.success(f"Deleted {doc['doc_name']} ✅")
                        st.session_state.uploaded_docs = [
                            d for d in st.session_state.uploaded_docs if d["doc_id"] != doc["doc_id"]
                        ]
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {resp.text}")
                except Exception as e:
                    st.error(f"❌ Exception: {e}")
    else:
        st.info("No documents uploaded yet.")

# ----------------------------
# Page: Upload Document
# ----------------------------
def upload_page():
    render_header_with_back()
    st.title("➕ Add Documents")

    uploaded_files = st.file_uploader(
        "Upload new documents (CSV, XLSX, PDF, DOCX)",
        type=["csv", "xlsx", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Upload"):
        try:
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            files_payload = [
                ("files", (f.name, BytesIO(f.getvalue()), "application/octet-stream"))
                for f in uploaded_files
            ]
            response = requests.post(
                f"{API_URL}/upload",
                params={"project_id": st.session_state.project_id},
                files=files_payload,
                headers=headers,
            )
            results = response.json()
            for result in results:
                if result.get("status") == "processing":
                    st.success(f"Uploaded: {result['doc_name']}")
                else:
                    st.error(result.get("message", "Upload error"))
            go_to_page("project_detail")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

# ----------------------------
# Page: Query (Chat + Feedback)
# ----------------------------
def query_page():
    render_header_with_back()
    st.title(f"💬 Queries - {st.session_state.project_name}")
    import streamlit.components.v1 as components

    # Reset button only if history exists
    if st.session_state.chat_history:
        if st.button("🆕 New Query"):
            st.session_state.chat_history = []
            st.session_state.conversation_id = None
            st.rerun()

    # Show past messages using st.chat_message
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                    f'<div class="chat-message user-message"><div class="user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                # Feedback only for assistant messages
                fb_col1, fb_col2, fb_col3 = st.columns([3, 1, 1])
                with fb_col1:
                    st.markdown("**Was this answer helpful?**")
                with fb_col2:
                    fb_up = st.button("👍", key=f"fb_up_{idx}")
                with fb_col3:
                    fb_down = st.button("👎", key=f"fb_down_{idx}")

                if f"feedback_type_{idx}" not in st.session_state:
                    st.session_state[f"feedback_type_{idx}"] = None
                if fb_up:
                    st.session_state[f"feedback_type_{idx}"] = "up"
                if fb_down:
                    st.session_state[f"feedback_type_{idx}"] = "down"

                if st.session_state[f"feedback_type_{idx}"] in ["up", "down"]:
                    comment = st.text_area("💬 Additional comment", key=f"fb_comment_{idx}", height=80)
                    if st.button("Submit Feedback", key=f"fb_submit_{idx}"):
                        feedback_payload = {
                            "query_id": msg.get("query_id", ""),
                            "project_id": st.session_state.project_id,
                            "query": msg.get("user_query", ""),
                            "answer": msg["content"],
                            "contexts": msg.get("contexts", []),
                            "feedback": st.session_state[f"feedback_type_{idx}"],
                            "comment": comment,
                            "user_id": st.session_state.user_id
                        }
                        try:
                            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                            resp = requests.post(f"{API_URL}/feedback", json=feedback_payload, headers=headers)
                            if resp.status_code == 200:
                                st.success("✅ Feedback submitted!")
                            else:
                                st.error(f"❌ Failed: {resp.text}")
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")

    # 🚀 Input is fixed at the bottom now
    if prompt := st.chat_input("Enter your Question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        payload = {
            "project_id": int(st.session_state.project_id),
            "query": prompt,
            "conversation_id": st.session_state.conversation_id,
        }
        try:
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            with st.spinner("Generating response..."):
                response = requests.post(f"{API_URL}/chat", json=payload, headers=headers)
                data = response.json()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": data.get("answer", ""),
                    "query_id": data.get("query_id"),
                    "user_query": prompt,
                    "contexts": data.get("contexts", [])
                })
                st.session_state.conversation_id = data.get("conversation_id")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Page: Query History
# ----------------------------
def query_history():
    render_header_with_back()
    st.title("🔍 Query History")

    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        resp = requests.get(f"{API_URL}/queryhistory/{st.session_state.project_id}", headers=headers)

        if resp.status_code == 200:
            all_queries = resp.json()

            if all_queries:
                selected_user = st.session_state.user_id

                user_convs = list(
                    {
                        q.get("conversation_id")
                        for q in all_queries
                        if q.get("user_id") == selected_user
                    }
                )

                if user_convs:
                    selected_conv = st.selectbox("Select Query Conversation", user_convs)

                    conv_history = [
                        q
                        for q in all_queries
                        if q.get("user_id") == selected_user
                        and q.get("conversation_id") == selected_conv
                    ]

                    for msg in conv_history[::-1]:
                        if msg.get("query"):
                            st.markdown(
                                f'<div class="chat-message user-message"><div class="user-bubble">{msg["query"]}</div></div>',
                                unsafe_allow_html=True,
                            )
                        if msg.get("answer"):
                            st.markdown(msg["answer"])
                else:
                    st.info("No Queries found for this user.")
            else:
                st.info("No Query history available yet.")
        else:
            st.error("Failed to load Query history")

    except Exception as e:
        st.error(f"Error loading Query history: {e}")

# ----------------------------
# Router
# ----------------------------
# ----------------------------
# Sidebar Navigation (Streamlit native)
# ----------------------------
if st.session_state.page != "login":
    with st.sidebar:
        # Always show user info + logout
        st.markdown("---")
        st.caption(f" 👤{st.session_state.username}")
        if st.button("🔓 Logout"):
            clear_access_token()
            go_to_page("login")
            st.rerun()

        # Show buttons based on current page
        if st.session_state.role == "admin":
            if st.button("📈 Analytics Dashboard"):
                go_to_page("analytics_dashboard")

        if st.session_state.page == "create_project":
            if st.button("📁 Projects"):
                go_to_page("projects")

        elif st.session_state.page == "project_detail":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("💬 Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")

        elif st.session_state.page == "upload":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("💬 Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")

        elif st.session_state.page == "query":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🔍 Query History"):
                go_to_page("query_history")
                
        elif st.session_state.page == "query_history":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("💬 Query"):
                go_to_page("query")
                
        elif st.session_state.page == "analytics_dashboard":
            if st.button("📁 Projects"):
                go_to_page("projects")


if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "projects":
    projects_page()
elif st.session_state.page == "create_project":
    create_project_page()
elif st.session_state.page == "project_detail":
    project_detail_page()
elif st.session_state.page == "query":
    query_page()
elif st.session_state.page == "upload":
    upload_page()
elif st.session_state.page == "analytics_dashboard":
    analytics_dashboard_page()
elif st.session_state.page == "query_history":  
    query_history()
