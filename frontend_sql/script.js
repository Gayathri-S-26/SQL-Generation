marked.setOptions({
  breaks: true,        // Convert \n to <br>
  gfm: true,           // GitHub Flavored Markdown
  paragraphs: true,    // Create proper <p> tags
  headerIds: false,    // Disable automatic header IDs
  mangle: false,       // Don't escape underscores
  sanitize: false,     // Don't sanitize HTML (allows proper formatting)
  smartLists: true,    // Use smarter list behavior
  smartypants: true,   // Use smart punctuation
  xhtml: false         // Don't use XHTML self-closing tags
});

document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION & STATE ---
    const API_BASE_URL = 'http://127.0.0.1:8000';
    let state = {
        token: localStorage.getItem('accessToken'), user: null, currentPage: 'query',
        projects: [], selectedProjectId: null, selectedProjectName: null,
        chatHistory: {}, conversationId: {}, charts: {}, isStreaming: false, regeneratingQueryId: null,
    };
    state.jiraConnections = [];
    state.currentJiraConnection = null;
    state.jiraProjects = [];
    state.jiraIssues = [];

    const $ = (selector) => document.querySelector(selector);
    const $$ = (selector) => document.querySelectorAll(selector);

    function init() {
        setupEventListeners();
        verifyTokenAndInit();
    }

    async function apiRequest(endpoint, options = {}) {
        showLoader(true);
        const headers = { ...options.headers };
        if (!(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }
        if (state.token) {
            headers['Authorization'] = `Bearer ${state.token}`;
        }
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, { ...options, headers });
            if (response.status === 401) { handleLogout(); throw new Error("Session expired. Please log in."); }
            if (options.stream) return response;
            if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.detail || `Server Error: ${response.status}`); }
            return response.status === 204 ? null : response.json();
        } catch (error) {
            showError(error.message);
            throw error;
        } finally {
            showLoader(false);
        }
    }

    function navigateTo(page) {
    console.log(`Attempting to navigate to page: ${page}`); // For debugging

    // Hide all page content divs first
    $$('.page-content').forEach(p => p.style.display = 'none');

    // Find the new page to display
    const newPage = $(`#${page}-page`);

    // **THIS IS THE FIX**
    // Check if the page element actually exists before trying to show it
    if (newPage) {
        newPage.style.display = 'block';
    } else {
        console.error(`Navigation failed: Page element with ID #${page}-page not found.`);
        return; // Stop the function if the page doesn't exist
    }

    // Update the active state of the sidebar buttons
    $$('.nav-btn').forEach(b => b.classList.remove('active'));
    const activeBtn = $(`button[data-page="${page}"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }

    state.currentPage = page;

    // Load data for the new page
    switch (page) {
        case 'projects': loadProjects(); break;
        case 'query': loadProjectsForQueryPage(); break;
        case 'query_history': loadProjectsForHistoryPage(); break;
        case 'analytics': loadAnalyticsData(); break;
        case 'jira': loadJiraConnections(); loadProjectsForJiraImport(); break;
    }
}
    const showLoader = (isLoading) => $('#loader').style.display = isLoading ? 'flex' : 'none';
    const showError = (message, container = '#auth-error') => $(container).textContent = message;

    async function verifyTokenAndInit() {
        if (state.token) {
            try {
                state.user = await apiRequest('/verify_access_token');
                $('#username-display').textContent = state.user.username;
                $('#auth-container').style.display = 'none';
                $('#app-wrapper').style.display = 'flex';
                if (state.user.role === 'admin') {
                    $$('.admin-only').forEach(el => el.style.display = 'block');
                }
                navigateTo(state.currentPage);
            } catch (error) { handleLogout(); }
        }
    }

    function handleLogout() {
        state.token = null; state.user = null;
        localStorage.removeItem('accessToken');
        location.reload();
    }

    function setupEventListeners() {
        $('#login-form').addEventListener('submit', handleLogin);
        $('#register-form').addEventListener('submit', handleRegister);
        $('#show-login-tab').addEventListener('click', () => switchAuthTab('login'));
        $('#show-register-tab').addEventListener('click', () => switchAuthTab('register'));
        $('#logout-btn').addEventListener('click', handleLogout);

        $('#sidebar-nav').addEventListener('click', (e) => {
            const button = e.target.closest('.nav-btn');
            if (button) {
                e.preventDefault(); // Prevent any default form submission behavior
                navigateTo(button.dataset.page);
            }
        });
        $('#sidebar-new-query-btn').addEventListener('click', () => {
            navigateTo('query');
            if (state.selectedProjectId) {
            // Reset chat + conversation for the selected project
            state.chatHistory[state.selectedProjectId] = [];
            state.conversationId[state.selectedProjectId] = null;

            // Refresh UI - this will now show welcome message
            renderChatHistory();
            }
        });

        $('#create-project-form').addEventListener('submit', handleCreateProject);
        $('#back-to-projects-btn').addEventListener('click', () => navigateTo('projects'));
        $$('.tabs .tab-btn').forEach(btn => btn.addEventListener('click', handleTabSwitch));
        $('#upload-form').addEventListener('submit', handleUploadFiles);
        $('#chat-form').addEventListener('submit', handleChatSubmit);
        $('#query-project-selector').addEventListener('change', (e) => {
            const selectedOption = e.target.options[e.target.selectedIndex];
            state.selectedProjectId = selectedOption.value;
            state.selectedProjectName = selectedOption.text;
            renderChatHistory(); // This will show welcome message for newly selected project
        });
        $('#history-project-selector').addEventListener('change', loadConversationsForHistory);
        $('#history-conversation-selector').addEventListener('change', renderConversationTranscript);
        $('#jira-connect-form').addEventListener('submit', handleJiraConnect);
        $('#jira-connection-selector').addEventListener('change', loadJiraProjects);
        $('#confirm-import-btn').addEventListener('click', confirmJiraImport);
        $('#cancel-import-btn').addEventListener('click', () => $('#jira-import-modal').style.display = 'none');
        $('#chat-input').addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent newline
                $('#chat-form').dispatchEvent(new Event('submit')); // Trigger send
            }
            // If Shift+Enter is pressed → allow newline
        });
        $('#chat-input').addEventListener('input', (e) => {
        const el = e.target;
        el.style.height = 'auto';                      // reset first
        el.style.height = Math.min(el.scrollHeight, 120) + 'px';      // then resize correctly
        });
        $('#sidebar-toggle').addEventListener('click', toggleSidebar);

    }

    function toggleSidebar() {
    $('#sidebar').classList.toggle('sidebar-collapsed');
    }
    

    async function handleJiraCreateProject() {
        const newProjectName = $('#new-project-name-input').value.trim();
        if (!newProjectName) {
            showError('Please enter a project name', '#main-error'); // Changed from '#auth-error'
            return;
        }

        try {
            // Use the existing create project endpoint
            const result = await apiRequest('/project', { 
                method: 'POST', 
                body: JSON.stringify({ project_name: newProjectName }) 
            });
            
            // Update the projects list and select the newly created project
            await loadProjectsForJiraImport();
            
            // Switch to existing project option and select the new project
            $('input[name="import-option"][value="existing"]').checked = true;
            $('#new-project-section').style.display = 'none';
            $('#existing-projects-select').style.display = 'block';
            $('#existing-projects-select').value = result.project_id;
            
            $('#new-project-name-input').value = '';
            
            showError('✅ Project created successfully!', '#main-error'); // Changed from '#auth-error'
            
        } catch (error) {
            showError(`Failed to create project: ${error.message}`, '#main-error'); // Changed from '#auth-error'
        }
    }

    // Auth Handlers
    async function handleLogin(e) {
        e.preventDefault();
        const username = $('#login-username').value; const password = $('#login-password').value;
        try {
            const data = await apiRequest('/login', { method: 'POST', body: JSON.stringify({ username, password }) });
            localStorage.setItem('accessToken', data.access_token);
            state.token = data.access_token;
            verifyTokenAndInit();
        } catch (error) {}
    }
    async function handleRegister(e) {
        e.preventDefault();
        const username = $('#register-username').value; const password = $('#register-password').value;
        if (password !== $('#register-confirm-password').value) return showError("Passwords do not match.");
        try {
            await apiRequest('/register', { method: 'POST', body: JSON.stringify({ username, password }) });
            alert("Registration successful! Please log in.");
            switchAuthTab('login');
        } catch (error) {}
    }
    function switchAuthTab(tabName) {
        $('#login-form').style.display = tabName === 'login' ? 'flex' : 'none';
        $('#register-form').style.display = tabName === 'register' ? 'flex' : 'none';
        $('#show-login-tab').classList.toggle('active', tabName === 'login');
        $('#show-register-tab').classList.toggle('active', tabName === 'register');
        showError('');
    }
    
    // Project & Document Logic
    async function loadProjects() {
        try {
            state.projects = await apiRequest('/projects');
            const grid = $('#projects-grid'); grid.innerHTML = '';
            if (!state.projects || state.projects.length === 0) { grid.innerHTML = '<p>No projects found. Create one above!</p>'; return; }
            state.projects.forEach(p => {
                grid.innerHTML += `<div class="card project-card"><h4>${p.project_name}</h4><p>Created: ${new Date(p.created_at).toLocaleDateString()}</p><button class="manage-docs-btn" data-id="${p.project_id}" data-name="${p.project_name}">Manage Documents</button></div>`;
            });
            $$('.manage-docs-btn').forEach(btn => btn.addEventListener('click', (e) => {
                state.selectedProjectId = e.target.dataset.id; state.selectedProjectName = e.target.dataset.name;
                navigateTo('project-detail'); loadProjectDetails();
            }));
        } catch (error) { console.error("Failed to load projects", error); }
    }
    async function handleCreateProject(e) {
        e.preventDefault(); const name = $('#new-project-name').value.trim(); if (!name) return;
        try { await apiRequest('/project', { method: 'POST', body: JSON.stringify({ project_name: name }) }); $('#new-project-name').value = ''; loadProjects(); } catch (error) {}
    }

    async function pollStatus(doc) {
        try {
            const statusData = await apiRequest(`/status/${doc.doc_id}`);
            if (statusData.status !== 'completed') {
                setTimeout(() => pollStatus(doc), 2000); // poll every 2s
            }
            loadProjectDetails(true); // refresh UI
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }

    async function loadProjectDetails(polling = false) {
        $('#project-detail-title').textContent = `Documents for: ${state.selectedProjectName}`;
        const data = await apiRequest(`/project/${state.selectedProjectId}/documents`);
        const list = $('#document-list');
        list.innerHTML = '';

        let hasPending = false;

        if (data.documents.length === 0) { 
            list.innerHTML = '<p>No documents found. Upload one!</p>'; 
            return; 
        }

        data.documents.forEach(doc => {
            if (doc.status.toLowerCase() === 'pending' || doc.status === 'indexing') {
                hasPending = true;
                if (!polling) pollStatus(doc); // start polling this doc
            }

            list.innerHTML += `
                <div class="document-list-item">
                    <span class="doc-name">${doc.doc_name}</span>
                    <div class="doc-status">
                        <span>${doc.status}</span>
                        <button class="delete-btn" data-id="${doc.doc_id}">🗑️ Delete</button>
                    </div>
                </div>`;
        });

        $$('.delete-btn').forEach(btn => btn.addEventListener('click', handleDeleteDocument));
    }

    async function handleDeleteDocument(e) {
        const docId = e.target.dataset.id; if (!confirm("Are you sure?")) return;
        try { await apiRequest(`/project/${state.selectedProjectId}/documents/${docId}`, { method: 'DELETE' }); loadProjectDetails(); } catch(error) {}
    }
    async function handleUploadFiles(e) {
        e.preventDefault(); const files = $('#file-upload').files; if (files.length === 0) return;
        const formData = new FormData();
        for (const file of files) { formData.append('files', file); }
        try { await apiRequest(`/upload?project_id=${state.selectedProjectId}`, { method: 'POST', body: formData }); $('#upload-form').reset(); loadProjectDetails(); } catch(error) {}
    }
    
    // Chat Logic (omitted for brevity - keep your existing working chat functions)
    async function loadProjectsForQueryPage() { 
        try { 
            state.projects = await apiRequest('/projects'); 
            const selector = $('#query-project-selector'); 
            selector.innerHTML = '<option value="">-- Select a Project --</option>'; 
            if (!state.projects || state.projects.length === 0) return; 
            state.projects.forEach(p => { 
                const option = document.createElement('option'); 
                option.value = p.project_id; 
                option.textContent = p.project_name; 
                selector.appendChild(option); 
            }); 
            

            if(state.projects.length > 0) { 
                // Don't auto-select first project - leave it as "Select a Project"
                if (state.selectedProjectId) {
                    selector.value = state.selectedProjectId;
                    state.selectedProjectName = selector.options[selector.selectedIndex].text; 
                    renderChatHistory(); // This will now show welcome message if no chats
                }
                else {
                // ✅ ADD WELCOME MESSAGE HERE when no project is selected
                const messagesContainer = $('#chat-messages'); 
                messagesContainer.innerHTML = `
                <div class="chat-bubble assistant-bubble welcome-bubble">
                    <div class="welcome-content">
                        <div class="welcome-text">
                            👋<br>
                            <strong>Welcome to AI Powered SQL Automation!</strong><br>I can generate SQL queries, analyze documents, and answer questions.
                        </div>
                    </div>
                </div>
                `;
            }
            } 

        } catch (error) {} 
    }
    function renderChatHistory() { 
        const messagesContainer = $('#chat-messages'); 
        messagesContainer.innerHTML = ''; 
        const history = state.chatHistory[state.selectedProjectId] || []; 

        console.log('History retrieved for rendering:', history);

        console.log('🧠 CURRENT CHAT HISTORY FOR RENDERING:', history);
        console.log('📋 Project ID:', state.selectedProjectId);
        console.log('🔢 Number of messages:', history.length);
        
        // Log each message with query_id
        history.forEach((msg, index) => {
            console.log(`📄 Message ${index + 1} (${msg.role}):`, {
                content: msg.content?.substring(0, 30) + '...',
                query_id: msg.query_id,
                messageId: msg.messageId
            });
        });
        
        // Show welcome message if no chat history
        if (history.length === 0) {
            messagesContainer.innerHTML = `
            <div class="chat-bubble assistant-bubble welcome-bubble">
                <div class="welcome-content">
                    <div class="welcome-text">
                        👋<br>
                        <strong>Welcome to AI Powered SQL Automation!</strong><br>I can generate SQL queries, analyze documents, and answer questions.
                    </div>
                </div>
            </div>
            `;
            return;
        }
        
        // Render existing chat history
        history.forEach(msg => {
        const messageDiv = renderEnhancedMessage(msg.content, msg.role, msg.messageId);
        
        // Store message ID in state for reference
        if (!msg.messageId) {
            msg.messageId = messageDiv.id;
        }
        
        messagesContainer.appendChild(messageDiv);
        
        // Add feedback section for assistant messages
        
        });

        setTimeout(autoScrollToBottom, 100);
        enhanceCodeBlocksInContainer(messagesContainer);
    }
    function enhanceCodeBlocks() {
        // Select all code blocks inside assistant messages
        document.querySelectorAll('.assistant-bubble pre').forEach(pre => {
            // Avoid adding multiple copy buttons
            if (pre.querySelector('.copy-btn')) return;

            // Create the copy button
            const button = document.createElement('button');
            button.className = 'copy-btn';
            button.innerHTML = 'Copy';

            // Add click functionality
            button.addEventListener('click', () => {
                const code = pre.querySelector('code')?.innerText || pre.innerText;
                navigator.clipboard.writeText(code).then(() => {
                    button.innerHTML = 'Copied!';
                    setTimeout(() => (button.innerHTML = 'Copy'), 1500);
                });
            });

            // Make the <pre> position relative and add the button
            pre.style.position = 'relative';
            pre.appendChild(button);
        });
    }

    // UI helpers to disable/enable send controls while streaming
function disableSendUI() {
    const submitBtn = document.querySelector('#chat-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.classList.add('disabled-button');
    }
    const input = document.querySelector('#chat-input');
    if (input) input.disabled = true;
    state.isStreaming = true;
}

function enableSendUI() {
    const submitBtn = document.querySelector('#chat-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.classList.remove('disabled-button');
    }
    const input = document.querySelector('#chat-input');
    if (input) input.disabled = false;
    state.isStreaming = false;
}

// Replace existing handleChatSubmit with this version
async function handleChatSubmit(e, isRegeneration = false) {
    if (e) e.preventDefault();

    // Prevent sending while a previous response is still streaming
    if (state.isStreaming) {
        console.warn('Send blocked: assistant is still responding.');
        return;
    }

    const input = $('#chat-input');
    const query = input.value.trim();

    if (!query) return;
    if (!state.selectedProjectId) {
        Swal.fire({
            text: 'Please select a project first',
            icon: 'warning',
            showConfirmButton: true,
            iconColor: '#3F0D66',
            background: "#ffffff",
            color: "#3F0D66",
            confirmButtonColor: "#3F0D66",
            confirmButtonText: "OK",
            buttonsStyling: true   // <- IMPORTANT
        });
        return;
    }
    disableSendUI(); // disable UI until response completes

    const projId = state.selectedProjectId;

    // Initialize chat history if needed
    if (!state.chatHistory[projId]) {
        state.chatHistory[projId] = [];
    }

    // Add user message to history
    const userMessage = {
        role: 'user',
        content: query,
        messageId: `msg-${Date.now()}-user-${Math.random().toString(36).substr(2, 5)}`,
        originalContent: query
    };

    state.chatHistory[projId].push(userMessage);

    // Render user message with actions
    const userMessageDiv = renderEnhancedMessage(query, 'user', userMessage.messageId);
    $('#chat-messages').appendChild(userMessageDiv);

    input.value = '';
    input.dispatchEvent(new Event('input'));

    // SCROLL: After user message is added
    autoScrollToBottom();

    // Create assistant message placeholder
    const assistantMessageId = `msg-${Date.now()}-assistant-${Math.random().toString(36).substr(2, 5)}`;
    const assistantBubble = document.createElement('div');
    assistantBubble.className = 'chat-bubble assistant-bubble';
    assistantBubble.id = assistantMessageId;

    let currentStatus = "🚀 Starting processing...";
    assistantBubble.innerHTML = `
        <div class="message-content">
            <span class="typing-cursor">${currentStatus}</span>
        </div>
    `;

    $('#chat-messages').appendChild(assistantBubble);
    autoScrollToBottom();

    // Add timeout handling
    const TIMEOUT_MS = 45000; // 45 seconds
    let responseComplete = false;
    const timeoutId = setTimeout(() => {
        if (!responseComplete) {
            currentStatus = "⚠️ Taking longer than expected...";
            const messageContent = assistantBubble.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = `<span class="typing-cursor">${currentStatus}</span>`;
            }
            autoScrollToBottom();
        }
    }, TIMEOUT_MS);

    try {
        const response = await apiRequest('/chat', {
            method: 'POST',
            stream: true,
            body: JSON.stringify({
                project_id: parseInt(projId),
                query: query,
                conversation_id: state.conversationId[projId] || null,
            }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";
        let metaData = {};

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            let chunk = decoder.decode(value, { stream: true });

            // Handle status updates
            if (chunk.includes("[[[STATUS]]]")) {
                currentStatus = chunk.replace("[[[STATUS]]]", "").trim();
                const messageContent = assistantBubble.querySelector('.message-content');
                if (messageContent) {
                    if (fullResponse) {
                        messageContent.innerHTML = `
                            <div class="status-container">
                                <span class="typing-cursor">${currentStatus}</span>
                                <div style="margin-top: 10px; opacity: 0.8;">${marked.parse(fullResponse + " ▌")}</div>
                            </div>
                        `;
                    } else {
                        messageContent.innerHTML = `<span class="typing-cursor">${currentStatus}</span>`;
                    }
                }
                autoScrollToBottom();
                continue;
            }

            // Handle metadata (may be emitted early or at end)
            // In handleChatSubmit - FIX METADATA PARSING:

            // Handle metadata (may be emitted early or at end)
            if (chunk.includes("[[[META]]]")) {
                const parts = chunk.split("[[[META]]]");
                chunk = parts[0];
                try {
                    const metaString = parts[1] || "{}";
                    console.log("🔍 Raw metadata string:", metaString); // DEBUG
                    metaData = JSON.parse(metaString);
                    console.log("✅ Parsed metadata:", metaData); // DEBUG
                    
                    // If meta contains conversation_id, persist immediately
                    if (metaData.conversation_id) {
                        state.conversationId[projId] = metaData.conversation_id;
                    }
                    // CRITICAL FIX: Update user message with query_id immediately
                    if (metaData.query_id && userMessage) {
                        userMessage.query_id = metaData.query_id;
                        console.log("🔄 Updated user message with query_id:", metaData.query_id);
                    }
                } catch (e) {
                    console.error('❌ Failed to parse metadata chunk:', e, 'Raw:', parts[1]);
                    metaData = {}; // Ensure it's always an object
                }
            }

            fullResponse += chunk;
            console.log("🧩 Current fullResponse:", fullResponse); // DEBUG

            // Update with both status and accumulated response
            const messageContent = assistantBubble.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = `
                    <div class="status-container">
                        <span class="typing-cursor">${currentStatus}</span>
                        <div style="margin-top: 10px; opacity: 0.8;">${marked.parse(fullResponse + " ▌")}</div>
                    </div>
                `;
            }

            autoScrollToBottom();
        }

        // Clear timeout and mark as complete
        clearTimeout(timeoutId);
        responseComplete = true;

        // Final update - remove status and show only the response with actions
        assistantBubble.innerHTML = `
            <div class="message-content">${marked.parse(fullResponse)}</div>
            <div class="message-actions">
                <button class="action-btn copy" title="Copy message content">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>
        `;

        // Attach copy functionality
        const copyBtn = assistantBubble.querySelector('.action-btn.copy');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const content = extractTextContent(assistantBubble);
                if (content && content.trim()) {
                    copyToClipboard(content);
                    showCopyFeedback(copyBtn);
                }
            });
        }

        enhanceCodeBlocks(assistantBubble);

        // Add to chat history
        const assistantMessage = {
            role: 'assistant',
            content: fullResponse,
            user_query: query,
            query_id: userMessage.query_id || metaData.query_id,
            contexts: metaData.contexts|| [],
            messageId: assistantMessageId
        };

        state.chatHistory[projId].push(assistantMessage);

        if (metaData.conversation_id) {
            state.conversationId[projId] = metaData.conversation_id;
        }
        // Update user message with query_id from metadata
        if (metaData.query_id) {
            userMessage.query_id = metaData.query_id;
        }

        addFeedbackSection(assistantBubble, assistantMessage);

        autoScrollToBottom();

    } catch (error) {
        clearTimeout(timeoutId);
        responseComplete = true;

        assistantBubble.innerHTML = `
            <div class="message-content" style="color: var(--danger);">
                <strong>Error:</strong> ${error.message}
            </div>
            <div class="message-actions">
                <button class="action-btn copy" title="Copy error message">
                    <i class="fas fa-copy"></i> Copy
                </button>
            </div>
        `;

        const copyBtn = assistantBubble.querySelector('.action-btn.copy');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const content = `Error: ${error.message}`;
                copyToClipboard(content);
                showCopyFeedback(copyBtn);
            });
        }

        const errorMessage = {
            role: 'assistant',
            content: `Error: ${error.message}`,
            user_query: query,
            messageId: assistantMessageId
        };
        state.chatHistory[projId].push(errorMessage);

        autoScrollToBottom();
    } finally {
        // Re-enable send UI only after assistant finished (success or error)
        enableSendUI();
    }
}

    function addFeedbackSection(bubble, message) { 
        console.log('Message in addFeedbackSection:', message); // DEBUG
        console.log('Contexts in addFeedbackSection:', message.contexts); // DEBUG
        const feedbackDiv = document.createElement('div'); 
        feedbackDiv.className = 'feedback-section'; 
        feedbackDiv.innerHTML = `
            <span class="feedback-text">Was this helpful?</span>
            <button class="feedback-btn" data-type="up">👍</button>
            <button class="feedback-btn" data-type="down">👎</button>
        `; 
        bubble.appendChild(feedbackDiv); 
        
        feedbackDiv.addEventListener('click', (e) => { 
            if (e.target.matches('.feedback-btn')) { 
                const feedbackType = e.target.dataset.type;
                
                // Show comment input instead of immediately submitting
                feedbackDiv.innerHTML = `
                    <div class="feedback-comment-section">
                        <span class="feedback-thanks">Thanks! Any additional comments?</span>
                        <textarea class="feedback-comment" placeholder="Optional: Share your thoughts..."></textarea>
                        <div class="feedback-actions">
                            <button class="feedback-submit-btn" data-type="${feedbackType}">Submit Feedback</button>
                            <button class="feedback-cancel-btn">Cancel</button>
                        </div>
                    </div>
                `;
                
                // Add event listeners for the new buttons
                feedbackDiv.querySelector('.feedback-submit-btn').addEventListener('click', () => {
                    const comment = feedbackDiv.querySelector('.feedback-comment').value;
                    submitFeedback(message, feedbackType, comment);
                    feedbackDiv.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
                });
                
                feedbackDiv.querySelector('.feedback-cancel-btn').addEventListener('click', () => {
                    // Restore original feedback buttons
                    feedbackDiv.innerHTML = `
                        <span class="feedback-text">Was this helpful?</span>
                        <button class="feedback-btn" data-type="up">👍</button>
                        <button class="feedback-btn" data-type="down">👎</button>
                    `;
                    // Re-attach event listeners to the new buttons
                    attachFeedbackListeners(feedbackDiv, message);
                });
            } 
        }); 
    }

    // Helper function to re-attach event listeners
    function attachFeedbackListeners(feedbackDiv, message) {
        feedbackDiv.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const feedbackType = e.target.dataset.type;
                
                // Show comment input for detailed feedback
                feedbackDiv.innerHTML = `
                    <div class="feedback-comment-section">
                        <span class="feedback-thanks">Thanks! Any additional comments?</span>
                        <textarea class="feedback-comment" placeholder="Optional: Share your thoughts about the SQL query..."></textarea>
                        <div class="feedback-actions">
                            <button class="feedback-submit-btn" data-type="${feedbackType}">Submit Feedback</button>
                            <button class="feedback-cancel-btn">Cancel</button>
                        </div>
                    </div>
                `;
                
                // Submit feedback with comment
                feedbackDiv.querySelector('.feedback-submit-btn').addEventListener('click', () => {
                    const comment = feedbackDiv.querySelector('.feedback-comment').value;
                    submitEnhancedFeedback(message, feedbackType, comment);
                    feedbackDiv.innerHTML = '<span class="feedback-thanks">Thanks for your feedback!</span>';
                });
                
                // Cancel feedback
                feedbackDiv.querySelector('.feedback-cancel-btn').addEventListener('click', () => {
                    feedbackDiv.innerHTML = `
                        <span class="feedback-text">Was this helpful?</span>
                        <button class="feedback-btn" data-type="up">👍</button>
                        <button class="feedback-btn" data-type="down">👎</button>
                    `;
                    attachFeedbackListeners(feedbackDiv, message);
                });
            });
        });
    }
    
    async function submitFeedback(msg, feedbackType, comment = "") {
        try {
            console.log('Before submit - msg.contexts:', msg.contexts);
            console.log('Before submit - full msg:', msg); // DEBUG
            const payload = { 
                query_id: msg.query_id || "", 
                project_id: parseInt(state.selectedProjectId), 
                query: msg.user_query, 
                answer: msg.content, 
                contexts: msg.contexts, 
                feedback: feedbackType,
                comment: comment, // Add comment here
                user_id: state.user.user_id 
            };
            await apiRequest('/feedback', { method: 'POST', body: JSON.stringify(payload) });
        } catch(error) {
            console.error('Failed to submit feedback:', error);
        }
    }

    // History & Analytics Logic
    async function loadProjectsForHistoryPage() { 
        try { 
            state.projects = await apiRequest('/projects'); 
            const selector = $('#history-project-selector'); 
            selector.innerHTML = '<option value="">-- Select Project --</option>'; 
            state.projects.forEach(p => { 
                selector.innerHTML += `<option value="${p.project_id}">${p.project_name}</option>`; 
            }); 
            
            // Show welcome message when page first loads (regardless of projects)
            $('#history-transcript').innerHTML = `
                <div class="chat-bubble assistant-bubble welcome-bubble">
                    <div class="welcome-content">
                        <div class="welcome-text">
                            👋<br>
                            <strong>Welcome to Query History!</strong><br>
                            Select a project and conversation to view past interactions and continue where you left off.
                        </div>
                    </div>
                </div>
            `;
            
            // Clear conversation selector
            $('#history-conversation-selector').innerHTML = '<option value="">-- Select Conversation --</option>';
            
        } catch(error) {
            console.error("Failed to load projects for history page:", error);
        } 
    }
    async function loadConversationsForHistory() { 
        const projectId = $('#history-project-selector').value; 
        const continueContainer = $('#continue-conversation-container');
        
        // Hide continue button when switching projects
        continueContainer.style.display = 'none';
        
        if (!projectId) {
            // Show welcome message when no project is selected
            $('#history-transcript').innerHTML = `
                <div class="chat-bubble assistant-bubble welcome-bubble">
                    <div class="welcome-content">
                        <div class="welcome-text">
                            👋<br>
                            <strong>Welcome to Query History!</strong><br>
                            Select a project and conversation to view past interactions and continue where you left off.
                        </div>
                    </div>
                </div>
            `;
            return; 
        }

        
        try { 
            const history = await apiRequest(`/queryhistory/${projectId}`); 
            const convSelector = $('#history-conversation-selector'); 
            convSelector.innerHTML = '<option value="">-- Select Conversation --</option>'; 
            
            if (history.length === 0) {
                // Show message when no conversations exist for selected project
                $('#history-transcript').innerHTML = `
                    <div class="chat-bubble assistant-bubble welcome-bubble">
                        <div class="welcome-content">
                            <div class="welcome-text">
                                👋<br>
                                <strong>No Conversations Found</strong><br>
                                This project doesn't have any query history yet. Start a conversation in the Query page!
                            </div>
                        </div>
                    </div>
                `;
                return;
            }
            
            // Group ALL queries by conversation_id for the full conversation data
            const allConversations = history.reduce((acc, q) => { 
                acc[q.conversation_id] = acc[q.conversation_id] || []; 
                acc[q.conversation_id].push(q); 
                return acc; 
            }, {});
            
            // For dropdown, we only need the first query from each conversation
            const dropdownConversations = {};
            for (const convId in allConversations) {
                // Sort by creation date to get the actual first query
                const sortedConversation = allConversations[convId].sort((a, b) => 
                    new Date(a.created_at) - new Date(b.created_at)
                );
                dropdownConversations[convId] = [sortedConversation[0]]; // Only first query
            }
            
            console.log('All conversations data:', allConversations); // Debug
            console.log('Dropdown conversations:', dropdownConversations); // Debug
            
            // Store ALL conversations for when user selects one
            convSelector.dataset.conversations = JSON.stringify(allConversations); 
            
            // For dropdown, show only first query
            for (const convId in dropdownConversations) { 
                const firstQuery = dropdownConversations[convId][0].query; 
                convSelector.innerHTML += `<option value="${convId}">"${firstQuery.substring(0, 50)}..."</option>`; 
            } 
        } catch(error) {
            console.error("Failed to load conversations:", error);
        } 
    }
    function renderConversationTranscript() {
        const convId = $('#history-conversation-selector').value;
        const transcriptDiv = $('#history-transcript');
        const continueContainer = $('#continue-conversation-container');
        
        // Hide continue button when no conversation is selected
        continueContainer.style.display = 'none';
        
        if (!convId) { 
            transcriptDiv.innerHTML = `
            <div class="chat-bubble assistant-bubble welcome-bubble">
                <div class="welcome-content">
                    <div class="welcome-text">
                        👋<br>
                        <strong>Welcome to Query History!</strong><br>
                        Select a project and conversation to view past interactions and continue where you left off.
                    </div>
                </div>
            </div>
            `;
            return;
        }
        
        const conversations = JSON.parse($('#history-conversation-selector').dataset.conversations);
        let messages = conversations[convId];
        
        // Sort messages by creation date to show in chronological order
        messages = messages.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
        

        // DEBUG: Log the raw data from database
        console.log('🔍 RAW MESSAGES FROM DATABASE:', messages);
        console.log('📊 Total messages in conversation:', messages.length);
        
        messages.forEach((msg, index) => {
            console.log(`📝 Message ${index + 1}:`, {
                query_id: msg.id,
                query: msg.query?.substring(0, 50) + '...',
                answer: msg.answer?.substring(0, 50) + '...',
                created_at: msg.created_at
            });
        });
        
        // Clear the transcript and add header
        transcriptDiv.innerHTML = '<h3>Conversation</h3>';
        
        // Build HTML content first, then set it all at once
        let messagesHTML = '';
        
        // Display ALL messages in chronological order
        messages.forEach(msg => {
            messagesHTML += `<div class="chat-bubble user-bubble">${msg.query}</div>`;
            messagesHTML += `<div class="chat-bubble assistant-bubble">${marked.parse(msg.answer)}</div>`;
        });
        
        transcriptDiv.innerHTML += messagesHTML;
        

        continueContainer.style.display = 'block';

        const continueBtn = $('#continue-conversation-btn');

        // Update the button click handler
        continueBtn.onclick = () => {
            // Set selected project & conversation ID
            const projectId = $('#history-project-selector').value;
            state.selectedProjectId = projectId;
            state.conversationId[projectId] = convId;

            // Load chat history for this conversation in correct INTERLEAVED order
            state.chatHistory[projectId] = [];
            
            // Process messages in chronological order and interleave user/assistant messages
            messages.forEach(msg => {
                const queryId = msg.id;
                // Add user message
                state.chatHistory[projectId].push({
                    role: 'user',
                    content: msg.query,
                    user_query: msg.query,
                    query_id: queryId,
                    originalContent: msg.query,
                    contexts: msg.contexts || []
                });
                
                // Add assistant message immediately after user message
                state.chatHistory[projectId].push({
                    role: 'assistant',
                    content: msg.answer,
                    user_query: msg.query,
                    query_id: queryId,
                    contexts: msg.contexts || []
                });
            });

            // Navigate to chat page
            navigateTo('query');
            renderChatHistory();
        };
        enhanceCodeBlocksInContainer(transcriptDiv);
    }

    async function loadAnalyticsData() { 
        try { 
            const data = await apiRequest('/analytics-data'); 
            renderKPIs(data); 
            renderQueryTrendChart(data.queries); 
            renderPerfKPIs(data.queries); 
            renderFeedbackChart(data.feedback);
            renderUserAnalytics(data.queries, data.users); // New function for user analytics tab
            renderQueryAnalytics(data.queries, data.projects); // New function for query analytics tab
            renderQualityMetrics(data.queries);
        } catch(error) {} 
    }
    function renderKPIs(data) { $('#kpi-container').innerHTML = `<div class="card"><h4>Total Projects</h4><p class="kpi-value">${data.projects.length}</p></div><div class="card"><h4>Total Queries</h4><p class="kpi-value">${data.queries.length}</p></div><div class="card"><h4>Unique Users</h4><p class="kpi-value">${data.users.length}</p></div><div class="card"><h4>Feedback Entries</h4><p class="kpi-value">${data.feedback.length}</p></div>`; }
    function renderQueryTrendChart(queries) { if (state.charts.queryTrend) state.charts.queryTrend.destroy(); const ctx = $('#query-trend-chart').getContext('2d'); const dailyCounts = queries.reduce((acc, q) => { const date = new Date(q.created_at).toISOString().split('T')[0]; acc[date] = (acc[date] || 0) + 1; return acc; }, {}); state.charts.queryTrend = new Chart(ctx, { type: 'line', data: { labels: Object.keys(dailyCounts).sort(), datasets: [{ label: 'Daily Query Volume', data: Object.values(dailyCounts), borderColor: 'var(--primary)', tension: 0.1 }] }, options: { responsive: true, maintainAspectRatio: false } }); }
    function renderPerfKPIs(queries) { 
        const responseTimes = queries.map(q => q.response_time).filter(Boolean);
        const avgRespTime = responseTimes.length ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length : 0;
        
        const fallbackCount = queries.filter(q => q.is_fallback).length;
        const fallbackRate = queries.length ? (fallbackCount / queries.length) * 100 : 0;
        
        const retryCount = queries.filter(q => q.attempts > 1).length;
        const retryRate = queries.length ? (retryCount / queries.length) * 100 : 0;

        $('#perf-kpi-container').innerHTML = `
            <div class="card">
                <h4><span class="icon-performance"></span> Performance Metrics</h4>
                <div class="metric-grid">
                    <div class="metric-item">
                        <span class="metric-label">Avg Response Time</span>
                        <span class="metric-value">${avgRespTime.toFixed(2)}s</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">LLM Fallback Rate</span>
                        <span class="metric-value">${fallbackRate.toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Retry Rate</span>
                        <span class="metric-value">${retryRate.toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    function renderFeedbackChart(feedback) {
        if (state.charts.feedback) state.charts.feedback.destroy();
        if (!feedback || feedback.length === 0) {
            $('#feedback-chart').closest('.chart-wrapper').innerHTML = '<p>No feedback data yet.</p>';
            return;
        }
        const ctx = $('#feedback-chart').getContext('2d');
        const computedStyles = getComputedStyle(document.documentElement);
        const successColor = computedStyles.getPropertyValue('--success').trim();
        const dangerColor = computedStyles.getPropertyValue('--danger').trim();
        console.log("Chart colors loaded:", { successColor, dangerColor }); // DEBUG
        const counts = feedback.reduce((acc, f) => { acc[f.feedback] = (acc[f.feedback] || 0) + 1; return acc; }, { up: 0, down: 0 });
        state.charts.feedback = new Chart(ctx, {
            type: 'pie', data: { labels: ['Up Votes', 'Down Votes'], datasets: [{ data: [counts.up, counts.down], backgroundColor: [successColor, dangerColor], borderColor: 'var(--card-bg)', borderWidth: 2 }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    // --- New Query Analytics Function ---
    function renderQueryAnalytics(queries, projects) {
        const container = $('#query-analytics-container');
        if (!queries || queries.length === 0) {
            container.innerHTML = '<p>No query analytics data available.</p>';
            return;
        }

        // Calculate queries per project
        const projectStats = {};
        queries.forEach(query => {
            const projectId = query.project_id;
            if (!projectStats[projectId]) {
                projectStats[projectId] = {
                    queryCount: 0,
                    projectName: projects.find(p => p.id === projectId)?.name || `Project ${projectId}`
                };
            }
            projectStats[projectId].queryCount++;
        });

        // Calculate conversation depth
        const conversationStats = {};
        queries.forEach(query => {
            const convId = query.conversation_id;
            if (!conversationStats[convId]) {
                conversationStats[convId] = 0;
            }
            conversationStats[convId]++;
        });
        
        const conversationDepths = Object.values(conversationStats);
        const avgConversationDepth = conversationDepths.reduce((a, b) => a + b, 0) / conversationDepths.length;

        // Prepare data for charts
        const projectData = Object.values(projectStats);
        const projectNames = projectData.map(project => project.projectName);
        const projectQueryCounts = projectData.map(project => project.queryCount);

        // Render query analytics
        container.innerHTML = `
            <div class="card">
                <h4><span class="icon-folder"></span> Queries per Project</h4>
                <div class="chart-wrapper">
                    <canvas id="queries-per-project-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h4><span class="icon-analytics"></span> Conversation Depth Distribution</h4>
                <div class="chart-wrapper">
                    <canvas id="conversation-depth-chart"></canvas>
                </div>
            </div>
            
        `;

        // Create charts
        createBarChart('queries-per-project-chart', projectNames, projectQueryCounts, 'Queries per Project', 'Number of Queries');
        createHistogramChart('conversation-depth-chart', conversationDepths, 'Conversation Depth Distribution', 'Number of Queries per Conversation');
    }

    // Helper function to create histogram charts
    function createHistogramChart(canvasId, data, title, xAxisLabel) {
        const ctx = $(`#${canvasId}`).getContext('2d');
        if (state.charts[canvasId]) state.charts[canvasId].destroy();
        
        // Create histogram bins
        const maxDepth = Math.max(...data);
        const binSize = Math.ceil(maxDepth / 10);
        const bins = {};
        
        for (let i = 1; i <= maxDepth; i += binSize) {
            const binLabel = `${i}-${i + binSize - 1}`;
            bins[binLabel] = 0;
        }
        
        data.forEach(depth => {
            const binIndex = Math.floor((depth - 1) / binSize);
            const binLabel = `${(binIndex * binSize) + 1}-${(binIndex * binSize) + binSize}`;
            bins[binLabel] = (bins[binLabel] || 0) + 1;
        });

        const binLabels = Object.keys(bins);
        const binCounts = Object.values(bins);

        state.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: 'Number of Conversations',
                    data: binCounts,
                    backgroundColor: 'rgba(106, 27, 154, 0.6)',
                    borderColor: 'rgba(106, 27, 154, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Conversations'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: xAxisLabel
                        }
                    }
                }
            }
        });
    }

    // --- New User Analytics Function with Charts ---
    function renderUserAnalytics(queries, users) {
        const container = $('#user-analytics-container');
        if (!queries || queries.length === 0) {
            container.innerHTML = '<p>No user analytics data available.</p>';
            return;
        }

        // Create a mapping of user_id to username
        const userMap = {};
        users.forEach(user => {
            userMap[user.id] = user.username;
        });

        // Calculate user analytics
        const userStats = {};
        const sessionStats = {};
        
        queries.forEach(query => {
            const userId = query.user_id;
            const sessionId = query.conversation_id;
            const username = userMap[userId] || `User ${userId}`;
            
            // Initialize user stats
            if (!userStats[userId]) {
                userStats[userId] = {
                    queryCount: 0,
                    sessions: new Set(),
                    totalResponseTime: 0,
                    username: username
                };
            }
            
            // Initialize session stats
            if (!sessionStats[sessionId]) {
                sessionStats[sessionId] = {
                    queryCount: 0,
                    userId: userId
                };
            }
            
            // Update counts
            userStats[userId].queryCount++;
            userStats[userId].sessions.add(sessionId);
            userStats[userId].totalResponseTime += query.response_time || 0;
            
            sessionStats[sessionId].queryCount++;
        });

        // Prepare data for charts
        const userData = Object.values(userStats);
        
        // Sort by query count (descending) for better chart display
        userData.sort((a, b) => b.queryCount - a.queryCount);
        
        const usernames = userData.map(user => user.username);
        const queryCounts = userData.map(user => user.queryCount);
        const sessionCounts = userData.map(user => user.sessions.size);

        // Calculate averages
        const totalUsers = userData.length;
        const totalSessions = Object.keys(sessionStats).length;
        const queriesPerUser = queryCounts.reduce((sum, count) => sum + count, 0) / totalUsers;
        const sessionsPerUser = sessionCounts.reduce((sum, count) => sum + count, 0) / totalUsers;
        const avgQueriesPerSession = queries.length / totalSessions;

        // Render user analytics with charts
        container.innerHTML = `
            <div class="card">
                <h4><span class="icon-analytics"></span> Queries per User</h4>
                <div class="chart-wrapper">
                    <canvas id="queries-per-user-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h4><span class="icon-analytics"></span> Sessions per User</h4>
                <div class="chart-wrapper">
                    <canvas id="sessions-per-user-chart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h4><span class="icon-trend"></span> User Engagement Summary</h4>
                <div class="metric-grid">
                    <div class="metric-item">
                        <span class="metric-label">Avg Queries per User</span>
                        <span class="metric-value">${queriesPerUser.toFixed(1)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Avg Sessions per User</span>
                        <span class="metric-value">${sessionsPerUser.toFixed(1)}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Avg Queries per Session</span>
                        <span class="metric-value">${avgQueriesPerSession.toFixed(1)}</span>
                    </div>
                </div>
            </div>
        `;

        // Create charts
        createBarChart('queries-per-user-chart', usernames, queryCounts, 'Queries per User', 'Number of Queries');
        createBarChart('sessions-per-user-chart', usernames, sessionCounts, 'Sessions per User', 'Number of Sessions');
    }
    // Helper function to create bar charts
    function createBarChart(canvasId, labels, data, title, yAxisLabel) {
        const ctx = $(`#${canvasId}`).getContext('2d');
        if (state.charts[canvasId]) state.charts[canvasId].destroy();
        
        state.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: yAxisLabel,
                    data: data,
                    backgroundColor: 'rgba(63, 13, 102, 0.6)',
                    borderColor: 'rgba(63, 13, 102, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: yAxisLabel
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Users'
                        }
                    }
                }
            }
        });
    }
    // --- New Quality Metrics Function ---
    function renderQualityMetrics(queries) {
        if (!queries || queries.length === 0) return;

        // Filter out null/undefined values
        const validQueries = queries.filter(q => 
            q.static_exec !== null && 
            q.static_exec !== undefined &&
            q.claim_support !== null &&
            q.claim_support !== undefined &&
            q.precision1 !== null &&
            q.precision1 !== undefined &&
            q.query_coverage !== null &&
            q.query_coverage !== undefined
        );

        if (validQueries.length === 0) return;

        // Calculate metrics
        const staticExecScores = validQueries.map(q => q.static_exec);
        const claimSupportScores = validQueries.map(q => q.claim_support);
        const precisionScores = validQueries.map(q => q.precision1);
        const queryCoverageScores = validQueries.map(q => q.query_coverage);

        const overallAccuracy = (staticExecScores.reduce((a, b) => a + b, 0) / staticExecScores.length + 
                                claimSupportScores.reduce((a, b) => a + b, 0) / claimSupportScores.length) / 2;
        
        const retrievalQuality = precisionScores.reduce((a, b) => a + b, 0) / precisionScores.length;
        const queryUnderstanding = queryCoverageScores.reduce((a, b) => a + b, 0) / queryCoverageScores.length;

        // Render quality metrics in performance tab
        const qualityMetricsHTML = `
            <div class="card">
                <h4><span class="icon-analytics"></span> Quality Metrics</h4>
                <div class="metric-grid">
                    <div class="metric-item">
                        <span class="metric-label">Overall Accuracy</span>
                        <span class="metric-value">${(overallAccuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Retrieval Quality</span>
                        <span class="metric-value">${(retrievalQuality * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;

        // Add to performance tab
        $('#perf-kpi-container').innerHTML += qualityMetricsHTML;
    }

    // Tab Switching
    function handleTabSwitch(e) {
        const parent = e.target.closest('.page-content');
        parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        parent.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
        parent.querySelector(`#${e.target.dataset.tab}-tab-content`).style.display = 'block';
    }

    
    // =============================================
// ENHANCED MESSAGE ACTIONS - COPY, EDIT, REGENERATE
// =============================================



/**
 * Enhanced code block enhancement with copy buttons
 */
function enhanceCodeBlocks(messageDiv) {
    const codeBlocks = messageDiv.querySelectorAll('pre');
    
    codeBlocks.forEach(pre => {
        // Skip if already has copy button
        if (pre.querySelector('.pre-copy-btn')) return;
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'pre-copy-btn';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        copyBtn.title = 'Copy code';
        
        copyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const code = pre.querySelector('code')?.textContent || pre.textContent;
            copyToClipboard(code);
            showCopyFeedback(copyBtn);
        });
        
        pre.appendChild(copyBtn);
    });
}

/**
 * Enable editing for a user message
 */
function enableMessageEditing(messageDiv, originalContent) {
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return;

    // Replace content with editable textarea
    const textarea = document.createElement('textarea');
    textarea.className = 'editable-message';
    textarea.value = originalContent;
    textarea.rows = 3;
    
    contentDiv.parentNode.replaceChild(textarea, contentDiv);
    
    // Transform action buttons
    const editBtn = messageDiv.querySelector('.action-btn.edit');
    const regenerateBtn = messageDiv.querySelector('.action-btn.regenerate');
    const copyBtn = messageDiv.querySelector('.action-btn.copy');
    
    if (editBtn) {
        editBtn.innerHTML = '<i class="fas fa-check"></i> Save';
        editBtn.title = 'Save changes';
        editBtn.classList.add('save-mode');
        
        // Replace button to clear events
        const newEditBtn = editBtn.cloneNode(true);
        editBtn.parentNode.replaceChild(newEditBtn, editBtn);
        
        newEditBtn.addEventListener('click', () => {
            saveEditedMessage(messageDiv, textarea.value);
        });
    }

    // Hide other buttons during editing
    if (regenerateBtn) regenerateBtn.style.display = 'none';
    if (copyBtn) copyBtn.style.display = 'none';
    
    // Add edit actions container
    const editActions = document.createElement('div');
    editActions.className = 'edit-actions';
    editActions.innerHTML = `
        <button class="action-btn cancel-edit">
            <i class="fas fa-times"></i> Cancel
        </button>
    `;
    
    messageDiv.querySelector('.message-actions').appendChild(editActions);
    
    // Cancel edit handler
    editActions.querySelector('.cancel-edit').addEventListener('click', () => {
        cancelMessageEditing(messageDiv, originalContent);
    });
    
    // Keyboard shortcuts
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            cancelMessageEditing(messageDiv, originalContent);
        } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            saveEditedMessage(messageDiv, textarea.value);
        }
    });
    
    // Focus and select
    textarea.focus();
    textarea.select();
}

/**
 * Save edited message and update database
 */
/**
 * Save edited message and trigger regeneration
 */
async function saveEditedMessage(messageDiv, newContent) {
    if (!newContent.trim()) {
        showError('Message cannot be empty', '#main-error');
        return;
    }

    const messageId = messageDiv.id;
    const projectId = state.selectedProjectId;
    
    if (!projectId) {
        showError('Please select a project first', '#main-error');
        return;
    }

    try {
        // Find the message in chat history
        const messageIndex = findMessageIndexInHistory(projectId, messageId);
        if (messageIndex === -1) {
            throw new Error('Message not found in history');
        }

        const message = state.chatHistory[projectId][messageIndex];
        
        if (!message.query_id) {
            throw new Error('No query ID found for this message');
        }

        // Update the message in chat history
        const originalContent = message.content;
        message.content = newContent;
        message.originalContent = originalContent; // Store original for reference
        message.isEdited = true;

        // Update the database
        if (message.role === 'user') {
            await updateQueryInDatabase(message.query_id, newContent, null);
            message.user_query = newContent;
            restoreMessageDisplay(messageDiv, newContent);
            // ✅ AUTO-REGENERATE AFTER EDITING USER MESSAGE
            console.log('🔄 Auto-regenerating after edit...');
            await regenerateResponse(messageDiv, newContent);
            
        } else if (message.role === 'assistant') {
            await updateQueryInDatabase(message.query_id, null, newContent);
            // For assistant messages, just update the display
            restoreMessageDisplay(messageDiv, newContent);
            showCopyFeedback(messageDiv.querySelector('.action-btn.edit'));
        }

    } catch (error) {
        console.error('Failed to save edited message:', error);
        showError('Failed to save changes', '#main-error');
        // Restore original content on error
        const originalMessage = state.chatHistory[projectId][findMessageIndexInHistory(projectId, messageId)];
        restoreMessageDisplay(messageDiv, originalMessage.originalContent || newContent);
    }
}

/**
 * Cancel message editing and restore original
 */
function cancelMessageEditing(messageDiv, originalContent) {
    restoreMessageDisplay(messageDiv, originalContent);
}

/**
 * Restore message display after editing (with regeneration in progress)
 */
function restoreMessageDisplay(messageDiv, content, isRegenerating = false) {
    // Remove edit actions
    const editActions = messageDiv.querySelector('.edit-actions');
    if (editActions) editActions.remove();
    
    // Remove textarea and restore content div
    const textarea = messageDiv.querySelector('.editable-message');
    if (textarea) {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isRegenerating) {
            // Show "Regenerating..." state
            contentDiv.innerHTML = `
                <div class="message-content">
                    <span class="typing-cursor">${content}</span>
                    <div class="regeneration-indicator">
                        <i class="fas fa-sync-alt fa-spin"></i> Regenerating...
                    </div>
                </div>
            `;
        } else {
            // Show final content
            contentDiv.innerHTML = marked.parse(escapeHtml(content));
        }
        
        textarea.parentNode.replaceChild(contentDiv, textarea);
    }
    
    if (!isRegenerating) {
        // Only restore action buttons when not regenerating
        const messageActions = messageDiv.querySelector('.message-actions');
        messageActions.innerHTML = `
            <button class="action-btn edit" title="Edit message"><i class="fas fa-edit"></i></button>
            <button class="action-btn regenerate" title="Regenerate response"><i class="fas fa-sync-alt"></i></button>
            <button class="action-btn copy" title="Copy message"><i class="fas fa-copy"></i></button>
        `;
        
        // Re-attach event handlers
        attachMessageActions(messageDiv, 'user', content);
    }
}

/**
 * Regenerate response for a message (including after edits)
 */
/**
 * Regenerate response for a message and REPLACE existing assistant message
 */
async function regenerateResponse(messageDiv, content) {
    const projectId = state.selectedProjectId;
    if (!projectId) {
        showError('Please select a project first', '#main-error');
        return;
    }

    // Find the user message that we're regenerating from
    const messageId = messageDiv.id;
    const messageIndex = findMessageIndexInHistory(projectId, messageId);
    
    if (messageIndex === -1) {
        showError('Message not found', '#main-error');
        return;
    }

    const userMessage = state.chatHistory[projectId][messageIndex];
    const originalQueryId = userMessage.query_id;

    if (!originalQueryId) {
        showError('Cannot regenerate: No query ID found', '#main-error');
        return;
    }

    // Add regeneration indicator to user message
    
    
    try {
        // Find the existing assistant message ID to replace
        const assistantMessageId = await findAssistantMessageId(projectId, originalQueryId);
        
        // Send the message for regeneration - this will REPLACE the existing assistant message
        await sendMessageForRegeneration(content, projectId, originalQueryId, assistantMessageId);
        
    } catch (error) {
        console.error('Regeneration failed:', error);
        showError('Failed to regenerate response', '#main-error');
        
    }
}

/**
 * Find message index in chat history
 */
function findMessageIndexInHistory(projectId, messageId) {
    if (!state.chatHistory[projectId]) return -1;
    
    return state.chatHistory[projectId].findIndex(msg => 
        msg.messageId === messageId || msg.elementId === messageId
    );
}

/**
 * Remove subsequent messages from history and UI (for clean regeneration)
 */
function removeSubsequentMessages(projectId, messageId) {
    if (!state.chatHistory[projectId]) return;
    
    const messageIndex = findMessageIndexInHistory(projectId, messageId);
    if (messageIndex !== -1) {
        // Only remove messages AFTER the assistant response
        // We want to keep: [User] -> [Assistant] structure
        const messagesToKeep = messageIndex + 2; // Keep user + assistant
        
        if (state.chatHistory[projectId].length > messagesToKeep) {
            state.chatHistory[projectId] = state.chatHistory[projectId].slice(0, messagesToKeep);
            
            // Remove from UI (only messages after the assistant)
            const messagesContainer = $('#chat-messages');
            const allMessages = messagesContainer.querySelectorAll('.message-container');
            
            for (let i = allMessages.length - 1; i >= messagesToKeep; i--) {
                allMessages[i].remove();
            }
        }
    }
}

 /**
 * Send message for regeneration and REPLACE existing assistant response
 */
async function sendMessageForRegeneration(content, projectId, existingQueryId) {
    const input = $('#chat-input');
    const originalValue = input.value;
    
    // Set input value
    input.value = content;
    
    try {
        // Store the existing query ID for the special API call
        state.regeneratingQueryId = existingQueryId;
        
        // Find and prepare to replace the existing assistant message
        const assistantMessageId = await findAssistantMessageId(projectId, existingQueryId);
        
        // Use a modified version that REPLACES the assistant message
        await handleRegenerationSubmit(content, projectId, existingQueryId, assistantMessageId);
        
    } catch (error) {
        console.error('Regeneration failed:', error);
        showError('Failed to regenerate response', '#main-error');
    } finally {
        // Restore input value and clear state
        input.value = originalValue;
        state.regeneratingQueryId = null;
    }
}

/**
 * Find the assistant message ID that corresponds to a user query
 */
async function findAssistantMessageId(projectId, queryId) {
    if (!state.chatHistory[projectId]) return null;
    
    // Find the user message with this query_id
    const userMessageIndex = state.chatHistory[projectId].findIndex(msg => 
        msg.query_id === queryId && msg.role === 'user'
    );
    
    if (userMessageIndex === -1 || userMessageIndex >= state.chatHistory[projectId].length - 1) {
        return null;
    }
    
    // The next message should be the assistant response
    const assistantMessage = state.chatHistory[projectId][userMessageIndex + 1];
    if (assistantMessage && assistantMessage.role === 'assistant') {
        return assistantMessage.messageId;
    }
    
    return null;
}
/**
 * Handle regeneration submit - UPDATES existing assistant message instead of creating new one
 */
/**
 * Handle regeneration submit - UPDATES existing assistant message instead of creating new one
 */
async function handleRegenerationSubmit(query, projectId, existingQueryId, assistantMessageIdToReplace = null) {
    if (!query || !projectId || !existingQueryId) return;

    const projId = projectId;

    disableSendUI();

    // Find the existing assistant message in chat history
    let existingAssistantMessage = null;
    let existingAssistantIndex = -1;
    
    if (state.chatHistory[projId]) {
        existingAssistantIndex = state.chatHistory[projId].findIndex(msg => 
            msg.query_id === existingQueryId && msg.role === 'assistant'
        );
        
        if (existingAssistantIndex !== -1) {
            existingAssistantMessage = state.chatHistory[projId][existingAssistantIndex];
        }
    }

    // Get or create assistant container
    let assistantContainer;
    let assistantMessageId;
    
    if (assistantMessageIdToReplace && $(`#${assistantMessageIdToReplace}`)) {
        // USE EXISTING container - don't replace the entire structure
        assistantMessageId = assistantMessageIdToReplace;
        assistantContainer = $(`#${assistantMessageIdToReplace}`);
        
        // ✅ FIX: Update content INSIDE the existing structure
        const messageContent = assistantContainer.querySelector('.message-content');
        const messageActions = assistantContainer.querySelector('.message-actions');
        
        if (messageContent) {
            messageContent.innerHTML = `<span class="typing-cursor">🔄 Regenerating response...</span>`;
        }
        
        // Hide actions during regeneration
        if (messageActions) {
            messageActions.style.display = 'none';
        }
        
    } else {
        // Fallback: create new container (shouldn't happen often)
        assistantMessageId = `msg-${Date.now()}-assistant-${Math.random().toString(36).substr(2, 5)}`;
        
        assistantContainer = document.createElement('div');
        assistantContainer.className = 'message-container assistant-container';
        assistantContainer.id = assistantMessageId;
        
        assistantContainer.innerHTML = `
            <div class="chat-bubble assistant-bubble">
                <div class="message-content">
                    <span class="typing-cursor">🔄 Regenerating response...</span>
                </div>
                <div class="message-actions" style="display: none;">
                    <button class="action-btn copy" title="Copy message content">
                        <i class="fas fa-copy"></i> Copy
                    </button>
                </div>
            </div>
        `;
        
        $('#chat-messages').appendChild(assistantContainer);
    }

    let currentStatus = "🔄 Regenerating response...";
    autoScrollToBottom();

    try {
        // Special API call for regeneration that updates existing query
        const response = await apiRequest('/chat/regenerate', {
            method: 'POST',
            stream: true,
            body: JSON.stringify({
                project_id: parseInt(projId),
                query: query,
                query_id: existingQueryId, // Tell backend to update existing query
                conversation_id: state.conversationId[projId] || null,
            }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";
        let metaData = {};

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            let chunk = decoder.decode(value, { stream: true });

            // Handle status updates
            if (chunk.includes("[[[STATUS]]]")) {
                currentStatus = chunk.replace("[[[STATUS]]]", "").trim();
                const messageContent = assistantContainer.querySelector('.message-content');
                if (messageContent) {
                    if (fullResponse) {
                        messageContent.innerHTML = `
                            <div class="status-container">
                                <span class="typing-cursor">${currentStatus}</span>
                                <div style="margin-top: 10px; opacity: 0.8;">${marked.parse(fullResponse + " ▌")}</div>
                            </div>
                        `;
                    } else {
                        messageContent.innerHTML = `<span class="typing-cursor">${currentStatus}</span>`;
                    }
                }
                autoScrollToBottom();
                continue;
            }

            // Handle metadata
            if (chunk.includes("[[[META]]]")) {
                const parts = chunk.split("[[[META]]]");
                chunk = parts[0];
                try {
                    const metaString = parts[1] || "{}";
                    console.log("🔍 Raw metadata string:", metaString); // DEBUG
                    metaData = JSON.parse(metaString);
                    console.log("✅ Parsed metadata:", metaData); // DEBUG
                    
                    // If meta contains conversation_id, persist immediately
                    if (metaData.conversation_id) {
                        state.conversationId[projId] = metaData.conversation_id;
                    }
                    console.log("🔄 Regeneration using query_id:", existingQueryId);
                } catch (e) {
                    console.error('❌ Failed to parse metadata chunk:', e, 'Raw:', parts[1]);
                    metaData = {}; // Ensure it's always an object
                }
            }

            fullResponse += chunk;

            // Update with both status and accumulated response
            const messageContent = assistantContainer.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = `
                    <div class="status-container">
                        <span class="typing-cursor">${currentStatus}</span>
                        <div style="margin-top: 10px; opacity: 0.8;">${marked.parse(fullResponse + " ▌")}</div>
                    </div>
                `;
            }

            autoScrollToBottom();
        }

        // ✅ FIX: Update content INSIDE existing structure, don't replace entire container
        const messageContent = assistantContainer.querySelector('.message-content');
        const messageActions = assistantContainer.querySelector('.message-actions');
        
        if (messageContent) {
            messageContent.innerHTML = marked.parse(fullResponse);
        }
        
        if (messageActions) {
            messageActions.style.display = 'flex';
            messageActions.innerHTML = `
                <button class="action-btn copy" title="Copy message content">
                    <i class="fas fa-copy"></i> Copy
                </button>
            `;
            
            // Re-attach copy functionality
            const copyBtn = messageActions.querySelector('.action-btn.copy');
            if (copyBtn) {
                copyBtn.addEventListener('click', () => {
                    const content = extractTextContent(assistantContainer);
                    if (content && content.trim()) {
                        copyToClipboard(content);
                        showCopyFeedback(copyBtn);
                    }
                });
            }
        }

        enhanceCodeBlocks(assistantContainer);

        // ✅ UPDATE existing assistant message in chat history (don't add new one)
        if (existingAssistantIndex !== -1) {
            // Update the existing assistant message
            state.chatHistory[projId][existingAssistantIndex] = {
                role: 'assistant',
                content: fullResponse,
                user_query: query,
                query_id: metaData.query_id || existingQueryId, // Same query_id as the user message
                contexts: metaData.contexts,
                messageId: assistantMessageId,
                isRegenerated: true
            };
        } else {
            // Fallback: add new message if existing not found
            const assistantMessage = {
                role: 'assistant',
                content: fullResponse,
                user_query: query,
                query_id: metaData.query_id || existingQueryId,
                contexts: metaData.contexts,
                messageId: assistantMessageId,
                isRegenerated: true
            };
            state.chatHistory[projId].push(assistantMessage);
        }

        // Update conversation ID if provided
        if (metaData.conversation_id) {
            state.conversationId[projId] = metaData.conversation_id;
        }

        // ✅ ADD FEEDBACK SECTION - SAME AS handleChatSubmit
        const assistantBubble = assistantContainer.querySelector('.assistant-bubble');
        if (assistantBubble) {
            // Remove existing feedback section if present
            const existingFeedback = assistantBubble.querySelector('.feedback-section');
            if (existingFeedback) {
                existingFeedback.remove();
            }
            addFeedbackSection(assistantBubble, state.chatHistory[projId][existingAssistantIndex !== -1 ? existingAssistantIndex : state.chatHistory[projId].length - 1]);
        }
        
        autoScrollToBottom();

    } catch (error) {
        console.error('Regeneration API error:', error);
        
        // ✅ FIX: Update content INSIDE existing structure for errors too
        const messageContent = assistantContainer.querySelector('.message-content');
        const messageActions = assistantContainer.querySelector('.message-actions');
        
        if (messageContent) {
            messageContent.innerHTML = `<div style="color: var(--danger);"><strong>Error:</strong> ${error.message}</div>`;
        }
        
        if (messageActions) {
            messageActions.style.display = 'flex';
            messageActions.innerHTML = `
                <button class="action-btn copy" title="Copy error message">
                    <i class="fas fa-copy"></i> Copy
                </button>
            `;
            
            const copyBtn = messageActions.querySelector('.action-btn.copy');
            if (copyBtn) {
                copyBtn.addEventListener('click', () => {
                    const content = `Error: ${error.message}`;
                    copyToClipboard(content);
                    showCopyFeedback(copyBtn);
                });
            }
        }

        // Update error in chat history too
        if (existingAssistantIndex !== -1) {
            state.chatHistory[projId][existingAssistantIndex] = {
                role: 'assistant',
                content: `Error: ${error.message}`,
                user_query: query,
                query_id: existingQueryId,
                messageId: assistantMessageId,
                isRegenerated: true,
                hasError: true
            };
        }
        autoScrollToBottom();
    } finally {
        // Re-enable send UI only after assistant finished (success or error)
        enableSendUI();
    }
}
/**
 * Update query in database (both query and answer)
 */
async function updateQueryInDatabase(queryId, newQuery = null, newAnswer = null) {
    if (!queryId) {
        console.warn('No query ID provided for update');
        return;
    }

    try {
        const updateData = {};
        if (newQuery !== null) updateData.query = newQuery;
        if (newAnswer !== null) updateData.answer = newAnswer;

        if (Object.keys(updateData).length === 0) {
            console.warn('No data to update');
            return;
        }

        // Call backend API to update the query
        await apiRequest(`/queries/${queryId}`, {
            method: 'PUT',
            body: JSON.stringify(updateData)
        });

        console.log('✅ Successfully updated query in database:', queryId);
        
    } catch (error) {
        console.error('❌ Failed to update query in database:', error);
        throw new Error(`Database update failed: ${error.message}`);
    }
}

/**
 * Copy to clipboard with enhanced feedback
 */
function copyToClipboard(text) {
    if (!text.trim()) return;
    
    navigator.clipboard.writeText(text).then(() => {
        console.log('Text copied to clipboard');
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
    });
}

/**
 * Show copy feedback animation
 */
function showCopyFeedback(button) {
    const feedback = document.createElement('div');
    feedback.className = 'copy-feedback';
    feedback.textContent = 'Copied!';
    
    document.body.appendChild(feedback);
    
    // Remove after animation
    setTimeout(() => {
        if (feedback.parentNode) {
            feedback.remove();
        }
    }, 2000);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


/**
 * Enhance all code blocks in a container
 */
function enhanceCodeBlocksInContainer(container) {
    const codeBlocks = container.querySelectorAll('.assistant-bubble pre');
    
    codeBlocks.forEach(pre => {
        if (pre.querySelector('.pre-copy-btn')) return;
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'pre-copy-btn';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        copyBtn.title = 'Copy code';
        
        copyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const code = pre.querySelector('code')?.textContent || pre.textContent;
            copyToClipboard(code);
            showCopyFeedback(copyBtn);
        });
        
        pre.appendChild(copyBtn);
    });
}

/**
 * FIXED: Extract text content from message div (handles both plain text and HTML)
 */
function extractTextContent(messageDiv) {
    const contentDiv = messageDiv.querySelector('.message-content');
    if (!contentDiv) return '';
    
    // For assistant messages with HTML content, get the text representation
    if (contentDiv.innerHTML !== contentDiv.textContent) {
        // Create a temporary div to extract text from HTML
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = contentDiv.innerHTML;
        return tempDiv.textContent || tempDiv.innerText || '';
    }
    
    // For plain text, return directly
    return contentDiv.textContent || contentDiv.innerText || '';
}

/**
 * FIXED: Enhanced message rendering with PROPER action buttons
 */
function renderEnhancedMessage(message, role, messageId = null) {
    // Create main container that holds both bubble and actions
    const messageContainer = document.createElement('div');
    messageContainer.className = `message-container ${role}-container`;
    
    if (messageId) {
        messageContainer.id = messageId;
    } else {
        messageContainer.id = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    // Use marked.parse for assistant messages, escapeHtml for user messages
    const content = role === 'assistant' ? marked.parse(message) : escapeHtml(message);
    
    // Create bubble (just the message content)
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}-bubble`;
    bubble.innerHTML = `<div class="message-content">${content}</div>`;
    
    // Create actions (separate element outside the bubble)
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.innerHTML = `
        ${role === 'user' ? `
            <button class="action-btn edit" title="Edit message">
                <i class="fas fa-edit"></i>
            </button>
            <button class="action-btn regenerate" title="Regenerate response">
                <i class="fas fa-sync-alt"></i>
            </button>
        ` : ''}
        <button class="action-btn copy" title="Copy message content">
            <i class="fas fa-copy"></i>
        </button>
    `;

    // Assemble the container
    messageContainer.appendChild(bubble);
    messageContainer.appendChild(actions);

    // Attach action handlers to the container (not the bubble)
    attachMessageActions(messageContainer, role, message);

    return messageContainer;
}

/**
 * FIXED: Attach event handlers to message action buttons
 */
function attachMessageActions(messageDiv, role, originalContent) {
    const messageId = messageDiv.id;
    
    // Copy functionality - WORKS FOR ALL MESSAGES
    const copyBtn = messageDiv.querySelector('.action-btn.copy');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const content = extractTextContent(messageDiv);
            if (content && content.trim()) {
                copyToClipboard(content);
                showCopyFeedback(copyBtn);
            } else {
                console.warn('No content to copy from message:', messageDiv);
            }
        });
    }

    // Edit functionality (user messages only)
    if (role === 'user') {
        const editBtn = messageDiv.querySelector('.action-btn.edit');
        if (editBtn) {
            editBtn.addEventListener('click', () => {
                enableMessageEditing(messageDiv, originalContent);
            });
        }

        // Regenerate functionality
        const regenerateBtn = messageDiv.querySelector('.action-btn.regenerate');
        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                regenerateResponse(messageDiv, originalContent);
            });
        }
    }

    // Enhanced code block copy buttons for assistant messages (BONUS)
    if (role === 'assistant') {
        enhanceCodeBlocks(messageDiv);
    }
}

function autoScrollToBottom() {
    const chatContainer = $('#chat-container');
    if (chatContainer) {
        // Small delay to ensure DOM is updated
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 10);
        
        // Additional scroll after a bit more time for stubborn cases
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 100);
    }
}


    // Jira connection handlers
   async function handleJiraConnect(e) {
        e.preventDefault(); // This is crucial to prevent form submission navigation
        
        const formData = {
            jira_url: $('#jira-url').value,
            username: $('#jira-username').value,
            api_token: $('#jira-api-token').value,
            is_default: $('#jira-is-default').checked
        };

        // Basic validation
        if (!formData.jira_url || !formData.username || !formData.api_token) {
            showError('Please fill in all Jira connection fields', '#auth-error');
            return;
        }

        try {
            const result = await apiRequest('/jira/connect', {
                method: 'POST',
                body: JSON.stringify(formData)
            });
            
            // Clear form and show success
            $('#jira-connect-form').reset();
            showError('✅ Successfully connected to Jira!', '#auth-error');
            
            // Reload connections and show browser
            await loadJiraConnections();
            
        } catch (error) {
            showError(`❌ Connection failed: ${error.message}`, '#auth-error');
        }
    }

    async function loadJiraConnections() {
        try {
            state.jiraConnections = await apiRequest('/jira/connections');
            renderJiraConnections();
            
            if (state.jiraConnections.length > 0) {
                $('#jira-browser').style.display = 'block';
                renderJiraConnectionSelector();
            }
        } catch (error) {
            console.error('Failed to load Jira connections:', error);
        }
    }

    function renderJiraConnections() {
        const container = $('#jira-connections-list');
        if (state.jiraConnections.length === 0) {
            container.innerHTML = '<p style="color: #666; font-style: italic;">No Jira connections found. Add one above.</p>';
            return;
        }

        container.innerHTML = state.jiraConnections.map(conn => `
            <div class="connection-item" style="border: 1px solid var(--border-color); padding: 1rem; margin: 0.5rem 0; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>${conn.jira_url}</strong><br>
                        <small>User: ${conn.username}</small>
                    </div>
                    <div>
                        ${conn.is_default ? '<span style="background: var(--success); color: white; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">Default</span>' : ''}
                    </div>
                </div>
                <small style="color: #666;">Connected: ${new Date(conn.created_at).toLocaleDateString()}</small>
            </div>
        `).join('');
    }

    function renderJiraConnectionSelector() {
        const selector = $('#jira-connection-selector');
        selector.innerHTML = '<option value="">Select Jira Connection</option>' +
            state.jiraConnections.map(conn => 
                `<option value="${conn.id}">${conn.jira_url} (${conn.username})${conn.is_default ? ' - Default' : ''}</option>`
            ).join('');
        
        // Auto-select the default connection if available
        const defaultConnection = state.jiraConnections.find(conn => conn.is_default);
        if (defaultConnection) {
            selector.value = defaultConnection.id;
            // Trigger change to load projects automatically
            selector.dispatchEvent(new Event('change'));
        }
    }

    async function loadJiraProjects() {
        const connectionId = $('#jira-connection-selector').value;
        if (!connectionId) return;

        try {
            state.jiraProjects = await apiRequest(`/jira/projects?connection_id=${connectionId}`);
            renderJiraProjects();
            state.currentJiraConnection = connectionId;
        } catch (error) {
            showError(error.message, '#auth-error');
        }
    }

    function renderJiraProjects() {
        const container = $('#jira-projects-list');
        if (state.jiraProjects.length === 0) {
            container.innerHTML = '<p>No projects found in Jira.</p>';
            return;
        }

        container.innerHTML = state.jiraProjects.map(project => `
            <div class="jira-project-card" data-key="${project.key}">
                <h4>${project.name}</h4>
                <p>${project.key} - ${project.description || 'No description'}</p>
            </div>
        `).join('');

        // Add click handlers
        $$('.jira-project-card').forEach(card => {
            card.addEventListener('click', () => loadJiraIssues(card.dataset.key));
        });
    }

    async function loadJiraIssues(projectKey) {
        if (!state.currentJiraConnection) return;

        try {
            state.jiraIssues = await apiRequest(
                `/jira/issues?connection_id=${state.currentJiraConnection}&project_key=${projectKey}`
            );
            renderJiraIssues();
            $('#current-project-name').textContent = projectKey;
            $('#jira-issues-section').style.display = 'block';
        } catch (error) {
            showError(error.message, '#auth-error');
        }
    }

    function renderJiraIssues() {
        const container = $('#jira-issues-list');
        if (state.jiraIssues.length === 0) {
            container.innerHTML = '<p>No issues found in this project.</p>';
            return;
        }

        container.innerHTML = state.jiraIssues.map(issue => `
            <div class="jira-issue-card">
                <div class="issue-content">
                    <div class="issue-header">
                        <span class="issue-key">${issue.key}</span>
                        <span class="issue-summary">${issue.summary}</span>
                    </div>
                    <div class="issue-meta">
                        <span>Type: ${issue.issue_type}</span>
                        <span>Status: ${issue.status}</span>
                        <span>Assignee: ${issue.assignee}</span>
                        <span>Attachments: ${issue.attachments.length}</span>
                    </div>
                </div>
                <button class="import-btn" data-issue='${JSON.stringify(issue).replace(/'/g, "&apos;")}'>
                    Import
                </button>
            </div>
        `).join('');

        // Add import button handlers
        $$('.import-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const issue = JSON.parse(btn.dataset.issue.replace(/&apos;/g, "'"));
                showImportModal(issue);
            });
        });
    }

    function showImportModal(issue) {
        state.selectedJiraIssue = issue;
        $('#import-issue-key').textContent = issue.key;
        $('#jira-import-modal').style.display = 'flex';
        
        // Set up radio button handlers when modal is shown
        $$('input[name="import-option"]').forEach(radio => {
            // Remove any existing listeners first
            radio.replaceWith(radio.cloneNode(true));
        });
        
        // Re-select the elements and add fresh listeners
        $$('input[name="import-option"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const showNewProject = e.target.value === 'new';
                $('#new-project-section').style.display = showNewProject ? 'block' : 'none';
                $('#existing-projects-select').style.display = showNewProject ? 'none' : 'block';
            });
        });
        
        // Also set up the create project button listener here
        const createBtn = $('#create-project-btn');
        if (createBtn) {
            createBtn.addEventListener('click', handleJiraCreateProject);
        }
        
        // Reset to default state
        $('input[name="import-option"][value="existing"]').checked = true;
        $('#new-project-section').style.display = 'none';
        $('#existing-projects-select').style.display = 'block';
    }

    async function loadProjectsForJiraImport() {
        try {
            state.projects = await apiRequest('/projects');
            const selector = $('#existing-projects-select');
            selector.innerHTML = '<option value="">Select Project</option>' +
                state.projects.map(p => `<option value="${p.project_id}">${p.project_name}</option>`).join('');
        } catch (error) {
            console.error('Failed to load projects for import:', error);
        }
    }

    async function confirmJiraImport() {
        if (!state.selectedJiraIssue) return;

        const useExisting = $('input[name="import-option"]:checked').value === 'existing';
        const targetProjectId = useExisting ? $('#existing-projects-select').value : null;
        const newProjectName = !useExisting ? $('#new-project-name-input').value : null;

        if ((useExisting && !targetProjectId) || (!useExisting && !newProjectName)) {
            showError('Please select a project or enter a new project name.', '#main-error'); // Changed here
            return;
        }

        try {
            const result = await apiRequest(
                `/jira/import-issue?connection_id=${state.currentJiraConnection}`,
                {
                    method: 'POST',
                    body: JSON.stringify({
                        issue_key: state.selectedJiraIssue.key,
                        target_project_id: targetProjectId,
                        new_project_name: newProjectName
                    })
                }
            );

            // Close modal
            $('#jira-import-modal').style.display = 'none';
            
            // Switch to query page with the imported context
            state.selectedProjectId = result.project_id;
            navigateTo('query');

            console.log("Jira Issue content: ",result.issue_context)
            
            // Pre-fill chat input with issue context
            $('#chat-input').value = `Regarding Jira issue ${state.selectedJiraIssue.key}: ${state.selectedJiraIssue.summary}\n\n${result.issue_context}`;
            
            console.log(`Successfully imported issue! ${result.attachments_processed} attachments processed.`, '#main-error'); // Changed here
            
        } catch (error) {
            showError(error.message, '#main-error'); // Changed here
        }
    }

    init();
});
