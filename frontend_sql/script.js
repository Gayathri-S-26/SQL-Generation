document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION & STATE ---
    const API_BASE_URL = 'http://127.0.0.1:8000';
    let state = {
        token: localStorage.getItem('accessToken'), user: null, currentPage: 'query',
        projects: [], selectedProjectId: null, selectedProjectName: null,
        chatHistory: {}, conversationId: {}, charts: {},
    };

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
            if (!state.selectedProjectId) return alert("Please select a project first.");

            // Reset chat + conversation for the selected project
            state.chatHistory[state.selectedProjectId] = [];
            state.conversationId[state.selectedProjectId] = null;

            // Refresh UI - this will now show welcome message
            renderChatHistory();
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
    async function loadProjectDetails(polling = false) {
        $('#project-detail-title').textContent = `Documents for: ${state.selectedProjectName}`;
        try {
            const data = await apiRequest(`/project/${state.selectedProjectId}/documents`);
            const list = $('#document-list'); 
            list.innerHTML = '';

            if (data.documents.length === 0) { 
                list.innerHTML = '<p>No documents found. Upload one!</p>'; 
                return; 
            }

            let hasPending = false;
            data.documents.forEach(doc => {
                if (doc.status.toLowerCase() === 'pending') hasPending = true;
                list.innerHTML += `
                <div class="document-list-item">
                    <span class="doc-name">${doc.doc_name}</span>
                    <div class="doc-status">
                        <span>${doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}</span>
                        <button class="delete-btn" data-id="${doc.doc_id}">🗑️ Delete</button>
                    </div>
                </div>`;
            });

            $$('.delete-btn').forEach(btn => btn.addEventListener('click', handleDeleteDocument));

            // Poll if any document is still pending
            if (hasPending && !polling) {
                setTimeout(() => loadProjectDetails(true), 2000); // poll every 2 seconds
            }

        } catch(error) {}
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
                selector.value = state.selectedProjectId || state.projects[0].project_id; 
                state.selectedProjectId = selector.value; 
                state.selectedProjectName = selector.options[selector.selectedIndex].text; 
                renderChatHistory(); // This will now show welcome message if no chats
            } 
        } catch (error) {} 
    }
    function renderChatHistory() { 
        const messagesContainer = $('#chat-messages'); 
        messagesContainer.innerHTML = ''; 
        const history = state.chatHistory[state.selectedProjectId] || []; 

        console.log('History retrieved for rendering:', history);
        
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
            const bubble = document.createElement('div'); 
            bubble.className = `chat-bubble ${msg.role}-bubble`; 
            bubble.innerHTML = marked.parse(msg.content); 
            messagesContainer.appendChild(bubble); 
            if (msg.role === 'assistant') addFeedbackSection(bubble, msg); 
        }); 
        $('#chat-container').scrollTop = $('#chat-container').scrollHeight; 
        enhanceCodeBlocks();
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

    async function handleChatSubmit(e) { e.preventDefault(); const input = $('#chat-input'); const query = input.value.trim(); if (!query || !state.selectedProjectId) return; const projId = state.selectedProjectId; if (!state.chatHistory[projId]) state.chatHistory[projId] = []; state.chatHistory[projId].push({ role: 'user', content: query }); renderChatHistory(); input.value = ''; const assistantBubble = document.createElement('div'); assistantBubble.className = 'chat-bubble assistant-bubble'; assistantBubble.innerHTML = '<span class="typing-cursor">Thinking...</span>'; $('#chat-messages').appendChild(assistantBubble); $('#chat-container').scrollTop = $('#chat-container').scrollHeight; try { const response = await apiRequest('/chat', { method: 'POST', stream: true, body: JSON.stringify({ project_id: parseInt(projId), query: query, conversation_id: state.conversationId[projId] || null, }), }); const reader = response.body.getReader(); const decoder = new TextDecoder(); let fullResponse = ""; let metaData = {}; while (true) { const { done, value } = await reader.read(); if (done) break; let chunk = decoder.decode(value, { stream: true }); if (chunk.includes("[[[META]]]")) { const parts = chunk.split("[[[META]]]"); chunk = parts[0]; metaData = JSON.parse(parts[1]); console.log('🔍 META DATA RECEIVED:', metaData); } fullResponse += chunk; assistantBubble.innerHTML = marked.parse(fullResponse + " ▌"); $('#chat-container').scrollTop = $('#chat-container').scrollHeight; } assistantBubble.innerHTML = marked.parse(fullResponse); const assistantMessage = { role: 'assistant', content: fullResponse, user_query: query, query_id: metaData.query_id, contexts: metaData.contexts }; console.log('Assistant message created:', assistantMessage); state.chatHistory[projId].push(assistantMessage); if (metaData.conversation_id) {state.conversationId[projId] = metaData.conversation_id;} addFeedbackSection(assistantBubble, assistantMessage); } catch (error) { assistantBubble.innerHTML = `Error: ${error.message}`; assistantBubble.style.color = 'red'; } enhanceCodeBlocks();}
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
                // ... same code as above to show comment section
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
                // Add user message
                state.chatHistory[projectId].push({
                    role: 'user',
                    content: msg.query,
                    user_query: msg.query,
                    query_id: msg.query_id || null,
                    contexts: msg.contexts || []
                });
                
                // Add assistant message immediately after user message
                state.chatHistory[projectId].push({
                    role: 'assistant',
                    content: msg.answer,
                    user_query: msg.query,
                    query_id: msg.query_id || null,
                    contexts: msg.contexts || []
                });
            });

            // Navigate to chat page
            navigateTo('query');
            renderChatHistory();
        };
        enhanceCodeBlocks();
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

    init();
});
