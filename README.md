# SQL-Generation

### PROBLEM STATEMENT: 

In modern data-driven organizations, business processes, ETL workflows, and analytics pipelines are often documented in various formats including PDFs, Word documents, Excel sheets, CSVs, and Jira tickets. These documents contain crucial information such as: 

 * Field mappings between source and target tables 

 * Transformation rules including conditional logic, derived columns, and aggregations 

 * Dependencies and relationships across multiple datasets 

Currently, converting these documents into SQL queries is manual and labor-intensive. Analysts or developers must carefully interpret the mapping documents, track cross-document references, and write SQL that accurately reflects the intended transformations. 

### The challenges of this manual process include: 

* Time-Consuming: Reading, understanding, and coding from multiple documents can take hours per project. 

* Error-Prone: Manual SQL generation is susceptible to mistakes, such as missing columns, incorrect joins, or faulty aggregations. 

* Non-Scalable: Each new project or update to existing documentation requires repeating the same manual effort. 

* Knowledge-Dependent: SQL generation heavily relies on individual expertise. New team members may struggle, leading to inconsistencies. 

* Lack of Feedback Incorporation: Existing processes do not systematically leverage previous corrections or user feedback to improve accuracy over time. 

As a result, organizations face delays, increased operational costs, and risks of inaccurate data transformations in critical business processes. 

### There is a need for an automated system that can: 

* Ingest multiple document types. 

* Understand hierarchical structures and relationships across documents. 

* Generate accurate SQL queries automatically. 

* Incorporate user feedback to continuously improve accuracy and efficiency. 

This project addresses these challenges by combining document ingestion, semantic search, LLM-based query generation, and a feedback-driven refinement loop to reduce manual effort, improve reliability, and accelerate SQL generation. 

## OBJECTIVE: 

* Automate SQL query generation from project documents, reducing manual effort and errors. 

* Support all common document types: PDF, Word, Excel, CSV, and TXT. 

* Generate accurate, standardized SQL for diverse transformation logic, including derived columns, conditional expressions, joins, and aggregations. 

* Enable semantic search and document reference resolution to provide context-aware answers. 

* Incorporate user feedback to refine SQL queries in real time and improve future results. 

* Minimize dependency on individual SQL expertise while improving productivity, consistency, and reliability. 

* Automate SQL generation from project documents. 

* Provide chat-like interface where users can interact naturally (ask questions, request SQL, refine). 

* Provide analytics dashboard to monitor usage, performance, and feedback. 

 
## APPROACH: 

The system is designed as a document ingestion, indexing, and LLM-based query platform with feedback-driven refinement: 

### 1. Project Creation 

Users can create projects to group and manage related documents for a specific workflow. 

### 2. Document Loading & Splitting 

Supports PDFs, Word, Excel, CSV, and TXT files. 

Documents are parsed using Unstructured and split into hierarchical chunks with metadata. 

Text elements respect headings and sections; tables and rows are preserved with schema metadata. 

### 3. Database Storage 

SQLite stores: 

Projects: project metadata. 

Documents: file info and processing status. 

Document Chunks: split content with metadata. 

### 4. Embeddings & Semantic Search 

Chunks are embedded using HuggingFace all-MiniLM-L6-v2 embeddings. 

Each project maintains a FAISS vectorstore for semantic retrieval. 

Semantic search allows retrieval of relevant content across the entire project. 

### 5. Query Retrieval 

Queries are processed in an agentic workflow: 

Detect mentioned document names. 

Retrieve relevant chunks from those documents. 

Retrieve additional project-wide chunks using FAISS. 

### 6. Gemini LLM Integration 

Structured context (query + chunks) is sent to Gemini LLM (gemini-2.5-flash). 

LLM generates coherent, context-aware SQL queries or explanations. 

### 7.Chat-like Interface

Users ask queries directly in a conversational chat UI. 

System detects intent (e.g., "Generate SQL", "Explain transformation"). 

Relevant document chunks retrieved and passed to LLM. 

SQL queries or answers are generated and returned in chat. 

History preserved per conversation (session-based). 

### 8. Feedback Mechanism 

Users provide feedback on generated SQL answers (positive 👍 or negative 👎). 

Users can provide feedback in the form of comments on generated SQL answers. 

Feedback is stored in the database along with query details and metadata. 

### 8. Workflow Summary 

Create Project → /project endpoint. 

Upload Documents → /upload endpoint; documents are split, indexed, and embedded. 

Query Documents → /query endpoint; returns LLM-generated SQL based on relevant chunks. 

Provide Feedback → thumbs up/down. 

### 9. Advantages

Hierarchical chunking ensures context is preserved. 

FAISS + embeddings provide semantic search across structured and unstructured content. 

Feedback loop improves accuracy over time, allowing real-time refinement. 

Project-based isolation supports multiple workflows independently. 



## Analytics Dashboard: 
 
### 1. Overview KPIs 

✅ Total Projects 

✅ Total Documents 

✅ Total Queries 

✅ Unique Conversations 

✅ Feedback Collected 

### 2. 🧑‍🤝‍🧑 User Analytics 

Queries per User 

Sessions per User (conversations per user) 

Average Queries per Session 

### 3. 📂 Query Analytics 

Queries per Project 

Conversation Depth (average queries per conversation + distribution histogram) 

Query Trends Over Time 

### 4. ⚡ Performance & Quality Metrics 

Average Response Time (sec) 

Latency Trend (response time over time) 

Retry Rate (% queries retried) 

Fallback Rate (% queries where fallback happened) 

### 5. 👍👎 Feedback Analytics 

Positive vs Negative Feedback Ratio 

Common Feedback Reasons (Top Keywords from comments) 

User Feedback Comments Explorer (table) 

### 6. 🔍 Conversation Explorer 

Filter by User ID 

Filter by Conversation ID (within user) 

Show Full Conversation (queries + answers) 
