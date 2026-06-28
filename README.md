# 📚 Production-Grade RAG System with Ollama

A fully Dockerized, CPU-friendly Retrieval-Augmented Generation (RAG) application for querying PDF documents locally.


## 🛠️ Tech Stack
| Component | Tool/Library |
|-----------|--------------|
| Frontend UI | Streamlit |
| LLM Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings Model | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama (qwen2.5:7b) |
| OCR Support | Tesseract, pdf2image, PyPDF |
| Containerization | Docker, Docker Compose |


## 📁 Project Structure
```
d:\rag demo
├── app.py                    # Streamlit frontend UI
├── ingest.py                 # PDF ingestion and ChromaDB indexing
├── rag.py                    # Core RAG pipeline (retrieval + generation)
├── Dockerfile                # Streamlit app container definition
├── docker-compose.yml        # Multi-container setup (Ollama + Streamlit)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── rag doc/                  # Local folder for your PDFs (mounted to /app/pdfs)
└── chroma_db/                # Local folder for ChromaDB persistence (mounted to /app/chroma_db)
```


## 🚀 How It Works
### End-to-End Workflow

```mermaid
graph TD
    A[User Uploads PDFs to rag doc] --> B[Click Index Documents in Streamlit]
    B --> C[Ingest PDFs: OCR if needed]
    C --> D[Split into Chunks (800 tokens, 150 overlap)]
    D --> E[Generate Embeddings with BAAI/bge-base-en-v1.5]
    E --> F[Store in ChromaDB]
    F --> G[User Asks Question in Streamlit]
    G --> H[MMR Retrieval (fetch_k=20, k=5)]
    H --> I[Cross-Encoder Reranking]
    I --> J[Assemble Context]
    J --> K[Generate Answer with Ollama qwen2.5:7b]
    K --> L[Show Answer + Retrieved Chunks]
```


## 🎯 Key Features
1. **Fully Dockerized**: No local Python/Ollama installation required!
2. **CPU-Only Friendly**: Works without NVIDIA GPU
3. **OCR Support**: Handles both text-based and scanned PDFs
4. **MMR Retrieval**: Reduces duplicate chunks
5. **Cross-Encoder Reranking**: Improves retrieval quality
6. **Persistent Storage**: ChromaDB and Ollama models survive container restarts
7. **Production-Ready UI**: Clean, user-friendly Streamlit interface


## 📋 Prerequisites
- Docker Desktop installed on Windows
- Your PDFs placed in `d:\rag demo\rag doc`


## 🚀 Quick Start

### Step 1: Prepare Your PDFs
Place your PDF documents in:
```
d:\rag demo\rag doc
```

### Step 2: Build & Run the App
Open a **Command Prompt (CMD)** in `d:\rag demo` and run:

```cmd
# Clean up old containers and volumes (optional but recommended)
docker compose down -v

# Build the images without cache
docker compose build --no-cache

# Start the application
docker compose up
```

### Step 3: Use the App
1. Wait for containers to start (first run takes time to download Ollama models)
2. Open your browser and go to: `http://localhost:8501`
3. Click **📥 Index Documents** in the sidebar (indexes all your PDFs)
4. Type a question about your PDFs and click **🔍 Get Answer**


## 📱 UI Sections
| Section | Purpose |
|---------|---------|
| 💭 Your Question | Shows the question you entered |
| 📑 Retrieved Context | Expandable chunks with source file, page number, and similarity score |
| 🤖 AI Answer | Final generated answer from qwen2.5:7b |


## 📝 Pipeline Details

### 1. Ingestion Pipeline
1. **PDF Loading**: Uses PyPDF to load PDFs; falls back to Tesseract OCR if text extraction yields too little text
2. **Chunking**: RecursiveCharacterTextSplitter with `chunk_size=800` and `chunk_overlap=150`
3. **Embedding**: BAAI/bge-base-en-v1.5 (small, fast, and good quality)
4. **Storage**: ChromaDB with persistent volume

### 2. Retrieval Pipeline
1. **MMR Retrieval**: Uses Maximal Marginal Relevance to balance relevance and diversity (fetch_k=20, k=5)
2. **Cross-Encoder Reranking**: Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to reorder chunks
3. **Context Assembly**: Formats chunks with source and page numbers for the LLM

### 3. Generation Pipeline
1. **Prompt Template**: Strictly instructs LLM to answer only from context
2. **LLM**: Ollama qwen2.5:7b (good quality, fast enough on CPU)
3. **Answer**: Natural language response with citations


## 📄 Required Environment Variables
All environment variables are set in `docker-compose.yml`:
- `PYTHONUNBUFFERED=1`: Ensures logs are visible immediately
- `OLLAMA_BASE_URL=http://ollama:11434`: Connects to Ollama service
- `OLLAMA_MODEL=qwen2.5:7b`: Specifies the LLM model


## 🛑 Stopping the Application
1. Press `Ctrl+C` in the CMD window where the app is running
2. Run this command to clean up:
```cmd
docker compose down
```


## ❓ FAQ
### Q: The first run is taking a really long time!
A: That's normal! First time setup downloads:
- Ollama image
- BAAI/bge-base-en-v1.5 embeddings model
- cross-encoder/ms-marco-MiniLM-L-6-v2 reranker
- qwen2.5:7b LLM

### Q: Do I need an NVIDIA GPU?
A: No! The system is configured to run entirely on CPU. It will work slower, but it works!

### Q: Can I use a different LLM model?
A: Yes! Change `OLLAMA_MODEL` in `docker-compose.yml` to any Ollama model (e.g., `llama3.1:8b`).


## 📜 License
This project is for educational and personal use.
