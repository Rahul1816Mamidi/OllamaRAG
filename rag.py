import os
import traceback
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "/app/chroma_db"
COLLECTION_NAME = "production_rag_docs"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Initialize components once at module load
try:
    print("🔄 Initializing cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("✅ Reranker initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize reranker: {e}")
    traceback.print_exc()
    reranker = None

def wait_for_ollama(max_retries: int = 30, retry_delay: int = 5) -> bool:
    """Wait for Ollama service to become available"""
    import requests
    print("🔍 Waiting for Ollama service...")
    for i in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Ollama service is online!")
                return True
        except Exception as e:
            print(f"⚠️ Attempt {i+1}/{max_retries}: Ollama not ready yet - {str(e)}")
        time.sleep(retry_delay)
    print("❌ Timed out waiting for Ollama service!")
    return False

def pull_ollama_model() -> bool:
    """Pull the specified Ollama model if not already present"""
    import requests
    print(f"🔄 Checking for model {OLLAMA_MODEL}...")
    try:
        # Check if model exists
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            if OLLAMA_MODEL in model_names:
                print(f"✅ Model {OLLAMA_MODEL} already exists!")
                return True
        
        # Pull the model
        print(f"📥 Pulling model {OLLAMA_MODEL}... this may take several minutes")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": OLLAMA_MODEL},
            stream=True,
            timeout=300
        )
        for line in response.iter_lines():
            if line:
                print(f"📥 {line.decode('utf-8')}")
        
        # Verify pull was successful
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]
        if OLLAMA_MODEL in model_names:
            print(f"✅ Model {OLLAMA_MODEL} pulled successfully!")
            return True
        else:
            print(f"❌ Model {OLLAMA_MODEL} not found after pull!")
            return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        traceback.print_exc()
        return False

def get_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store

def ask_question(question: str) -> Tuple[str, List[Document]]:
    """
    Returns:
        answer: Ollama's natural language answer
        docs: Retrieved and reranked Document objects
    """
    print(f"\n🔍 Processing question: {question}")
    
    try:
        vector_store = get_vector_store()
        collection = vector_store._collection
        total_chunks = collection.count()
        
        if total_chunks == 0:
            print("⚠️ No documents in vector store!")
            return ("Please index your documents first by clicking '📥 Index Documents' in the sidebar!", [])
        
        print(f"📊 Found {total_chunks} total chunks available")
        
        # Step 1: MMR retrieval using retriever pattern (fetch 20, return 5 initially)
        print("📌 Step 1: MMR retrieval (fetch_k=20, k=5)...")
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        mmr_docs = retriever.invoke(question)
        
        # Get relevance scores by querying directly
        results_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=5)
        mmr_scores = [score for doc, score in results_with_scores]
        
        if not mmr_docs:
            print("❌ No relevant chunks found with MMR")
            return ("No relevant information found in your documents.", [])
        
        # Step 2: Cross-encoder reranking
        print("📌 Step 2: Cross-encoder reranking (top 5)...")
        final_docs = mmr_docs
        final_scores = mmr_scores
        if reranker:
            doc_texts = [doc.page_content for doc in mmr_docs]
            pairs = [[question, text] for text in doc_texts]
            rerank_scores = reranker.predict(pairs)
            
            scored_docs = list(zip(mmr_docs, rerank_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, score in scored_docs]
            final_scores = [score for doc, score in scored_docs]
            print(f"✅ Reranking complete! Avg score: {sum(final_scores)/len(final_scores):.3f}")
        
        # Step 3: Assemble context with sources
        print("📌 Step 3: Assembling context and calling Ollama...")
        context_parts = []
        for i, doc in enumerate(final_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            context_parts.append(f"[Source {i}: {source} (page {page})]\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        
        # Step 4: Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful RAG assistant. Follow these rules strictly:
1. Answer ONLY using information from the provided context
2. If the answer is not available, say EXACTLY: "I could not find this information in the provided documents."
3. Do NOT make up any information or hallucinate
4. Use bullet points when listing multiple items
5. Keep your answer clear, concise, and natural-sounding"""),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        # Step 5: Wait for Ollama and pull model if needed
        if not wait_for_ollama():
            return ("Ollama service unavailable. Please try again later.", final_docs)
        if not pull_ollama_model():
            return ("Failed to load the LLM model. Please try again later.", final_docs)
        
        # Step 6: Call Ollama
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            num_predict=1024
        )
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
        
        # Add similarity scores to document metadata for UI
        for doc, score in zip(final_docs, final_scores):
            doc.metadata["similarity_score"] = score
        
        print("✅ RAG pipeline complete!")
        return (answer, final_docs)
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline: {e}")
        traceback.print_exc()
        return (f"An error occurred: {str(e)}", [])
