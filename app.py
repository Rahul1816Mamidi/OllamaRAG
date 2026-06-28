import streamlit as st
from ingest import run_indexing
from rag import ask_question

# Page configuration
st.set_page_config(
    page_title="📚 RAG with Ollama",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📚 Production-Grade RAG with Ollama (qwen2.5:7b)")
st.markdown("---")

# Sidebar for indexing and info
with st.sidebar:
    st.header("⚙️ System Controls")
    st.markdown("---")
    
    # Indexing button
    if st.button("📥 Index Documents", type="primary", use_container_width=True):
        with st.spinner("⏳ Indexing documents (this may take a few minutes)..."):
            try:
                success = run_indexing()
                if success:
                    st.success("✅ Indexing complete!")
                    st.balloons()
                else:
                    st.error("❌ Indexing failed!")
            except Exception as e:
                st.error(f"❌ Error during indexing: {str(e)}")
                st.exception(e)
    
    st.markdown("---")
    st.subheader("ℹ️ System Info")
    st.info("""
    - LLM: Ollama qwen2.5:7b
    - Embedding Model: BAAI/bge-base-en-v1.5
    - Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Chunk Size: 800 tokens
    - Chunk Overlap: 150 tokens
    - Retrieval: MMR (k=5, fetch_k=20)
    """)

# Main chat area
st.subheader("💬 Ask a Question")
user_question = st.text_input(
    "Enter your question about your documents:",
    placeholder="e.g., What are the key AI business themes mentioned?",
    label_visibility="visible"
)

if st.button("🔍 Get Answer", use_container_width=True):
    if not user_question:
        st.warning("⚠️ Please enter a question first!")
    else:
        st.subheader("💭 Your Question")
        st.write(user_question)
        
        with st.spinner("🔍 Processing query (retrieving, reranking, and generating answer)..."):
            try:
                answer, docs = ask_question(user_question)
                
                if docs:
                    st.markdown("---")
                    st.subheader("📑 Retrieved Context")
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", 1)
                        score = doc.metadata.get("similarity_score", None)
                        with st.expander(f"Chunk {i}: {source} (page {page})" + (f" | Similarity: {score:.3f}" if score else "")):
                            st.write(doc.page_content)
                
                st.markdown("---")
                st.subheader("🤖 AI Answer")
                st.markdown(answer)
                            
            except Exception as e:
                st.error(f"❌ Error getting answer: {str(e)}")
                st.exception(e)
