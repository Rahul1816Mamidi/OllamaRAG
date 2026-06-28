import os
import traceback
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

PDF_FOLDER = "/app/pdfs"
CHROMA_PATH = "/app/chroma_db"
COLLECTION_NAME = "production_rag_docs"

def extract_text_with_ocr(pdf_path: str) -> str:
    """Extract text from scanned/image PDFs using OCR"""
    try:
        print(f"  🧪 Attempting OCR for scanned PDF...")
        pages = convert_from_path(pdf_path)
        ocr_text = ""
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            ocr_text += f"--- Page {i+1} ---\n{text}\n"
        return ocr_text.strip()
    except Exception as e:
        print(f"  ❌ OCR failed: {e}")
        return ""

def load_single_pdf(pdf_path: str) -> List[Document]:
    """Load a single PDF, try PyPDF first, then OCR if needed"""
    filename = os.path.basename(pdf_path)
    print(f"📄 Loading: {filename}")
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        total_chars = sum(len(doc.page_content) for doc in docs)
        
        if total_chars < 50:  # Threshold for "empty" PDF
            print(f"  ⚠️ Very little text extracted ({total_chars} chars), trying OCR...")
            ocr_text = extract_text_with_ocr(pdf_path)
            if ocr_text and len(ocr_text) > 50:
                print(f"  ✅ OCR successful! Extracted {len(ocr_text)} chars")
                docs = [Document(page_content=ocr_text, metadata={"source": filename, "page": 1})]
            else:
                print(f"  ❌ No usable text found, skipping")
                return []
        else:
            # Enhance metadata with source filename and page numbers
            for doc in docs:
                doc.metadata["source"] = filename
                doc.metadata["page"] = doc.metadata.get("page", 1)
            print(f"  ✅ Loaded {len(docs)} pages, {total_chars} chars total")
        
        return docs
    except Exception as e:
        print(f"  ❌ Failed to load PDF {filename}: {e}")
        traceback.print_exc()
        return []

def run_indexing() -> bool:
    print("\n" + "="*70)
    print("📚 PRODUCTION DOCUMENT INDEXING STARTED")
    print("="*70 + "\n")
    
    # Verify prerequisites
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ ERROR: PDF folder not found: {PDF_FOLDER}")
        return False
    
    # Find PDFs
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDF files found in the folder!")
        return False
    print(f"📂 Found {len(pdf_files)} PDF files to process\n")
    
    # Load all valid documents
    all_documents = []
    successful_files = []
    failed_files = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        docs = load_single_pdf(pdf_path)
        if docs:
            all_documents.extend(docs)
            successful_files.append(pdf_file)
        else:
            failed_files.append(pdf_file)
    
    print("\n" + "-"*70)
    if successful_files:
        print(f"✅ Successfully loaded {len(successful_files)} files: {', '.join(successful_files)}")
    if failed_files:
        print(f"❌ Failed to load {len(failed_files)} files: {', '.join(failed_files)}")
    print("-"*70 + "\n")
    
    if not all_documents:
        print("❌ No valid documents to index!")
        return False
    
    # Chunking
    print("✂️ Chunking documents (chunk_size=800, chunk_overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    
    # Filter out empty chunks
    valid_chunks = []
    for chunk in chunks:
        if chunk.page_content and len(chunk.page_content.strip()) > 20:
            valid_chunks.append(chunk)
    
    print(f"✅ Chunking complete! Created {len(valid_chunks)} valid chunks (filtered out {len(chunks)-len(valid_chunks)} empty chunks)\n")
    
    if not valid_chunks:
        print("❌ No valid chunks to index!")
        return False
    
    # Embeddings with bge-base-en-v1.5
    print("🧠 Initializing embedding model (BAAI/bge-base-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    print("💾 Setting up ChromaDB vector store...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    # Clear existing collection to avoid duplicates
    print("🗑️ Clearing existing index...")
    vector_store.delete_collection()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print("📥 Adding chunks to ChromaDB...")
    vector_store.add_documents(valid_chunks)
    
    # Verify
    collection = vector_store._collection
    count = collection.count()
    
    print("\n" + "="*70)
    print(f"🎉 INDEXING COMPLETE!")
    print(f"   Collection name: {COLLECTION_NAME}")
    print(f"   Total chunks stored: {count}")
    print(f"   Persistence directory: {CHROMA_PATH}")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    success = run_indexing()
    exit(0 if success else 1)
