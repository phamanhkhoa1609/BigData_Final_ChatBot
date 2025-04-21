import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Vui lòng cài đặt GEMINI_API_KEY trong file .env")

genai.configure(api_key=GEMINI_API_KEY)

def load_pdfs(directory: str) -> List[str]:
    """Load all PDF files from the specified directory."""
    pdf_files = []
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            pdf_files.append(os.path.join(directory, file))
    return pdf_files

def process_documents(pdf_files: List[str]) -> List[Document]:
    """Process PDF files and split them into chunks with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    all_docs = []
    for pdf_file in pdf_files:
        # Lấy tên file PDF
        file_name = os.path.basename(pdf_file)
        
        # Load và xử lý PDF
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        
        # Thêm metadata cho mỗi trang
        for i, page in enumerate(pages):
            page.metadata = {
                "source": file_name,
                "page": i + 1,
                "file_path": pdf_file
            }
        
        # Chia nhỏ văn bản thành các chunk
        chunks = text_splitter.split_documents(pages)
        all_docs.extend(chunks)
    
    return all_docs

def create_vector_db(documents: List[Document], persist_directory: str = "vector_db"):
    """Create and persist a vector database using ChromaDB with Gemini embeddings."""
    # Khởi tạo embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Tạo vector store với embeddings
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="bigdata_docs"
    )
    
    # Lưu database
    vector_store.persist()
    return vector_store

def main():
    print("Bắt đầu xử lý dữ liệu...")
    
    # Load và xử lý PDFs
    pdf_files = load_pdfs("data")
    print(f"Đã tìm thấy {len(pdf_files)} file PDF")
    
    # Xử lý documents
    documents = process_documents(pdf_files)
    print(f"Đã xử lý {len(documents)} chunks văn bản")
    
    # Tạo vector database
    vector_store = create_vector_db(documents)
    print("Đã tạo vector database thành công!")
    print(f"Database được lưu tại: {os.path.abspath('vector_db')}")

if __name__ == "__main__":
    main() 