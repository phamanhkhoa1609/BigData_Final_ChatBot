# Import thư viện os để xử lý đường dẫn file và thư mục trong hệ thống
import os
# Import List từ typing để định nghĩa kiểu dữ liệu cho danh sách (dùng trong type hint)
from typing import List
# Import PyPDFLoader để đọc nội dung file PDF và chuyển thành các trang
from langchain_community.document_loaders import PyPDFLoader
# Import RecursiveCharacterTextSplitter để chia văn bản thành các đoạn nhỏ (chunks)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import Chroma để tạo và quản lý vector database (ChromaDB)
from langchain_community.vectorstores import Chroma
# Import Document để tạo đối tượng văn bản với nội dung và metadata
from langchain.schema import Document
# Import Embeddings để định nghĩa lớp embedding tùy chỉnh (tương thích với LangChain)
from langchain_core.embeddings import Embeddings
# Import load_dotenv để đọc biến môi trường từ file .env
from dotenv import load_dotenv
# Import google.generativeai để sử dụng Gemini API cho embedding và trả lời
import google.generativeai as genai

# Load biến môi trường từ file .env (nếu có)
load_dotenv()
# Lấy giá trị GEMINI_API_KEY từ biến môi trường
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Kiểm tra xem API key có tồn tại không, nếu không thì báo lỗi và dừng chương trình
if not GEMINI_API_KEY:
    raise ValueError("Vui lòng cài đặt GEMINI_API_KEY trong file .env")

# Cấu hình Gemini API với API key để sử dụng các dịch vụ như embedding
genai.configure(api_key=GEMINI_API_KEY)

# Định nghĩa lớp GeminiEmbedding kế thừa từ Embeddings (tương thích với LangChain)
class GeminiEmbedding(Embeddings):
    """Custom embedding class for Gemini API."""
    # Phương thức embed_documents: Tạo embeddings cho danh sách văn bản (dùng cho tài liệu)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Khởi tạo danh sách rỗng để lưu embeddings
        embeddings = []
        # Duyệt qua từng văn bản trong danh sách
        for text in texts:
            # Gọi Gemini API để tạo embedding cho văn bản
            response = genai.embed_content(
                model="models/embedding-001",  # Mô hình embedding của Gemini
                content=text,  # Văn bản cần tạo embedding
                task_type="RETRIEVAL_DOCUMENT"  # Loại tác vụ: embedding cho tài liệu
            )
            # Lấy embedding từ phản hồi và thêm vào danh sách
            embeddings.append(response["embedding"])
        # Trả về danh sách các embeddings (mỗi embedding là một danh sách số thực)
        return embeddings
    
    # Phương thức embed_query: Tạo embedding cho một truy vấn (dùng khi tìm kiếm)
    def embed_query(self, text: str) -> List[float]:
        # Gọi Gemini API để tạo embedding cho truy vấn
        response = genai.embed_content(
            model="models/embedding-001",  # Mô hình embedding của Gemini
            content=text,  # Truy vấn cần tạo embedding
            task_type="RETRIEVAL_QUERY"  # Loại tác vụ: embedding cho truy vấn
        )
        # Trả về embedding của truy vấn (một danh sách số thực)
        return response["embedding"]

# Hàm load_pdfs: Tìm và trả về danh sách đường dẫn các file PDF trong thư mục
def load_pdfs(directory: str) -> List[str]:
    """Load all PDF files from the specified directory."""
    # Khởi tạo danh sách rỗng để lưu đường dẫn file PDF
    pdf_files = []
    # Duyệt qua tất cả file trong thư mục được chỉ định
    for file in os.listdir(directory):
        # Kiểm tra nếu file có đuôi .pdf
        if file.endswith('.pdf'):
            # Thêm đường dẫn đầy đủ của file PDF vào danh sách
            pdf_files.append(os.path.join(directory, file))
    # Trả về danh sách các đường dẫn file PDF
    return pdf_files

# Hàm process_documents: Đọc file PDF, chia thành chunks và thêm metadata
def process_documents(pdf_files: List[str]) -> List[Document]:
    """Process PDF files and split them into chunks with metadata."""
    # Khởi tạo text splitter để chia văn bản thành các đoạn nhỏ
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Kích thước tối đa của mỗi chunk (1000 ký tự)
        chunk_overlap=200,  # Độ chồng lấn giữa các chunk (200 ký tự) để giữ ngữ cảnh
        length_function=len,  # Hàm tính độ dài của văn bản (dùng len mặc định)
    )
    
    # Khởi tạo danh sách rỗng để lưu tất cả các chunk (Document)
    all_docs = []
    # Duyệt qua từng file PDF trong danh sách
    for pdf_file in pdf_files:
        # Lấy tên file từ đường dẫn (ví dụ: who_oral_health.pdf)
        file_name = os.path.basename(pdf_file)
        # Tạo loader để đọc nội dung file PDF
        loader = PyPDFLoader(pdf_file)
        # Đọc file PDF và chia thành các trang (mỗi trang là một Document)
        pages = loader.load()
        
        # Duyệt qua từng trang để thêm metadata
        for i, page in enumerate(pages):
            # Gán metadata cho trang, bao gồm:
            page.metadata = {
                "source": file_name,  # Tên file (ví dụ: who_oral_health.pdf)
                "page": i + 1,  # Số trang (bắt đầu từ 1)
                "file_path": pdf_file  # Đường dẫn đầy đủ của file
            }
        
        # Chia các trang thành các chunk nhỏ hơn bằng text splitter
        chunks = text_splitter.split_documents(pages)
        # Thêm các chunk vào danh sách tổng
        all_docs.extend(chunks)
    
    # Trả về danh sách tất cả các chunk (Document)
    return all_docs

# Hàm create_vector_db: Tạo ChromaDB từ các Document và lưu vào thư mục
def create_vector_db(documents: List[Document], persist_directory: str = "vector_db"):
    """Create a vector database using ChromaDB with Gemini embeddings."""
    # Khởi tạo đối tượng embedding bằng lớp GeminiEmbedding
    embedding = GeminiEmbedding()
    
    # Tạo ChromaDB từ các Document
    vector_store = Chroma.from_documents(
        documents=documents,  # Danh sách các chunk (Document) cần lưu
        embedding=embedding,  # Đối tượng embedding để chuyển văn bản thành vector
        persist_directory=persist_directory,  # Thư mục lưu ChromaDB (vector_db)
        collection_name="bigdata_docs"  # Tên collection trong ChromaDB
    )
    
    # Trả về đối tượng vector_store (ChromaDB)
    return vector_store

# Hàm main: Điều phối toàn bộ quy trình
def main():
    # In thông báo để báo hiệu bắt đầu
    print("Bắt đầu xử lý dữ liệu...")
    
    # Tìm tất cả file PDF trong thư mục data
    pdf_files = load_pdfs("data")
    # In số lượng file PDF tìm thấy
    print(f"Đã tìm thấy {len(pdf_files)} file PDF")
    
    # Xử lý các file PDF để tạo các chunk (Document)
    documents = process_documents(pdf_files)
    # In số lượng chunk đã xử lý
    print(f"Đã xử lý {len(documents)} chunks văn bản")
    
    # Tạo ChromaDB từ các chunk
    vector_store = create_vector_db(documents)
    # In thông báo thành công
    print("Đã tạo vector database thành công!")
    # In đường dẫn tuyệt đối của thư mục lưu ChromaDB
    print(f"Database được lưu tại: {os.path.abspath('vector_db')}")

# Kiểm tra nếu file được chạy trực tiếp (không phải import)
if __name__ == "__main__":
    # Gọi hàm main để thực thi
    main()