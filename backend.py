# Import các thư viện cần thiết
from flask import Flask, request, jsonify  # Flask: tạo API web; request, jsonify: xử lý yêu cầu và trả về JSON
from flask_cors import CORS  # Cho phép truy cập API từ các nguồn khác (cross-origin)
import os  # Xử lý đường dẫn file và hệ thống
from typing import List  # Cung cấp kiểu List để định nghĩa kiểu dữ liệu
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever  # Tìm kiếm từ khóa kiểu BM25
from langchain.retrievers import EnsembleRetriever  # Kết hợp vector search và BM25
from langchain_core.embeddings import Embeddings  # Lớp cơ sở cho embeddings trong LangChain
from langchain.schema import Document  # Đối tượng Document để lưu văn bản và metadata
import google.generativeai as genai  # Gọi API Gemini để tạo embeddings và trả lời
from dotenv import load_dotenv  # Load biến môi trường từ file .env
import re  # Để trích xuất số từ văn bản
from datetime import datetime  # Để xử lý ngày tháng và thời gian
import calendar  # Để lấy tên ngày trong tuần
import unicodedata  # Để chuẩn hóa chuỗi tiếng Việt

# Load biến môi trường từ file .env
load_dotenv()  # Đọc các biến môi trường từ file .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Lấy API key của Gemini
if not GEMINI_API_KEY:
    raise ValueError("Vui lòng cài đặt GEMINI_API_KEY trong file .env")  # Báo lỗi nếu không có API key

# Cấu hình Gemini với API key
genai.configure(api_key=GEMINI_API_KEY)  # Thiết lập Gemini API với API key

# Định nghĩa lớp GeminiEmbedding để tạo embeddings cho ChromaDB
class GeminiEmbedding(Embeddings):
    """Lớp tùy chỉnh để tạo embeddings từ Gemini API, tương thích với LangChain."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []  # Khởi tạo danh sách để lưu embeddings
        for text in texts:  # Duyệt qua từng văn bản
            response = genai.embed_content(
                model="models/embedding-001",  # Mô hình embedding của Gemini
                content=text,  # Văn bản cần tạo embedding
                task_type="RETRIEVAL_DOCUMENT"  # Loại tác vụ: embedding cho tài liệu
            )
            embeddings.append(response["embedding"])  # Thêm embedding vào danh sách
        return embeddings  # Trả về danh sách các embeddings
    
    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model="models/embedding-001",  # Mô hình embedding của Gemini
            content=text,  # Truy vấn cần tạo embedding
            task_type="RETRIEVAL_QUERY"  # Loại tác vụ: embedding cho truy vấn
        )
        return response["embedding"]  # Trả về embedding của truy vấn

# Khởi tạo Flask app
app = Flask(__name__)  # Tạo ứng dụng Flask
CORS(app)  # Kích hoạt CORS để frontend (JavaScript) gọi API

# Load ChromaDB từ thư mục đã lưu
PERSIST_DIR = "./vector_db"  # Đường dẫn tới thư mục chứa ChromaDB (tạo bởi vector_db.py)
vector_store = Chroma(
    persist_directory=PERSIST_DIR,  # Thư mục chứa dữ liệu ChromaDB
    embedding_function=GeminiEmbedding(),  # Dùng lớp GeminiEmbedding để tạo embeddings
    collection_name="bigdata_docs"  # Tên collection (phải khớp với vector_db.py)
)

# Hàm chuẩn hóa chuỗi để so sánh (loại bỏ dấu tiếng Việt)
def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi bằng cách loại bỏ dấu tiếng Việt và chuyển về chữ thường."""
    text = unicodedata.normalize('NFKD', text.lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.strip()

# Hàm kiểm tra ngôn ngữ (tiếng Việt hay không)
def is_vietnamese(text: str) -> bool:
    """Kiểm tra xem câu hỏi có phải tiếng Việt không dựa trên ký tự Unicode."""
    vietnamese_chars = set('ăâđêôơư')
    return any(char in vietnamese_chars for char in text.lower())

# Hàm kiểm tra loại câu hỏi
def is_summary_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có yêu cầu tóm tắt không."""
    summary_keywords = ["tóm tắt", "tong quan", "summary", "overview"]
    return any(keyword in normalize_text(query) for keyword in summary_keywords)

def is_date_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có hỏi về ngày tháng hoặc thời gian không."""
    date_keywords = [
        "hom nay", "ngay bao nhieu", "thu may", "what day", "today", "date",
        "ngay thang", "thang may", "nam nay", "ngay nao", "what is the date",
        "ngay may", "hien tai", "bay gio", "ngay hom nay", "thoi gian",
        "may gio", "time", "what time", "gio nay"  # Thêm từ khóa cho thời gian
    ]
    return any(keyword in normalize_text(query) for keyword in date_keywords)

def is_simple_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là câu hỏi đơn giản không (ngoài ngày tháng)."""
    simple_keywords = [
        "ban la ai", "ten ban", "ten cua ban", "ban lam gi", "ban co the",
        "who are you", "what can you do",
        "chao ban", "hello", "hi", "xin chao", "chao buoi sang", "hello there",  # Thêm các biến thể chào hỏi
    ]
    normalized_query = normalize_text(query)
    return any(keyword in normalized_query for keyword in simple_keywords)

def is_dental_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có liên quan đến sức khỏe răng miệng không."""
    dental_keywords = [
        "rang", "mieng", "sau rang", "nuou", "tooth", "dental", "oral", "plaque",
        "fluoride", "caries", "gum", "benh rang", "oral health", "viem nuou",
        "teeth", "gingivitis", "periodontal", "decay", "enamel", "dentist",
        "brushing", "flossing", "mouth", "sugar", "prevention"
    ]
    return any(keyword in normalize_text(query) for keyword in dental_keywords)

def is_translation_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có yêu cầu dịch Anh-Việt hoặc Việt-Anh không."""
    translation_keywords = [
        "dich", "translate", "anh sang viet", "viet sang anh", "english to vietnamese",
        "vietnamese to english", "dịch", "dịch sang tiếng anh", "dịch sang tiếng việt"
    ]
    return any(keyword in normalize_text(query) for keyword in translation_keywords)

# Hàm trả lời câu hỏi về ngày tháng và thời gian
def answer_date_query(query: str, is_vietnamese: bool) -> str:
    """Trả lời các câu hỏi về ngày tháng và thời gian."""
    today = datetime.now()
    day = today.day
    month = today.month
    year = today.year
    weekday = calendar.day_name[today.weekday()]
    hour = today.hour
    minute = today.minute
    weekday_vn = {
        "Monday": "Thứ Hai", "Tuesday": "Thứ Ba", "Wednesday": "Thứ Tư",
        "Thursday": "Thứ Năm", "Friday": "Thứ Sáu", "Saturday": "Thứ Bảy",
        "Sunday": "Chủ Nhật"
    }.get(weekday, weekday)

    query_lower = normalize_text(query)
    if is_vietnamese:
        if "thu may" in query_lower or "what day" in query_lower:
            return f"Hôm nay là {weekday_vn}."
        elif "may gio" in query_lower or "gio nay" in query_lower or "time" in query_lower:
            return f"Bây giờ là {hour} giờ {minute} phút."
        elif any(keyword in query_lower for keyword in ["ngay bao nhieu", "date", "ngay nao", "ngay may", "ngay hom nay"]):
            return f"Hôm nay là ngày {day} tháng {month} năm {year}."
        elif "thang may" in query_lower:
            return f"Hôm nay là tháng {month} năm {year}."
        elif "nam nay" in query_lower:
            return f"Hôm nay là năm {year}."
        else:
            return f"Hôm nay là {weekday_vn}, ngày {day} tháng {month} năm {year}, lúc {hour} giờ {minute} phút."
    else:
        if "thu may" in query_lower or "what day" in query_lower:
            return f"Today is {weekday}."
        elif "may gio" in query_lower or "gio nay" in query_lower or "time" in query_lower:
            return f"It's {hour}:{minute} now."
        elif any(keyword in query_lower for keyword in ["ngay bao nhieu", "date", "ngay nao", "ngay may", "ngay hom nay"]):
            return f"Today is {day} {calendar.month_name[month]} {year}."
        elif "thang may" in query_lower:
            return f"It's {calendar.month_name[month]} {year}."
        elif "nam nay" in query_lower:
            return f"It's {year}."
        else:
            return f"Today is {weekday}, {day} {calendar.month_name[month]} {year}, at {hour}:{minute}."

# Hàm trả lời câu hỏi đơn giản (ngoài ngày tháng)
def answer_simple_query(query: str, is_vietnamese: bool) -> str:
    """Trả lời các câu hỏi đơn giản như 'Bạn là ai?' hoặc các câu chào hỏi."""
    query_lower = normalize_text(query)
    if is_vietnamese:
        if "ban la ai" in query_lower or "ten ban" in query_lower or "ten cua ban" in query_lower:
            return "Tôi là Chatbot Sức Khỏe Răng Miệng, được tạo để trả lời các câu hỏi về răng miệng và các câu hỏi đơn giản!"
        elif "ban lam gi" in query_lower or "ban co the" in query_lower:
            return "Tôi có thể trả lời các câu hỏi về sức khỏe răng miệng dựa trên tài liệu từ CDC và WHO, hoặc các câu hỏi đơn giản như ngày tháng, thời gian, hoặc chào hỏi!"
        elif any(keyword in query_lower for keyword in ["chao ban", "xin chao", "chao buoi sang"]):
            return "Chào bạn! Rất vui được trò chuyện với bạn. Hỏi tôi về sức khỏe răng miệng hoặc các câu hỏi đơn giản nhé!"
        else:
            return "Tôi có thể giúp gì cho bạn? Hỏi về sức khỏe răng miệng hoặc các câu hỏi đơn giản nhé!"
    else:
        if "who are you" in query_lower or "ten ban" in query_lower or "ten cua ban" in query_lower:
            return "I am the Oral Health Chatbot, created to answer questions about dental health and simple queries!"
        elif "what can you do" in query_lower or "ban co the" in query_lower:
            return "I can answer questions about oral health based on CDC and WHO documents, or simple queries like dates, times, or greetings!"
        elif any(keyword in query_lower for keyword in ["hello", "hi", "hello there"]):
            return "Hello! Nice to chat with you. Ask me about oral health or simple questions!"
        else:
            return "How can I help you? Ask about oral health or simple questions!"

# Hàm xử lý yêu cầu dịch
def handle_translation(query: str, is_vietnamese: bool) -> str:
    """Dịch Anh-Việt hoặc Việt-Anh bằng Gemini API."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    query_lower = normalize_text(query)

    # Xác định hướng dịch
    if "anh sang viet" in query_lower or "english to vietnamese" in query_lower:
        text_to_translate = query.replace("dịch anh sang việt", "").replace("english to vietnamese", "").strip()
        prompt = f"Translate the following English text to Vietnamese: {text_to_translate}"
    elif "viet sang anh" in query_lower or "vietnamese to english" in query_lower:
        text_to_translate = query.replace("dịch việt sang anh", "").replace("vietnamese to english", "").strip()
        prompt = f"Translate the following Vietnamese text to English: {text_to_translate}"
    else:
        if is_vietnamese:
            prompt = f"Translate the following Vietnamese text to English: {query}"
        else:
            prompt = f"Translate the following English text to Vietnamese: {query}"

    response = model.generate_content(prompt)
    return response.text.strip()

# Hàm thực hiện hybrid search (kết hợp vector search và BM25)
def hybrid_search(query: str, k: int = 3):
    """Tìm kiếm tài liệu liên quan bằng kết hợp vector search và BM25, sau đó rerank."""
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    chroma_data = vector_store.get()
    documents = [
        Document(
            page_content=text,
            metadata={
                "source": meta.get("source", "N/A"),
                "file_path": meta.get("file_path", "N/A"),
                "page": meta.get("page", "N/A")
            }
        )
        for text, meta in zip(chroma_data['documents'], chroma_data['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    results = ensemble_retriever.invoke(query)
    model = genai.GenerativeModel("gemini-1.5-flash")
    reranked = []
    for doc in results:
        response = model.generate_content(
            f"Rate the relevance of the following document to the query '{query}' on a scale of 0 to 1. Return only the numerical score (e.g., 0.8): {doc.page_content[:200]}"
        )
        text = response.text.strip()
        match = re.search(r'^\d*\.?\d+$', text)
        score = float(match.group()) if match else 0.0
        reranked.append((doc, score))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:k]]

# Route xử lý truy vấn từ client
@app.route('/api/query', methods=['POST'])
def handle_query():
    """Xử lý truy vấn từ client, trả về câu trả lời và nguồn tài liệu."""
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Kiểm tra ngôn ngữ của câu hỏi
        is_vn = is_vietnamese(query)

        # Xử lý câu hỏi
        if is_date_query(query):
            # Câu hỏi về ngày tháng hoặc thời gian
            answer = answer_date_query(query, is_vn)
            result = {
                "answer": answer,
                "sources": []  # Không có nguồn cho câu hỏi đơn giản
            }
            return jsonify(result), 200

        elif is_simple_query(query):
            # Câu hỏi đơn giản (ngoài ngày tháng)
            answer = answer_simple_query(query, is_vn)
            result = {
                "answer": answer,
                "sources": []  # Không có nguồn cho câu hỏi đơn giản
            }
            return jsonify(result), 200

        elif is_translation_query(query):
            # Yêu cầu dịch Anh-Việt hoặc Việt-Anh
            answer = handle_translation(query, is_vn)
            result = {
                "answer": answer,
                "sources": []  # Không có nguồn cho câu hỏi dịch
            }
            return jsonify(result), 200

        elif not is_dental_query(query):
            # Câu hỏi không liên quan đến sức khỏe răng miệng
            answer = "Tôi chỉ có thể trả lời các câu hỏi về sức khỏe răng miệng hoặc các câu hỏi đơn giản. Bạn có thể hỏi về chủ đề này không?" if is_vn else "I can only answer questions about oral health or simple queries. Can you ask about these topics?"
            result = {
                "answer": answer,
                "sources": []  # Không có nguồn cho câu hỏi không liên quan
            }
            return jsonify(result), 200

        else:
            # Câu hỏi về sức khỏe răng miệng, thực hiện hybrid search
            query_text = query
            if is_summary_query(query):
                query_text = f"Tóm tắt: {query}"  # Thêm từ khóa để Gemini trả về câu trả lời tổng quan

            docs = hybrid_search(query_text)
            context = "\n".join([doc.page_content for doc in docs])
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Answer based on the following context:\n{context}\nQuery: {query_text}\nProvide a concise answer."
            )
            result = {
                "answer": response.text,
                "sources": [
                    {
                        "file_name": doc.metadata.get("source", "N/A"),
                        "file_path": doc.metadata.get("file_path", "N/A"),
                        "page": doc.metadata.get("page", "N/A"),
                        "text": doc.page_content[:200]
                    }
                    for doc in docs
                ]
            }
            return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route kiểm tra trạng thái API
@app.route('/api/health', methods=['GET'])
def health_check():
    """Kiểm tra xem API có hoạt động không."""
    return jsonify({"status": "healthy"}), 200

# Chạy Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)