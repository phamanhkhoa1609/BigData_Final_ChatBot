import streamlit as st
import requests
import re
from datetime import datetime
import calendar
import unicodedata

# Thiết lập tiêu đề và mô tả
st.set_page_config(page_title="Chatbot Sức Khỏe Răng Miệng", page_icon="🦷", layout="wide")
st.title("Chatbot Sức Khỏe Răng Miệng 🦷")
st.markdown("""
Hỏi về sức khỏe răng miệng, tôi sẽ trả lời dựa trên tài liệu từ CDC và WHO!  
Tôi cũng có thể trả lời các câu hỏi đơn giản (bằng tiếng Việt hoặc tiếng Anh).  
Ví dụ:  
- Cụ thể: "Nguyên nhân chính của sâu răng là gì?"  
- Tóm tắt: "Tóm tắt các bệnh răng miệng phổ biến"  
- Đơn giản: "Hôm nay là thứ mấy?", "Bạn là ai?"  
""")

# Custom CSS cho giao diện
st.markdown("""
<style>
    .response-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .source-container {
        background-color: #e6e9ef;
        border-radius: 5px;
        padding: 10px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state để lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! Hỏi tôi về sức khỏe răng miệng hoặc các câu hỏi đơn giản nhé!"}
    ]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Nguồn tài liệu"):
                for source in message["sources"]:
                    st.markdown(f"- **File**: {source['file_name']} (Trang: {source.get('page', 'N/A')})")
                    st.markdown(f"  **Nội dung**: {source['text']}")

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
    """Kiểm tra xem câu hỏi có hỏi về ngày tháng không."""
    date_keywords = [
        "hom nay", "ngay bao nhieu", "thu may", "what day", "today", "date",
        "ngay thang", "thang may", "nam nay", "ngay nao", "what is the date",
        "ngay may", "hien tai", "bay gio", "ngay hom nay", "thoi gian"
    ]
    return any(keyword in normalize_text(query) for keyword in date_keywords)

def is_simple_query(query: str) -> bool:
    """Kiểm tra xem câu hỏi có phải là câu hỏi đơn giản không (ngoài ngày tháng)."""
    simple_keywords = [
        "ban la ai", "ban la ai", "ten ban", "ten ban", "ban lam gi", "ban lam gi",
        "who are you", "what can you do", "ten cua ban", "ten cua ban", "ban co the",
        "ban co the", "chao ban", "chao ban", "hello", "hi"
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

# Hàm trả lời câu hỏi về ngày tháng
def answer_date_query(query: str, is_vietnamese: bool) -> str:
    """Trả lời các câu hỏi về ngày tháng."""
    today = datetime.now()
    day = today.day
    month = today.month
    year = today.year
    weekday = calendar.day_name[today.weekday()]
    weekday_vn = {
        "Monday": "Thứ Hai", "Tuesday": "Thứ Ba", "Wednesday": "Thứ Tư",
        "Thursday": "Thứ Năm", "Friday": "Thứ Sáu", "Saturday": "Thứ Bảy",
        "Sunday": "Chủ Nhật"
    }.get(weekday, weekday)

    query_lower = normalize_text(query)
    if is_vietnamese:
        if "thu may" in query_lower or "what day" in query_lower:
            return f"Hôm nay là {weekday_vn}."
        elif any(keyword in query_lower for keyword in ["ngay bao nhieu", "date", "ngay nao", "ngay may", "ngay hom nay"]):
            return f"Hôm nay là ngày {day} tháng {month} năm {year}."
        elif "thang may" in query_lower:
            return f"Hôm nay là tháng {month} năm {year}."
        elif "nam nay" in query_lower:
            return f"Hôm nay là năm {year}."
        else:
            return f"Hôm nay là {weekday_vn}, ngày {day} tháng {month} năm {year}."
    else:
        if "thu may" in query_lower or "what day" in query_lower:
            return f"Today is {weekday}."
        elif any(keyword in query_lower for keyword in ["ngay bao nhieu", "date", "ngay nao", "ngay may", "ngay hom nay"]):
            return f"Today is {day} {calendar.month_name[month]} {year}."
        elif "thang may" in query_lower:
            return f"It's {calendar.month_name[month]} {year}."
        elif "nam nay" in query_lower:
            return f"It's {year}."
        else:
            return f"Today is {weekday}, {day} {calendar.month_name[month]} {year}."

# Hàm trả lời câu hỏi đơn giản (ngoài ngày tháng)
def answer_simple_query(query: str, is_vietnamese: bool) -> str:
    """Trả lời các câu hỏi đơn giản như 'Bạn là ai?'."""
    query_lower = normalize_text(query)
    if is_vietnamese:
        if "ban la ai" in query_lower or "ten ban" in query_lower or "ten cua ban" in query_lower:
            return "Tôi là Chatbot Sức Khỏe Răng Miệng, được tạo để trả lời các câu hỏi về răng miệng và các câu hỏi đơn giản!"
        elif "ban lam gi" in query_lower or "ban co the" in query_lower:
            return "Tôi có thể trả lời các câu hỏi về sức khỏe răng miệng dựa trên tài liệu từ CDC và WHO, hoặc các câu hỏi đơn giản như ngày tháng, giới thiệu bản thân!"
        elif "chao ban" in query_lower or "hello" in query_lower or "hi" in query_lower:
            return "Chào bạn! Rất vui được trò chuyện với bạn. Hỏi tôi về sức khỏe răng miệng nhé!"
        else:
            return "Tôi có thể giúp gì cho bạn? Hỏi về sức khỏe răng miệng hoặc các câu hỏi đơn giản nhé!"
    else:
        if "who are you" in query_lower or "ten ban" in query_lower or "ten cua ban" in query_lower:
            return "I am the Oral Health Chatbot, created to answer questions about dental health and simple queries!"
        elif "what can you do" in query_lower or "ban co the" in query_lower:
            return "I can answer questions about oral health based on CDC and WHO documents, or simple queries like dates and introductions!"
        elif "chao ban" in query_lower or "hello" in query_lower or "hi" in query_lower:
            return "Hello! Nice to chat with you. Ask me about oral health!"
        else:
            return "How can I help you? Ask about oral health or simple questions!"

# Nhập câu hỏi từ người dùng
if prompt := st.chat_input("Hỏi về sức khỏe răng miệng hoặc câu hỏi đơn giản (ví dụ: Nguyên nhân sâu răng? Hôm nay thứ mấy? Bạn là ai?):"):
    # Thêm câu hỏi vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Kiểm tra ngôn ngữ của câu hỏi
    is_vn = is_vietnamese(prompt)

    # Xử lý câu hỏi
    if is_date_query(prompt):
        # Câu hỏi về ngày tháng
        answer = answer_date_query(prompt, is_vn)
        with st.chat_message("assistant"):
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown(f"**Câu trả lời:** {answer}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    elif is_simple_query(prompt):
        # Câu hỏi đơn giản (ngoài ngày tháng)
        answer = answer_simple_query(prompt, is_vn)
        with st.chat_message("assistant"):
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown(f"**Câu trả lời:** {answer}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    elif not is_dental_query(prompt):
        # Câu hỏi không liên quan đến sức khỏe răng miệng
        answer = "Tôi chỉ có thể trả lời các câu hỏi về sức khỏe răng miệng hoặc các câu hỏi đơn giản. Bạn có thể hỏi về chủ đề này không?" if is_vn else "I can only answer questions about oral health or simple queries. Can you ask about these topics?"
        with st.chat_message("assistant"):
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            st.markdown(f"**Câu trả lời:** {answer}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        # Câu hỏi về sức khỏe răng miệng, gọi API
        query = prompt
        if is_summary_query(prompt):
            query = f"Tóm tắt: {prompt}"  # Thêm từ khóa để API trả về câu trả lời tổng quan

        try:
            with st.spinner("Đang xử lý câu hỏi..."):
                response = requests.post(
                    "http://localhost:5000/api/query",
                    headers={"Content-Type": "application/json"},
                    json={"query": query}
                )
                response.raise_for_status()  # Báo lỗi nếu không phải HTTP 200
                result = response.json()

            # Hiển thị câu trả lời
            answer = result["answer"].strip()
            sources = result["sources"]
            with st.chat_message("assistant"):
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown(f"**Câu trả lời:** {answer}")
                if sources:
                    with st.expander("📚 Nguồn tài liệu"):
                        for source in sources:
                            st.markdown('<div class="source-container">', unsafe_allow_html=True)
                            st.markdown(f"- **File**: {source['file_name']} (Trang: {source.get('page', 'N/A')})")
                            st.markdown(f"  **Nội dung**: {source['text']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Lưu câu trả lời vào lịch sử
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi gọi API: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại!" if is_vn else "An error occurred while processing your question. Please try again!"
            })