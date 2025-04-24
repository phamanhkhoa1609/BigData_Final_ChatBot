import os
import pdfplumber
import pandas as pd
import re

# Đường dẫn thư mục
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

# Tạo thư mục processed nếu chưa có
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Hàm làm sạch văn bản
def clean_text(text):
    # Loại bỏ ký tự thừa, khoảng trắng liên tiếp
    text = re.sub(r'\s+', ' ', text.strip())
    # Loại bỏ tiêu đề/chân trang phổ biến (ví dụ: "Page 1", "CDC 2019")
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'CDC \d{4}', '', text)
    return text

# Hàm trích xuất văn bản từ PDF
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Trích xuất văn bản
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
            # Trích xuất bảng (nếu có)
            tables = page.extract_tables()
            for table in tables:
                # Chuyển bảng thành chuỗi văn bản
                for row in table:
                    row_text = " | ".join([str(cell) for cell in row if cell])
                    text += row_text + " "
    return clean_text(text)

# Trích xuất từ tất cả file PDF
pdf_files = [
    "Oral-Health-Surveillance-Report-2019-h.pdf",
    "rr5217.pdf",
    "who_oral_health.pdf"
]

for pdf_file in pdf_files:
    pdf_path = os.path.join(RAW_DIR, pdf_file)
    output_file = os.path.join(PROCESSED_DIR, pdf_file.replace(".pdf", ".txt"))
    
    # Trích xuất và lưu văn bản
    text = extract_pdf_text(pdf_path)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Đã trích xuất: {output_file}")