import os
import pdfplumber
import pandas as pd
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Đường dẫn thư mục
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"

# Hàm làm sạch văn bản nâng cao, giữ xuống dòng
def clean_text_advanced(text):
    # Loại bỏ ký tự thừa, giữ xuống dòng
    text = re.sub(r'[^\w\s.,!?\n]', '', text)  # Giữ \n
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'CDC \d{4}', '', text)
    # Thay nhiều khoảng trắng bằng 1, nhưng giữ xuống dòng
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# Hàm xử lý bảng thành danh sách
def process_table(table):
    processed = []
    for row in table:
        row = [str(cell) for cell in row if cell]
        if row:
            processed.append(" | ".join(row))
    return "\n".join(processed)

# Hàm trích xuất và xử lý PDF
def extract_and_process_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            tables = page.extract_tables()
            for table in tables:
                table_text = process_table(table)
                text += table_text + "\n"
    return clean_text_advanced(text)

# Hàm phân đoạn văn bản
def chunk_text(text, chunk_size=200, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# Trích xuất, xử lý, và phân đoạn
pdf_files = [
    "Oral-Health-Surveillance-Report-2019-h.pdf",
    "rr5217.pdf",
    "who_oral_health.pdf"
]

for pdf_file in pdf_files:
    pdf_path = os.path.join(RAW_DIR, pdf_file)
    output_base = os.path.join(PROCESSED_DIR, pdf_file.replace(".pdf", ""))
    
    # Trích xuất và xử lý
    text = extract_and_process_pdf(pdf_path)
    
    # Lưu văn bản thô
    with open(f"{output_base}_raw.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    # Phân đoạn
    chunks = chunk_text(text)
    
    # Lưu các đoạn
    with open(f"{output_base}_chunks.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")
    
    print(f"Đã xử lý: {pdf_file}")

# Minh chứng số dòng
total_lines = 0
for pdf_file in pdf_files:
    file_path = os.path.join(PROCESSED_DIR, pdf_file.replace(".pdf", "_raw.txt"))
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Đếm dòng không rỗng, giữ xuống dòng
            lines = sum(1 for line in f.read().splitlines() if line.strip())
            total_lines += lines
            print(f"Số dòng {file_path}: {lines}")
    except Exception as e:
        print(f"Lỗi khi đọc {file_path}: {e}")
print(f"Tổng số dòng: {total_lines}")