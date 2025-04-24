import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Tải API key từ .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Đường dẫn thư mục
PROCESSED_DIR = "data/processed/"

# Hàm tạo embedding bằng Gemini API
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result["embedding"]
    except Exception as e:
        print(f"Lỗi khi tạo embedding: {e}")
        return None

# Khởi tạo ChromaDB
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.create_collection(name="dental_health", get_or_create=True)

# Đọc các đoạn từ file _chunks.txt
chunk_files = [
    "Oral-Health-Surveillance-Report-2019-h_chunks.txt",
    "rr5217_chunks.txt",
    "who_oral_health_chunks.txt"
]

# Lưu vào ChromaDB
for chunk_file in chunk_files:
    file_path = os.path.join(PROCESSED_DIR, chunk_file)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = content.split("Chunk ")[1:]  # Tách từng đoạn
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.split("\n\n")[0].strip()
                if chunk_text:
                    embedding = get_embedding(chunk_text)
                    if embedding:
                        collection.add(
                            embeddings=[embedding],
                            documents=[chunk_text],
                            metadatas=[{
                                "file": chunk_file,
                                "chunk_id": i+1
                            }],
                            ids=[f"{chunk_file}_{i+1}"]
                        )
                        print(f"Đã thêm chunk {i+1} từ {chunk_file} vào ChromaDB")
    except Exception as e:
        print(f"Lỗi khi đọc {chunk_file}: {e}")

print("Đã tạo kho vector ChromaDB tại data/chroma_db")

# Kiểm tra kho vector
count = collection.count()
print(f"Tổng số đoạn trong ChromaDB: {count}")