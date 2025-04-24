import os
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

# Đọc các đoạn từ file _chunks.txt
chunk_files = [
    "Oral-Health-Surveillance-Report-2019-h_chunks.txt",
    "rr5217_chunks.txt",
    "who_oral_health_chunks.txt"
]

# Lưu embedding và metadata
embeddings = []
metadata = []

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
                        embeddings.append(embedding)
                        metadata.append({
                            "file": chunk_file,
                            "chunk_id": i+1,
                            "text": chunk_text
                        })
                        print(f"Đã tạo embedding cho chunk {i+1} từ {chunk_file}")
    except Exception as e:
        print(f"Lỗi khi đọc {chunk_file}: {e}")

# Lưu vào file (giả lập kho vector)
output_file = "data/processed/basic_vector_store.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
        f.write(f"Chunk {i+1}: {meta['file']} - {meta['text'][:50]}...\n")
print(f"Đã lưu vector store cơ bản vào {output_file}")