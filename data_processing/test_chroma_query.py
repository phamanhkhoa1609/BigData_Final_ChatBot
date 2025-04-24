import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Tải API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Hàm tạo embedding
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

# Kết nối ChromaDB
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection(name="dental_health")

# Truy vấn thử
query = "What is the prevalence of dental caries in children?"
query_embedding = get_embedding(query)
if query_embedding:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    print("Kết quả truy vấn:")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"Kết quả {i+1}:")
        print(f"File: {meta['file']}, Chunk ID: {meta['chunk_id']}")
        print(f"Nội dung: {doc[:100]}...")
        print()
else:
    print("Lỗi khi tạo embedding truy vấn")