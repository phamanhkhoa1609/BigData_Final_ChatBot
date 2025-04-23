from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import sys
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("API key của Gemini chưa được đặt. Hãy tạo file .env và thêm GEMINI_API_KEY='YOUR_API_KEY'")

# Cấu hình logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Khởi tạo mô hình
Settings.llm = Gemini(api_key=GEMINI_API_KEY, model_name="models/gemini-2.0-flash")
Settings.embed_model = GeminiEmbedding(
    api_key=GEMINI_API_KEY, model_name="models/embedding-001"
)

try:
    DATA_DIR = "data"
    reader = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True)
    documents = reader.load_data()

    print(f"\nĐã tìm thấy {len(documents)} tài liệu:")
    for doc in documents:
        print(f"- {doc.metadata.get('file_name', 'Unknown')}")

    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = node_parser.get_nodes_from_documents(documents)

    PERSIST_DIR_VECTOR = "./storage_vector"
    PERSIST_DIR_SUMMARY = "./storage_summary"

    if not os.path.exists(PERSIST_DIR_VECTOR):
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.storage_context.persist(persist_dir=PERSIST_DIR_VECTOR)
    else:
        storage_context_vector = StorageContext.from_defaults(persist_dir=PERSIST_DIR_VECTOR)
        vector_index = load_index_from_storage(storage_context_vector)

    if not os.path.exists(PERSIST_DIR_SUMMARY):
        summary_index = SummaryIndex(nodes, show_progress=True)
        summary_index.storage_context.persist(persist_dir=PERSIST_DIR_SUMMARY)
    else:
        storage_context_summary = StorageContext.from_defaults(persist_dir=PERSIST_DIR_SUMMARY)
        summary_index = load_index_from_storage(storage_context_summary)

    vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)
    
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize", use_async=True
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name="vector_search_tool",
        description="Hữu ích để tìm kiếm thông tin cụ thể từ tài liệu",
    )
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name="summary_tool",
        description="Hữu ích để tóm tắt nội dung của tài liệu",
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
        query_engine_tools=[vector_tool, summary_tool],
        verbose=True,
    )

except Exception as e:
    print(f"\nLỗi khi khởi tạo hệ thống: {str(e)}")
    query_engine = None

def serialize_source_node(node):
    score = getattr(node, "score", None)
    try:
        score = round(float(score), 4) if score is not None else "N/A"
    except Exception:
        score = "N/A"

    return {
        "file_name": node.metadata.get("file_name", "N/A"),
        "file_path": node.metadata.get("file_path", "N/A"),
        "score": score,
        "text": node.text if hasattr(node, "text") else ""
    }

@app.route('/api/query', methods=['POST'])
def handle_query():
    if query_engine is None:
        return jsonify({"error": "Hệ thống chưa được khởi tạo. Vui lòng đặt tài liệu vào thư mục data."}), 503

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = query_engine.query(query)

        result = {
            "answer": str(response.response),
            "sources": []
        }

        if hasattr(response, "source_nodes") and response.source_nodes:
            seen = set()
            unique_sources = []
            for node in response.source_nodes:
                score = getattr(node, "score", None)
                try:
                    score = float(score)
                    if score < 0.3:  # Bỏ những kết quả không liên quan
                        continue
                except (ValueError, TypeError):
                    continue

                key = (node.metadata.get("file_name"), node.metadata.get("file_path"))
                if key not in seen:
                    seen.add(key)
                    node.score = round(score, 4)  # Gán lại score đã xử lý
                    unique_sources.append(serialize_source_node(node))
            result["sources"] = unique_sources

        return jsonify(result), 200

    except Exception as e:
        print(f"Error in handle_query: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
