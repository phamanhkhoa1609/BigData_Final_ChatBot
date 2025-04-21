import streamlit as st
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

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Hỏi Đáp Tài Liệu sử dụng Gemini & LlamaIndex",
    page_icon="📚",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .response-container {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0;
    }
    .source-container {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    code {
        color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề
st.title("Hệ thống Hỏi Đáp Tài Liệu sử dụng Gemini ")
st.caption("Đặt câu hỏi về nội dung các file ")

# Khởi tạo session state
if 'query_engine' not in st.session_state:
    # Load environment variables
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("API key của Gemini chưa được đặt. Hãy tạo file .env và thêm GEMINI_API_KEY='YOUR_API_KEY'")
        st.stop()

    # Cấu hình logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with st.spinner("Đang khởi tạo hệ thống..."):
        # Khởi tạo mô hình
        Settings.llm = Gemini(api_key=GEMINI_API_KEY, model_name="models/gemini-2.0-flash")
        Settings.embed_model = GeminiEmbedding(
            api_key=GEMINI_API_KEY, model_name="models/embedding-001"
        )

        # Load và xử lý tài liệu
        DATA_DIR = "data"
        try:
            reader = SimpleDirectoryReader(input_dir=DATA_DIR, recursive=True)
            documents = reader.load_data()
            
            # Phân tách tài liệu
            node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            nodes = node_parser.get_nodes_from_documents(documents)

            # Tạo hoặc tải index
            PERSIST_DIR_VECTOR = "./storage_vector"
            PERSIST_DIR_SUMMARY = "./storage_summary"

            # Vector Index
            if not os.path.exists(PERSIST_DIR_VECTOR):
                vector_index = VectorStoreIndex(nodes, show_progress=True)
                vector_index.storage_context.persist(persist_dir=PERSIST_DIR_VECTOR)
            else:
                storage_context_vector = StorageContext.from_defaults(persist_dir=PERSIST_DIR_VECTOR)
                vector_index = load_index_from_storage(storage_context_vector)

            # Summary Index
            if not os.path.exists(PERSIST_DIR_SUMMARY):
                summary_index = SummaryIndex(nodes, show_progress=True)
                summary_index.storage_context.persist(persist_dir=PERSIST_DIR_SUMMARY)
            else:
                storage_context_summary = StorageContext.from_defaults(persist_dir=PERSIST_DIR_SUMMARY)
                summary_index = load_index_from_storage(storage_context_summary)

            # Tạo query engines
            vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize", use_async=True
            )

            # Tạo tools
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

            # Tạo router query engine
            st.session_state.query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
                query_engine_tools=[vector_tool, summary_tool],
                verbose=True,
            )

        except Exception as e:
            st.error(f"Lỗi khi khởi tạo hệ thống: {str(e)}")
            st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Bạn muốn hỏi gì về các tài liệu?"}]

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Xử lý input
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Thêm câu hỏi vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Hiển thị câu trả lời
    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang tìm câu trả lời..."):
                response = st.session_state.query_engine.query(prompt)
                
                # Hiển thị câu trả lời trong container với style
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown("**Question:** " + prompt)
                st.markdown("**Answer:**")
                st.code(response.response, language="")
                
                # Hiển thị metadata và source nodes
                if hasattr(response, "metadata"):
                    st.markdown("**Attributes:**")
                    st.code(f"response: The response text.\nmetadata: {response.metadata}", language="python")
                
                if hasattr(response, "source_nodes") and response.source_nodes:
                    st.markdown("**Source Nodes:**")
                    for node in response.source_nodes:
                        with st.container():
                            st.markdown('<div class="source-container">', unsafe_allow_html=True)
                            st.code(
                                f"file_name: {node.metadata.get('file_name', 'N/A')}\n"
                                f"file_path: {node.metadata.get('file_path', 'N/A')}\n"
                                f"score: {node.score:.4f}",
                                language="python"
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Thêm câu trả lời vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": response.response})
        except Exception as e:
            error_message = f"Lỗi khi xử lý câu hỏi: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message}) 