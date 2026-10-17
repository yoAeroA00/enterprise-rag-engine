import logging
import traceback
from datetime import datetime

import streamlit as st

from config import Config
from document_processor import DocumentProcessor
from rag_chain import RAGChain

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logging
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_rag_chain():
    """Initialize RAG chain with caching"""
    try:
        return RAGChain()
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}")
        st.stop()


@st.cache_resource
def initialize_document_processor():
    """Initialize document processor with caching"""
    return DocumentProcessor()


def validate_configuration():
    """Validate application configuration"""
    config_validation = Config.validate_config()

    if not config_validation["valid"]:
        st.error("❌ Configuration Error")
        st.write("The following required environment variables are missing:")
        for missing in config_validation["missing"]:
            st.write(f"- {missing}")
        st.write(
            "Please check your .env file and ensure all required variables are set."
        )
        st.stop()

    if config_validation["warnings"]:
        with st.sidebar:
            st.warning("⚠️ Configuration Warnings")
            for warning in config_validation["warnings"]:
                st.write(f"- {warning}")


def display_health_status():
    """Display system health status in sidebar"""
    with st.sidebar, st.expander("🏥 System Health", expanded=False):
        try:
            health_status = st.session_state.rag_chain.health_check()

            if health_status["rag_chain_healthy"]:
                st.success("✅ RAG Chain: Healthy")
            else:
                st.error("❌ RAG Chain: Issues detected")

            if health_status["llm_working"]:
                st.success("✅ Azure OpenAI: Connected")
            else:
                st.error("❌ Azure OpenAI: Connection failed")

            vs_status = health_status.get("vector_store_status", {})
            if vs_status.get("qdrant_connection"):
                st.success("✅ Qdrant: Connected")
            else:
                st.error("❌ Qdrant: Connection failed")

            if vs_status.get("embeddings_working"):
                st.success("✅ Embeddings: Working")
            else:
                st.error("❌ Embeddings: Failed")

            if health_status["errors"]:
                st.error("Errors:")
                for error in health_status["errors"][:3]:  # Show max 3 errors
                    st.write(f"- {error}")

        except Exception as e:
            st.error(f"Health check failed: {e}")


def display_vector_store_info():
    """Display vector store information in sidebar"""
    with st.sidebar, st.expander("📊 Vector Store Info", expanded=False):
        try:
            info = st.session_state.rag_chain.get_vector_store_info()

            if "error" in info:
                st.error(f"Error: {info['error']}")
            else:
                st.metric("Documents", info.get("points_count", 0))
                st.metric("Vectors", info.get("vectors_count", 0))
                st.write(f"**Status:** {info.get('status', 'Unknown')}")
                st.write(f"**Collection:** {info.get('name', 'Unknown')}")

        except Exception as e:
            st.error(f"Failed to get vector store info: {e}")


def chat_interface():
    """Main chat interface"""
    st.title("🤖 RAG Document Assistant")
    st.write("Ask questions about your uploaded documents!")

    # Chat history container
    chat_container = st.container()

    # Display conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]

                    # Show response time
                    if "response_time" in metadata:
                        st.caption(f"⏱️ Response time: {metadata['response_time']:.2f}s")

                    # Show source documents
                    if metadata.get("source_documents"):
                        with st.expander(
                            f"📚 Sources ({len(metadata['source_documents'])} documents)",
                            expanded=False,
                        ):
                            for i, doc in enumerate(metadata["source_documents"]):
                                st.write(f"**Document {i + 1}:** {doc['filename']}")
                                st.write(
                                    f"**Similarity:** {doc['similarity_score']:.3f}"
                                )
                                if doc.get("chunk_id") is not None:
                                    st.write(f"**Chunk:** {doc['chunk_id'] + 1}")
                                st.write(f"**Preview:** {doc['content_preview']}")
                                st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                response_data = st.session_state.rag_chain.generate_response(prompt)

                if "error" in response_data:
                    st.error(f"Error: {response_data['error']}")
                else:
                    st.write(response_data["response"])

                    # Show response metadata
                    if response_data.get("source_documents"):
                        st.caption(
                            f"⏱️ Response time: {response_data['response_time']:.2f}s"
                        )

                        with st.expander(
                            f"📚 Sources ({len(response_data['source_documents'])} documents)",
                            expanded=False,
                        ):
                            for i, doc in enumerate(response_data["source_documents"]):
                                st.write(f"**Document {i + 1}:** {doc['filename']}")
                                st.write(
                                    f"**Similarity:** {doc['similarity_score']:.3f}"
                                )
                                if doc.get("chunk_id") is not None:
                                    st.write(f"**Chunk:** {doc['chunk_id'] + 1}")
                                st.write(f"**Preview:** {doc['content_preview']}")
                                st.divider()
                    else:
                        st.warning("No relevant documents found for this query.")

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_data["response"],
                        "metadata": response_data,
                    }
                )

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                logger.error(f"Chat error: {e}\n{traceback.format_exc()}")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_chain.clear_history()
        st.rerun()


def document_upload_interface():
    """Document upload and management interface"""
    st.title("📄 Document Management")
    st.write("Upload documents to add them to your knowledge base.")

    # File upload
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "md"],
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s):")

        for file in uploaded_files:
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            st.write(f"- **{file.name}** ({file_size_mb:.2f} MB)")

        # Processing options
        col1, col2 = st.columns(2)

        with col1:
            add_metadata = st.checkbox("Add custom metadata", value=False)

        with col2:
            if st.button("🚀 Process Documents", type="primary"):
                process_documents(uploaded_files, add_metadata)


def process_documents(uploaded_files, add_metadata: bool):
    """Process and add documents to vector store"""
    doc_processor = st.session_state.doc_processor

    progress_bar = st.progress(0)
    status_text = st.empty()

    successfully_processed = 0
    failed_files = []

    for i, file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))

            # Prepare additional metadata
            additional_metadata = {"upload_timestamp": datetime.now().isoformat()}

            if add_metadata:
                # In a real app, you might want to collect this metadata from the user
                additional_metadata.update(
                    {"uploaded_by": "user", "category": "general"}
                )

            # Process the file
            documents = doc_processor.process_file(file, additional_metadata)

            # Add to vector store
            success = st.session_state.rag_chain.add_documents(documents)

            if success:
                successfully_processed += 1
                st.success(
                    f"✅ Successfully processed {file.name} ({len(documents)} chunks)"
                )
            else:
                failed_files.append(file.name)
                st.error(f"❌ Failed to add {file.name} to vector store")

        except Exception as e:
            failed_files.append(file.name)
            st.error(f"❌ Error processing {file.name}: {e}")
            logger.error(
                f"Document processing error for {file.name}: {e}\n{traceback.format_exc()}"
            )

    # Show final results
    status_text.text("Processing complete!")
    progress_bar.progress(1.0)

    st.success(
        f"Successfully processed {successfully_processed} out of {len(uploaded_files)} files!"
    )

    if failed_files:
        st.error(f"Failed to process: {', '.join(failed_files)}")

    # Update vector store info display
    st.rerun()


def main():
    """Main application function"""
    # Validate configuration
    validate_configuration()

    # Initialize components
    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_chain = initialize_rag_chain()

    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = initialize_document_processor()

    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 RAG Assistant")

        page = st.selectbox("Navigate", ["💬 Chat", "📄 Manage Documents"], index=0)

        st.divider()

        # Display system information
        display_health_status()
        display_vector_store_info()

        st.divider()

        # Additional controls
        if st.button("🔄 Refresh System Status"):
            st.rerun()

        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Main content based on navigation
    if page == "💬 Chat":
        chat_interface()
    elif page == "📄 Manage Documents":
        document_upload_interface()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")
        st.write("Please check the logs for more details.")
