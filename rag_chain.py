import logging
from datetime import datetime
from typing import Any

from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from config import Config
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG (Retrieval-Augmented Generation) chain implementation"""

    def __init__(self):
        try:
            self.vector_store = VectorStore()
            self.llm = None
            self._initialize_llm()
            self.conversation_history = []
            self.system_prompt_template = self._create_system_prompt_template()
        except Exception as e:
            logger.critical(f"Failed to initialize RAGChain: {e}")
            raise

    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM"""
        try:
            api_key = (
                SecretStr(Config.AZURE_OPENAI_API_KEY)
                if Config.AZURE_OPENAI_API_KEY
                else None
            )

            self.llm = AzureChatOpenAI(
                azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_key=api_key,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
            )
            # Test the LLM
            self.llm.invoke("Hello")
            logger.info("Successfully initialized and tested Azure OpenAI LLM")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise ValueError(f"Could not initialize Azure OpenAI LLM: {e}") from e

    def _create_system_prompt_template(self) -> PromptTemplate:
        """Create system prompt template for the RAG chain"""
        prompt_str = """You are a helpful AI assistant that answers questions based on the provided context from documents.

Instructions:
1. Use ONLY the information provided in the 'Context from documents' to answer the question.
2. If the context does not contain enough information to provide a definitive answer, clearly state that the answer cannot be found in the provided documents. Do not use any external knowledge.
3. Be concise but comprehensive in your responses.
4. If you are unsure about something, express that uncertainty.
5. Maintain a helpful and professional tone.
6. If the user asks about the source of information, refer to the document metadata when available.

Context from documents:
{context}

Previous conversation:
{chat_history}"""  # noqa: E501
        return PromptTemplate(
            template=prompt_str, input_variables=["context", "chat_history"]
        )

    def _format_context(self, documents: list[tuple[Document, float]]) -> str:
        """Format retrieved documents as context"""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, (doc, score) in enumerate(documents):
            metadata = doc.metadata
            source_info = f"Source: {metadata.get('filename', 'Unknown')}"
            if "chunk_id" in metadata:
                source_info += f" (Chunk {metadata.get('chunk_id', -1) + 1})"  # Use .get with default

            context_parts.append(
                f"Document {i + 1} ({source_info}, Similarity Score: {score:.4f}):\n{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)

    def _format_chat_history(self) -> str:
        """Format conversation history"""
        if not self.conversation_history:
            return "No previous conversation."

        history_parts = []
        # Take the last 5 exchanges (10 messages)
        for entry in self.conversation_history[-5:]:
            history_parts.append(f"Human: {entry['human']}")
            history_parts.append(f"Assistant: {entry['ai']}")

        return "\n".join(history_parts)

    def retrieve_documents(
        self, query: str, k: int = 5, score_threshold: float = 0.5
    ) -> list[tuple[Document, float]]:
        """Retrieve relevant documents for the query"""
        try:
            documents = self.vector_store.similarity_search(
                query=query, k=k, score_threshold=score_threshold
            )
            logger.info(
                f"Retrieved {len(documents)} documents for query: {query[:50]}..."
            )
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def generate_response(self, query: str, use_history: bool = True) -> dict[str, Any]:
        """Generate response using RAG"""
        try:
            start_time = datetime.now()

            # 1. Retrieve relevant documents
            documents = self.retrieve_documents(query)

            # 2. Format context and chat history
            context = self._format_context(documents)
            chat_history = (
                self._format_chat_history()
                if use_history
                else "No previous conversation."
            )

            # 3. Create the prompt
            final_prompt = self.system_prompt_template.format(
                context=context, chat_history=chat_history
            )

            messages = [
                SystemMessage(content=final_prompt),
                HumanMessage(content=query),
            ]

            # 4. Generate response
            if self.llm is None:
                raise RuntimeError("LLM is not initialized")
            response = self.llm.invoke(messages)
            response_text = response.content

            # 5. Update conversation history
            if use_history:
                self.conversation_history.append(
                    {
                        "human": query,
                        "ai": response_text,
                        "timestamp": datetime.now().isoformat(),
                        "documents_used": len(documents),
                    }
                )
                # Keep history to a reasonable size
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]

            response_time = (datetime.now() - start_time).total_seconds()

            # 6. Prepare response metadata for the UI
            source_documents = [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:200] + "...",
                }
                for doc, score in documents
            ]

            result = {
                "response": response_text,
                "source_documents": source_documents,
                "response_time": response_time,
                "documents_retrieved": len(documents),
            }

            logger.info(
                f"Generated response in {response_time:.2f}s using {len(documents)} documents."
            )
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"I apologize, but I encountered an error: {e}",
                "source_documents": [],
                "error": e,
            }

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Cleared conversation history.")

    def add_documents(self, documents: list[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            logger.error(f"Error in RAGChain adding documents: {e}")
            return False

    def get_vector_store_info(self) -> dict[str, Any]:
        """Get information about the vector store"""
        return self.vector_store.get_collection_info()

    def health_check(self) -> dict[str, Any]:
        """Perform health check on the RAG chain and its components"""
        health_status = {
            "rag_chain_healthy": True,
            "llm_working": False,
            "vector_store_status": {},
            "errors": [],
        }

        # Check LLM
        try:
            if self.llm is None:
                raise RuntimeError("LLM is not initialized")
            self.llm.invoke("Health check")
            health_status["llm_working"] = True
        except Exception as e:
            health_status["errors"].append(f"LLM health check failed: {e}")
            health_status["rag_chain_healthy"] = False

        # Check vector store
        try:
            vs_health = self.vector_store.health_check()
            health_status["vector_store_status"] = vs_health
            if vs_health.get("errors"):
                health_status["errors"].extend(vs_health["errors"])
                health_status["rag_chain_healthy"] = False
        except Exception as e:
            health_status["errors"].append(f"Vector store health check failed: {e}")
            health_status["rag_chain_healthy"] = False

        return health_status
