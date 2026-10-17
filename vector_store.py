import logging
import uuid
from datetime import datetime
from typing import Any

from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """Handle vector storage operations with Qdrant"""

    def __init__(self):
        self.client = None
        self.embeddings = None
        self.vector_store = None
        self._initialize_client()
        self._initialize_embeddings()
        self._initialize_collection()

    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            self.client = QdrantClient(path="local_qdrant")

            # Test connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant.")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Could not connect to Qdrant database: {e}") from e

    def _initialize_embeddings(self):
        """Initialize Azure OpenAI embeddings"""
        try:
            api_key = (
                SecretStr(Config.AZURE_OPENAI_API_KEY)
                if Config.AZURE_OPENAI_API_KEY
                else None
            )

            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_key=api_key,
            )

            # Test embeddings by embedding a simple text
            test_embedding = self.embeddings.embed_query("test")
            if not test_embedding:
                raise ValueError("Embedding test failed")

            logger.info("Successfully initialized Azure OpenAI embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise ValueError(
                f"Could not initialize Azure OpenAI embeddings: {e}"
            ) from e

    def _initialize_collection(self):
        """
        Initialize Qdrant collection, automatically recreating it if vector
        dimensions have changed.
        """
        try:
            if self.client is None:
                raise RuntimeError("Qdrant client is not initialized")
            if self.embeddings is None:
                raise RuntimeError("Embeddings are not initialized")

            collection_name = Config.QDRANT_COLLECTION_NAME
            embedding_dim = len(self.embeddings.embed_query("test"))

            logger.info(
                f"Verifying Qdrant collection '{collection_name}' with embedding dimension {embedding_dim}."
            )

            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            collection_exists = collection_name in collection_names

            needs_recreation = False
            if collection_exists:
                logger.info(
                    f"Found existing collection: '{collection_name}'. Checking dimensions."
                )
                collection_info = self.client.get_collection(
                    collection_name=collection_name
                )

                collection_info_dict = collection_info.model_dump()
                config_params = collection_info_dict.get("config", {}).get("params", {})
                vectors_config = config_params.get("vectors")

                if not vectors_config:
                    raise ValueError(
                        "Could not find vector configuration in collection info."
                    )

                if "size" in vectors_config:
                    existing_dim = vectors_config["size"]
                elif "" in vectors_config:
                    existing_dim = vectors_config[""]["size"]
                else:
                    raise ValueError(
                        f"Could not determine vector size from config: {vectors_config}"
                    )

                if existing_dim != embedding_dim:
                    logger.warning(
                        f"Dimension mismatch in collection '{collection_name}'. Existing: {existing_dim}, New: {embedding_dim}. Collection will be recreated."
                    )
                    needs_recreation = True
                else:
                    logger.info(
                        "Existing collection dimensions match. Using existing collection."
                    )

            if not collection_exists or needs_recreation:
                self.client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    f"Ensured collection '{collection_name}' exists with {embedding_dim} dimensions."
                )

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
            logger.info("Successfully initialized QdrantVectorStore wrapper.")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}", exc_info=True)
            raise ValueError(f"Could not initialize Qdrant collection: {e}") from e

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store"""
        try:
            if not documents:
                raise ValueError("No documents provided")

            # Add timestamp to metadata
            for doc in documents:
                doc.metadata["indexed_at"] = datetime.now().isoformat()
                doc.metadata["document_id"] = str(uuid.uuid4())

            # Add documents to vector store
            if self.vector_store is None:
                raise RuntimeError("Vector store is not initialized")
            ids = self.vector_store.add_documents(documents)

            logger.info(
                f"Successfully added {len(documents)} documents to vector store"
            )
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise ValueError(f"Failed to add documents to vector store: {e}") from e

    def similarity_search(
        self, query: str, k: int = 5, score_threshold: float = 0.0
    ) -> list[tuple[Document, float]]:
        """Perform similarity search"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            # Perform similarity search with scores
            if self.vector_store is None:
                raise RuntimeError("Vector store is not initialized")
            results = self.vector_store.similarity_search_with_score(
                query=query.strip(), k=k
            )

            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results if score >= score_threshold
            ]

            logger.info(f"Found {len(filtered_results)} relevant documents for query")
            return filtered_results

        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise ValueError(f"Failed to perform similarity search: {e}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection"""
        try:
            if self.client is None:
                raise RuntimeError("Qdrant client is not initialized")
            collection_info = self.client.get_collection(Config.QDRANT_COLLECTION_NAME)

            return {
                "name": Config.QDRANT_COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count or 0,
                "indexed_vectors_count": collection_info.indexed_vectors_count or 0,
                "points_count": collection_info.points_count or 0,
                "status": collection_info.status,
                "optimizer_status": str(collection_info.optimizer_status)
                if collection_info.optimizer_status
                else "unknown",
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": e}

    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution)"""
        try:
            if self.client is None:
                raise RuntimeError("Qdrant client is not initialized")
            self.client.delete_collection(Config.QDRANT_COLLECTION_NAME)
            logger.info(f"Deleted collection: {Config.QDRANT_COLLECTION_NAME}")

            # Reinitialize collection
            self._initialize_collection()
            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def search_by_metadata(
        self, metadata_filter: dict[str, Any], limit: int = 10
    ) -> list[Document]:
        """Search documents by metadata"""
        try:
            # Convert metadata filter to Qdrant filter format
            filter_conditions = []
            for key, value in metadata_filter.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{key}", match=models.MatchValue(value=value)
                    )
                )

            if not filter_conditions:
                raise ValueError("No filter conditions provided")

            # Perform search
            if self.client is None:
                raise RuntimeError("Qdrant client is not initialized")
            search_result = self.client.scroll(
                collection_name=Config.QDRANT_COLLECTION_NAME,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=limit,
                with_payload=True,
            )

            # Convert results to Documents
            documents = []
            for point in search_result[
                0
            ]:  # search_result is a tuple (points, next_page_offset)
                if point.payload:
                    doc = Document(
                        page_content=point.payload.get("page_content", ""),
                        metadata=point.payload.get("metadata", {}),
                    )
                    documents.append(doc)

            logger.info(f"Found {len(documents)} documents matching metadata filter")
            return documents

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            raise ValueError(f"Failed to search by metadata: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Perform health check on the vector store"""
        health_status = {
            "qdrant_connection": False,
            "embeddings_working": False,
            "collection_exists": False,
            "collection_info": None,
            "errors": [],
        }

        try:
            # Check Qdrant connection
            if self.client is None:
                raise RuntimeError("Qdrant client is not initialized")
            self.client.get_collections()
            health_status["qdrant_connection"] = True
        except Exception as e:
            health_status["errors"].append(f"Qdrant connection failed: {e}")

        try:
            # Check embeddings
            if self.embeddings is None:
                raise RuntimeError("Embeddings are not initialized")
            self.embeddings.embed_query("health check")
            health_status["embeddings_working"] = True
        except Exception as e:
            health_status["errors"].append(f"Embeddings failed: {e}")

        try:
            # Check collection
            health_status["collection_info"] = self.get_collection_info()
            if "error" not in health_status["collection_info"]:
                health_status["collection_exists"] = True
        except Exception as e:
            health_status["errors"].append(f"Collection check failed: {e}")

        return health_status
