import logging
import os
from typing import Any, ClassVar

from dotenv import load_dotenv

# Load environment variables
dot_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dot_env_path, verbose=True, override=True, encoding="utf-8")


class Config:
    """Configuration class for the RAG application"""

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    # Qdrant Configuration
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")

    # Application Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

    # File Upload Configuration
    MAX_FILE_SIZE: ClassVar[int] = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))
    ALLOWED_EXTENSIONS: ClassVar[list[str]] = os.getenv("ALLOWED_EXTENSIONS", ".txt,.pdf,.docx,.md").split(",")

    @classmethod
    def validate_config(cls) -> dict[str, Any]:
        """Validate required configuration values"""
        missing_configs = []
        validation_result = {"valid": True, "missing": [], "warnings": []}

        required_configs = [
            ("AZURE_OPENAI_API_KEY", cls.AZURE_OPENAI_API_KEY),
            ("AZURE_OPENAI_ENDPOINT", cls.AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_DEPLOYMENT_NAME", cls.AZURE_OPENAI_DEPLOYMENT_NAME),
            (
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                cls.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            ),
        ]

        for config_name, config_value in required_configs:
            if not config_value:
                missing_configs.append(config_name)

        if missing_configs:
            validation_result["valid"] = False
            validation_result["missing"] = missing_configs

        return validation_result

    @classmethod
    def setup_logging(cls) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=int(os.getenv("LOGGING_LEVEL", "10")),
            format="%(asctime)s | %(name)-7s | %(levelname)-7s | %(funcName)s : %(lineno)d | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("rag_app.log")],
        )


# Initialize logging
Config.setup_logging()
