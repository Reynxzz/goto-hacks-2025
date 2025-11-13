"""RAG Milvus Tool for CrewAI - Semantic search across internal knowledge base"""
import os
import json
import requests
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pymilvus import MilvusClient

from src.config.settings import get_settings
from src.config.constants import (
    RAG_COLLECTION_NAME,
    DEFAULT_RAG_TOP_K,
    DEFAULT_EMBEDDING_TIMEOUT
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGMilvusToolSchema(BaseModel):
    """Input schema for RAGMilvusTool."""
    query: str = Field(..., description="Search query to find relevant information in the knowledge base")


class RAGMilvusTool(BaseTool):
    """
    Tool for searching internal knowledge base using semantic search.

    Uses Milvus vector database with a single 'combined_item' collection
    containing data from multiple sources (user_income, user_occupation, dge,
    genie, push_notifications, pills, etc.). Results include source metadata.
    """

    name: str = "Internal Knowledge Base Search"
    description: str = (
        "⚠️ THIS IS NOT THE GITLAB TOOL ⚠️\n"
        "Searches the internal knowledge base using semantic search (NOT GitLab repositories). "
        "The database contains a 'combined_item' collection with data from multiple sources. "
        "Results include text content and source field (e.g., user_income, dge, genie, pills). "
        "This tool returns knowledge base articles, NOT project files or commits. "
        "Useful for finding relevant information about user segments, data engineering, "
        "experimentation platforms (Genie), and internal tools. "
        "Input should be a natural language query about these topics. "
        "Output format: JSON with 'sources_found' and 'results' containing 'source' fields."
    )
    args_schema: Type[BaseModel] = RAGMilvusToolSchema

    # Pydantic fields for configuration
    db_path: str = ""
    model_name: str = ""
    embedding_endpoint: str = ""
    top_k: int = DEFAULT_RAG_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        db_path: str = None,
        model_name: str = None,
        embedding_endpoint: str = None,
        top_k: int = None,
        **kwargs
    ):
        """
        Initialize RAG Milvus tool.

        Args:
            db_path: Path to Milvus database file (optional, reads from settings)
            model_name: Embedding model name (optional, reads from settings)
            embedding_endpoint: URL for embedding generation (optional, reads from settings)
            top_k: Number of top results to return (optional, default from settings)
        """
        settings = get_settings()

        # Use provided values or fallback to settings
        super().__init__(
            db_path=db_path or settings.rag.db_path,
            model_name=model_name or settings.rag.embedding_model,
            embedding_endpoint=embedding_endpoint or settings.rag.embedding_endpoint,
            top_k=top_k or settings.rag.top_k,
            **kwargs
        )

        object.__setattr__(self, '_initialized', False)
        object.__setattr__(self, '_client', None)

        # Initialize Milvus client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Milvus client and verify database exists."""
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Milvus database not found at {self.db_path}")
                return

            client = MilvusClient(self.db_path)
            object.__setattr__(self, '_client', client)
            logger.info(f"RAG Milvus client initialized with database: {self.db_path}")
            object.__setattr__(self, '_initialized', True)
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")

    def is_available(self) -> bool:
        """Check if RAG tool is available."""
        return getattr(self, '_initialized', False)

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }
        headers = {"Content-Type": "application/json"}

        try:
            logger.debug(f"Generating embedding for query: {text[:50]}...")
            response = requests.post(
                self.embedding_endpoint,
                json=payload,
                headers=headers,
                timeout=DEFAULT_EMBEDDING_TIMEOUT
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
            logger.debug(f"Successfully generated embedding: {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            error_msg = f"Failed to generate embedding: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _run(self, query: str) -> str:
        """
        Search the knowledge base for relevant information.

        The database uses a single 'combined_item' collection with a 'source' field
        to indicate the origin of each document.

        Args:
            query: Natural language search query

        Returns:
            JSON string with search results including text and source information
        """
        logger.info("=" * 80)
        logger.info("🔍 INTERNAL KNOWLEDGE BASE SEARCH TOOL CALLED")
        logger.info(f"Query: {query}")
        logger.info("This is the RAG tool, NOT the GitLab tool!")
        logger.info("=" * 80)

        if not self.is_available():
            error_msg = "RAG Milvus tool not available. Check database path and initialization."
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            logger.info(f"Searching knowledge base for: {query}")

            # Generate embedding for query
            query_vector = self._generate_embedding(query)

            # Get Milvus client
            client = getattr(self, '_client', None)
            if not client:
                raise RuntimeError("Milvus client not initialized")

            # Search the combined collection
            logger.info(f"Searching collection: {RAG_COLLECTION_NAME}")
            search_results = client.search(
                collection_name=RAG_COLLECTION_NAME,
                data=[query_vector],
                limit=self.top_k,
                output_fields=["text", "source"]  # Include source field
            )

            # Format results
            results = []
            sources = set()

            for hit in search_results[0]:
                source = hit["entity"].get("source", "N/A")
                sources.add(source)

                results.append({
                    "score": float(hit["distance"]),
                    "text": hit["entity"]["text"],
                    "source": source
                })

            logger.info(f"Found {len(results)} results from {len(sources)} different sources")

            return json.dumps({
                "query": query,
                "collection": RAG_COLLECTION_NAME,
                "results_count": len(results),
                "sources_found": sorted(list(sources)),
                "results": results
            }, indent=2)

        except Exception as e:
            error_msg = f"Error searching knowledge base: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({
                "error": error_msg,
                "query": query
            })
