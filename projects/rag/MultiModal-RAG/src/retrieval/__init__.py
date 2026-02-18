# ============================================================
# Enterprise-RAG: Retrieval Module
# ============================================================
"""
Document retrieval strategies and embedding services.

This module provides:
- Embedding generation with caching
- Vector store abstraction (ChromaDB)
- Dense vector retrieval
- Sparse BM25 retrieval
- Hybrid retrieval strategies
- Cross-encoder reranking
"""

from src.retrieval.embedding_service import EmbeddingService, create_embedding_service
from src.retrieval.vector_store import (
    ChromaVectorStore,
    SearchResult,
    VectorStoreBase,
    VectorStoreStats,
    create_vector_store,
    create_vector_store_from_settings,
)
from src.retrieval.hybrid_retriever import create_hybrid_retriever
from src.retrieval.reranker import CrossEncoderReranker

__all__ = [
    # Embedding service
    "EmbeddingService",
    "create_embedding_service",
    # Vector store
    "VectorStoreBase",
    "ChromaVectorStore",
    "SearchResult",
    "VectorStoreStats",
    "create_vector_store",
    "create_vector_store_from_settings",
    # Hybrid retrieval
    "create_hybrid_retriever",
    # Reranking
    "CrossEncoderReranker",
]
