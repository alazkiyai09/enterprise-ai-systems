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

from src.retrieval.embedding_service import (
    EmbeddingService,
    create_embedding_service,
)
from src.retrieval.vector_store import (
    ChromaVectorStore,
    SearchResult,
    VectorStoreBase,
    VectorStoreStats,
    create_vector_store,
    create_vector_store_from_settings,
)
from src.retrieval.sparse_retriever import (
    BM25Retriever,
    SparseSearchResult,
    BM25Stats,
    create_bm25_retriever,
)
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedSearchResult,
)
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    create_hybrid_retriever,
)

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
    # Sparse retrieval (BM25)
    "BM25Retriever",
    "SparseSearchResult",
    "BM25Stats",
    "create_bm25_retriever",
    # Reranker
    "CrossEncoderReranker",
    "RerankedSearchResult",
    # Hybrid retriever
    "HybridRetriever",
    "HybridSearchResult",
    "create_hybrid_retriever",
]
