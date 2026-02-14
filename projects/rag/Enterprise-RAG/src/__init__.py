# ============================================================
# Enterprise-RAG: Production-Grade RAG System
# Main Package Initialization
# ============================================================
"""
Enterprise-RAG: A production-grade Retrieval-Augmented Generation system
with hybrid retrieval, cross-encoder reranking, and comprehensive evaluation.

This package provides:
- Hybrid retrieval combining dense vector search and sparse BM25
- Cross-encoder reranking for improved retrieval accuracy
- Multi-format document ingestion (PDF, DOCX, MD, TXT)
- RAGAS evaluation framework integration
- Production-ready FastAPI backend
- Streamlit demo interface

Example:
    >>> from src.generation import create_rag_chain
    >>> from src.config import settings
    >>> rag_chain = create_rag_chain()
    >>> response = rag_chain.query("What is the company's refund policy?")
    >>> print(response.answer)

Author: AI Engineer
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Engineer"
__license__ = "MIT"

# Import key classes for convenient access
from src.generation import RAGChain, create_rag_chain
from src.ingestion import DocumentProcessor, create_processor_from_settings
from src.config import settings

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "RAGChain",
    "create_rag_chain",
    "DocumentProcessor",
    "create_processor_from_settings",
    "settings",
]

# Package metadata
PYTHON_MIN_VERSION = (3, 11)
PACKAGE_NAME = "enterprise-rag"
DESCRIPTION = "Production-Grade RAG System with Hybrid Retrieval & Evaluation"

# Version info tuple for programmatic access
VERSION_INFO = tuple(int(x) for x in __version__.split(".") if x.isdigit())
