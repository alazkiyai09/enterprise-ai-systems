# ============================================================
# Enterprise-RAG: Core Module
# ============================================================
"""
Core RAG functionality including engine, document processing, and evaluation.

Note: This module re-exports key classes from their actual locations
in the generation, ingestion, and evaluation packages.
"""

from src.generation import RAGChain, create_rag_chain
from src.ingestion import DocumentProcessor, create_processor_from_settings
from src.evaluation import RAGEvaluator, create_evaluator

__all__ = [
    "RAGChain",
    "create_rag_chain",
    "DocumentProcessor",
    "create_processor_from_settings",
    "RAGEvaluator",
    "create_evaluator",
]
