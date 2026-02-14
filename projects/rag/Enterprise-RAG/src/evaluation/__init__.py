# ============================================================
# Enterprise-RAG: Evaluation Module
# ============================================================
"""
RAG evaluation using RAGAS metrics.

This module provides:
- RAGAS metric integration
- Test dataset management
- Batch evaluation
- Evaluation reports
"""

from src.evaluation.rag_evaluator import (
    DEFAULT_TEST_SAMPLES,
    EvaluationResult,
    EvaluationSample,
    RAGEvaluator,
    create_evaluator,
)

__all__ = [
    "RAGEvaluator",
    "EvaluationSample",
    "EvaluationResult",
    "create_evaluator",
    "DEFAULT_TEST_SAMPLES",
]
