# ============================================================
# Enterprise-RAG: Rate Limiting
# ============================================================
"""
Local rate limiting for the RAG API using slowapi.

This module provides:
- IP-based rate limiting
- Configurable limits per endpoint
- Custom exception handling

Example:
    >>> from src.api.rate_limit import limiter
    >>> from fastapi import Request
    >>>
    >>> @router.post("/query")
    >>> @limiter.limit("10/minute")
    >>> async def query(request: Request):
    ...     # Rate-limited endpoint
    ...     pass
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from typing import Callable

# Create the limiter instance
limiter = Limiter(key_func=get_remote_address)


async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom exception handler for rate limit exceeded.

    Args:
        request: FastAPI request object
        exc: RateLimitExceeded exception

    Returns:
        JSON response with rate limit error details
    """
    # Extract rate limit info from exception
    response = JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
            "retry_after": getattr(exc, "retry_after", None),
        },
    )
    return response


def get_rate_limit_key(request: Request) -> str:
    """
    Get the rate limit key for a request.

    Uses X-Forwarded-For header if available (for reverse proxies),
    otherwise falls back to client IP.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit key (IP address)
    """
    # Check for X-Forwarded-For header (reverse proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Use the first IP in the chain (original client)
        return forwarded.split(",")[0].strip()

    # Fall back to direct client IP
    return get_remote_address(request)


# Predefined rate limit decorators
def rate_limit_query() -> Callable:
    """Rate limit for query endpoints: 30/minute."""
    return limiter.limit("30/minute")


def rate_limit_ingest() -> Callable:
    """Rate limit for document ingestion: 10/minute."""
    return limiter.limit("10/minute")


def rate_limit_evaluation() -> Callable:
    """Rate limit for evaluation endpoints: 5/minute."""
    return limiter.limit("5/minute")


def rate_limit_global() -> Callable:
    """Global rate limit: 100/minute."""
    return limiter.limit("100/minute")
