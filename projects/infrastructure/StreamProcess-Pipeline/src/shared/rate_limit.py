# ============================================================
# StreamProcess-Pipeline: Rate Limiting Module
# ============================================================
"""
Rate limiting utilities for the Stream Processing Pipeline.

This module provides:
- limiter: Configured SlowAPI rate limiter instance
- rate_limit_exception_handler: FastAPI exception handler for rate limit errors
- RateLimitExceeded: Custom exception for rate limit violations
"""

import logging
from typing import Any, Dict

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded as SlowAPIRateLimitExceeded


logger = logging.getLogger(__name__)


# ============================================================
# Exception Classes
# ============================================================

class RateLimitExceeded(Exception):
    """
    Exception raised when a rate limit is exceeded.

    This exception is raised when a client makes too many requests
    within a given time period.

    Attributes:
        detail: Human-readable error message
        retry_after: Seconds until the client can retry
    """

    def __init__(self, detail: str = "Rate limit exceeded", retry_after: int = 60) -> None:
        """
        Initialize the RateLimitExceeded exception.

        Args:
            detail: Error message
            retry_after: Seconds until retry is allowed
        """
        self.detail = detail
        self.retry_after = retry_after
        super().__init__(detail)


# ============================================================
# Rate Limiter Configuration
# ============================================================

def _get_identifier(request: Request) -> str:
    """
    Get the unique identifier for rate limiting.

    Uses the remote address by default, but can be extended to use
    API keys or authenticated user IDs.

    Args:
        request: The FastAPI request object

    Returns:
        Unique identifier string
    """
    # Try to get user ID from authentication first
    # Fall back to IP address if not authenticated
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # In a real implementation, decode the token and get user ID
        # For now, use IP address
        pass

    return get_remote_address(request)


# Create the limiter instance
limiter = Limiter(
    key_func=_get_identifier,
    default_limits=["200/minute", "1000/hour"],
    storage_uri="memory://",
    headers_enabled=True,
    strategy="fixed-window",  # Options: fixed-window, moving-window, fixed-window-elastic-expiry
)


# ============================================================
# Exception Handler
# ============================================================

async def rate_limit_exception_handler(
    request: Request,
    exc: RateLimitExceeded | SlowAPIRateLimitExceeded
) -> Response:
    """
    FastAPI exception handler for rate limit violations.

    Returns a 429 Too Many Requests response with appropriate headers.

    Args:
        request: The FastAPI request object
        exc: The rate limit exception

    Returns:
        JSON response with error details
    """
    # Handle both our custom exception and SlowAPI's exception
    if isinstance(exc, SlowAPIRateLimitExceeded):
        detail = str(exc)
        retry_after = getattr(exc, "retry_after", 60)
    else:
        detail = exc.detail
        retry_after = exc.retry_after

    logger.warning(
        f"Rate limit exceeded for {request.client.host if request.client else 'unknown'}: {detail}"
    )

    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": detail,
            "retry_after": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": "200",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(retry_after),
        },
    )


# ============================================================
# Decorator Helpers
# ============================================================

def limit(
    limit_value: str,
    key_func: Any | None = None,
    exempt_when: Any | None = None,
    override_defaults: bool = False,
) -> Any:
    """
    Convenience wrapper for the limiter.limit decorator.

    Args:
        limit_value: Rate limit string (e.g., "10/minute", "100/hour")
        key_func: Optional custom key function
        exempt_when: Optional function to determine when to exempt
        override_defaults: Whether to override default limits

    Returns:
        Decorator function

    Example:
        >>> @app.get("/api/endpoint")
        >>> @limit("5/minute")
        >>> async def endpoint():
        >>>     return {"message": "limited"}
    """
    return limiter.limit(
        limit_value,
        key_func=key_func,
        exempt_when=exempt_when,
        override_defaults=override_defaults,
    )


# ============================================================
# Storage Backends
# ============================================================

def get_redis_storage_uri(redis_url: str | None = None) -> str:
    """
    Get a Redis storage URI for distributed rate limiting.

    Args:
        redis_url: Custom Redis URL (uses default if None)

    Returns:
        Redis storage URI
    """
    if redis_url is None:
        import os
        redis_url = os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/1"
        )
    return redis_url


def configure_redis_storage(redis_url: str) -> None:
    """
    Reconfigure the limiter to use Redis storage.

    This is useful for distributed systems where multiple instances
    need to share rate limit state.

    Args:
        redis_url: Redis connection URL
    """
    global limiter

    # Create a new limiter with Redis storage
    limiter = Limiter(
        key_func=_get_identifier,
        storage_uri=redis_url,
        headers_enabled=True,
        strategy="fixed-window",
    )

    logger.info(f"Rate limiter configured with Redis storage: {redis_url}")
