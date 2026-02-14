# ============================================================
# Enterprise-RAG: API Authentication
# ============================================================
"""
API key authentication for the RAG API.

This module provides:
- API key validation via X-API-Key header
- Optional authentication (no auth if no keys configured)
- Integration with FastAPI dependency injection

Example:
    >>> from src.api.auth import verify_api_key
    >>> from fastapi import Depends
    >>>
    >>> @router.post("/query")
    >>> async def query(api_key: str = Depends(verify_api_key)):
    ...     # Protected endpoint
    ...     pass
"""

import os
from typing import Optional

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# API Key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load valid API keys from environment (comma-separated)
_API_KEYS_ENV = os.getenv("API_KEYS", "")
VALID_API_KEYS: list[str] = (
    [key.strip() for key in _API_KEYS_ENV.split(",") if key.strip()]
    if _API_KEYS_ENV
    else []
)

# Check if authentication is enabled
AUTH_ENABLED = bool(VALID_API_KEYS)


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """
    Verify API key from X-API-Key header.

    If no API keys are configured (API_KEYS env var empty), authentication
    is disabled and all requests are allowed (development mode).

    Args:
        api_key: API key from request header

    Returns:
        The validated API key (or "dev-mode" if auth disabled)

    Raises:
        HTTPException: 401 if API key is missing or invalid

    Example:
        >>> # In a route handler:
        >>> @router.post("/query")
        >>> async def query(api_key: str = Depends(verify_api_key)):
        ...     return {"api_key": api_key}
    """
    # If no API keys configured, allow all requests (development mode)
    if not VALID_API_KEYS:
        return "dev-mode"

    # Validate API key
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def is_auth_enabled() -> bool:
    """
    Check if authentication is enabled.

    Returns:
        True if API keys are configured, False otherwise
    """
    return AUTH_ENABLED


def get_valid_api_keys() -> list[str]:
    """
    Get list of configured API keys (for debugging/testing).

    Returns:
        List of valid API key prefixes (first 8 chars only for security)
    """
    return [f"{key[:8]}..." if len(key) > 8 else "***" for key in VALID_API_KEYS]
