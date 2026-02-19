# ============================================================
# StreamProcess-Pipeline: Shared Module
# ============================================================
"""
Shared module for the Stream Processing Pipeline.

This module provides common utilities and components used across
the application, including:
- Security: Sensitive data filtering for logs
- Rate limiting: API rate limiting with slowapi
- Authentication: JWT-based authentication and authorization
- Error handling: Custom exceptions and error handlers
- Configuration: Settings and secrets management
"""

from shared.security import SensitiveDataFilter, install_security_filter
from shared.rate_limit import limiter, rate_limit_exception_handler, RateLimitExceeded
from shared.auth import (
    get_current_user,
    require_role,
    require_admin,
    create_access_token,
    create_refresh_token,
    verify_token,
    User,
    UserCreate,
    Token,
    TokenData,
    Role,
    InMemoryUserStore,
    authenticate_user,
    login_user,
    refresh_user_token,
)
from shared.errors import (
    register_error_handlers,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    DatabaseError,
    ExternalAPIError,
)
from shared.secrets import get_settings

__all__ = [
    # Security
    "SensitiveDataFilter",
    "install_security_filter",
    # Rate limiting
    "limiter",
    "rate_limit_exception_handler",
    "RateLimitExceeded",
    # Authentication
    "get_current_user",
    "require_role",
    "require_admin",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "User",
    "UserCreate",
    "Token",
    "TokenData",
    "Role",
    "InMemoryUserStore",
    "authenticate_user",
    "login_user",
    "refresh_user_token",
    # Errors
    "register_error_handlers",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "DatabaseError",
    "ExternalAPIError",
    # Configuration
    "get_settings",
]
