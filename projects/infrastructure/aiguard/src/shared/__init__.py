# ============================================================
# Shared Modules for AIGuard
# ============================================================
"""
Shared utilities for AIGuard application.

Includes:
- security: Sensitive data filtering for logs
- rate_limit: Rate limiting with slowapi
- auth: JWT authentication with bcrypt
- errors: Error handling utilities
- secrets: Configuration management
"""

from .security import SensitiveDataFilter, install_security_filter, redact_sensitive_data
from .rate_limit import limiter, rate_limit_exception_handler, RateLimitExceeded
from .auth import (
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
)
from .errors import (
    register_error_handlers,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    DatabaseError,
    ExternalAPIError,
)
from .secrets import get_settings

__all__ = [
    # Security
    "SensitiveDataFilter",
    "install_security_filter",
    "redact_sensitive_data",
    # Rate limiting
    "limiter",
    "rate_limit_exception_handler",
    "RateLimitExceeded",
    # Auth
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
    # Errors
    "register_error_handlers",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "DatabaseError",
    "ExternalAPIError",
    # Config
    "get_settings",
]
