# ============================================================
# StreamProcess-Pipeline: Error Handling Module
# ============================================================
"""
Error handling utilities for the Stream Processing Pipeline.

This module provides:
- Custom exception classes for domain-specific errors
- Error handler registration for FastAPI
- Consistent error response formatting
"""

import logging
import traceback
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


# ============================================================
# Exception Classes
# ============================================================

class APIError(Exception):
    """
    Base exception class for API errors.

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code to return
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the API error.

        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class AuthenticationError(APIError):
    """
    Exception raised for authentication failures.

    This should be used when credentials are invalid, missing, or expired.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the authentication error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthorizationError(APIError):
    """
    Exception raised for authorization failures.

    This should be used when a user is authenticated but lacks
    permission to perform an action.
    """

    def __init__(
        self,
        message: str = "Insufficient permissions",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the authorization error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ValidationError(APIError):
    """
    Exception raised for validation errors.

    This should be used when request data fails validation.
    """

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the validation error.

        Args:
            message: Error message
            details: Additional error details (e.g., field errors)
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class NotFoundError(APIError):
    """
    Exception raised when a requested resource is not found.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the not found error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class ConflictError(APIError):
    """
    Exception raised for conflict errors.

    This should be used when a request conflicts with existing state.
    """

    def __init__(
        self,
        message: str = "Resource conflict",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the conflict error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class RateLimitError(APIError):
    """
    Exception raised for rate limit errors.

    This should be used when a client exceeds rate limits.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the rate limit error.

        Args:
            message: Error message
            retry_after: Seconds until retry is allowed
            details: Additional error details
        """
        merged_details = {**(details or {}), "retry_after": retry_after}
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=merged_details,
        )
        self.retry_after = retry_after


class DatabaseError(APIError):
    """
    Exception raised for database-related errors.

    This should be used when database operations fail.
    """

    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the database error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class ExternalAPIError(APIError):
    """
    Exception raised for external API failures.

    This should be used when calls to external services fail.
    """

    def __init__(
        self,
        message: str = "External API request failed",
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the external API error.

        Args:
            message: Error message
            service_name: Name of the external service
            details: Additional error details
        """
        merged_details = {**(details or {})}
        if service_name:
            merged_details["service"] = service_name
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=merged_details,
        )


class ServiceUnavailableError(APIError):
    """
    Exception raised when a service is unavailable.

    This should be used when required services are down.
    """

    def __init__(
        self,
        message: str = "Service unavailable",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the service unavailable error.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


# ============================================================
# Error Handlers
# ============================================================

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """
    Handler for APIError exceptions.

    Returns a JSON response with the error details.

    Args:
        request: The FastAPI request object
        exc: The APIError exception

    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": _get_error_type(exc),
            "message": exc.message,
            "details": exc.details,
        },
    )


async def http_exception_handler(request: Request, exc) -> JSONResponse:
    """
    Handler for FastAPI HTTPException.

    Provides consistent error response format.

    Args:
        request: The FastAPI request object
        exc: The HTTPException

    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "details": {},
        },
    )


async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    """
    Handler for FastAPI RequestValidationError.

    Formats validation errors consistently.

    Args:
        request: The FastAPI request object
        exc: The RequestValidationError

    Returns:
        JSON response with error details
    """
    from fastapi.exceptions import RequestValidationError

    # Format pydantic errors
    field_errors = {}
    for error in exc.errors():
        loc = " -> ".join(str(item) for item in error["loc"])
        field_errors[loc] = error["msg"]

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": {"fields": field_errors},
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler for unhandled exceptions.

    Logs the full traceback and returns a generic error message.

    Args:
        request: The FastAPI request object
        exc: The unhandled exception

    Returns:
        JSON response with error details
    """
    # Log the full traceback for debugging
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method},
    )

    # Return generic error to client
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An internal error occurred",
            "details": {"request_id": getattr(request.state, "request_id", "unknown")},
        },
    )


def _get_error_type(exc: APIError) -> str:
    """
    Get the error type name for an APIError.

    Args:
        exc: The APIError exception

    Returns:
        Error type string
    """
    error_type_map = {
        AuthenticationError: "authentication_error",
        AuthorizationError: "authorization_error",
        ValidationError: "validation_error",
        NotFoundError: "not_found_error",
        ConflictError: "conflict_error",
        RateLimitError: "rate_limit_error",
        DatabaseError: "database_error",
        ExternalAPIError: "external_api_error",
        ServiceUnavailableError: "service_unavailable_error",
    }
    return error_type_map.get(type(exc), "api_error")


# ============================================================
# Error Handler Registration
# ============================================================

def register_error_handlers(app: FastAPI) -> None:
    """
    Register all error handlers with the FastAPI application.

    This should be called during application initialization.

    Args:
        app: The FastAPI application instance

    Example:
        >>> app = FastAPI()
        >>> register_error_handlers(app)
    """
    # Register custom API error handler
    app.add_exception_handler(APIError, api_error_handler)

    # Register FastAPI HTTPException handler
    from fastapi import HTTPException
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Register validation error handler
    from fastapi.exceptions import RequestValidationError
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Register general exception handler (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Error handlers registered")


# ============================================================
# Error Response Utilities
# ============================================================

def create_error_response(
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    error_type: str = "error",
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """
    Create a standardized error response.

    Args:
        message: Error message
        status_code: HTTP status code
        error_type: Type of error
        details: Additional error details

    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": error_type,
            "message": message,
            "details": details or {},
        },
    )


def log_error(
    error: Exception,
    message: Optional[str] = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log an error with context.

    Args:
        error: The exception to log
        message: Optional custom message
        level: Logging level (default: ERROR)
    """
    msg = message or str(error)
    logger.log(
        level,
        msg,
        exc_info=error,
        extra={"error_type": type(error).__name__},
    )
