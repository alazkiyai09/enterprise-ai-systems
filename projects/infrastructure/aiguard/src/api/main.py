# ============================================================
# AIGuard: FastAPI Application
# ============================================================
"""
FastAPI application for AIGuard security guardrails.

This module provides:
- RESTful API for security scanning
- Prompt injection detection
- PII detection and redaction
- Jailbreak detection
- Output filtering

Security Features:
- Request body size limits
- Rate limiting
- Production environment validation
- Secure CORS configuration
"""

import logging
import os
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, status, Request, Depends, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Shared modules
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

# AIGuard imports
from guardrails.prompt_injection.prompt_injection import (
    PromptInjectionDetector,
    ThreatType,
)
from guardrails.pii.pii_detector import PIIDetector
from guardrails.output_filter.output_guard import OutputGuard
from guardrails.jailbreak.jailbreak_detector import JailbreakDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Security Configuration
# ============================================================

# Maximum request body size (10MB)
MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024

# Known weak secret patterns that should never be used in production
WEAK_SECRET_PATTERNS = [
    "change-in-production",
    "changeme",
    "secret",
    "password",
    "aiguard-super-secret",
    "aiguard-secret-key",
    "not-for-production",
]


def validate_production_security():
    """
    Validate security configuration for production environment.

    Raises:
        RuntimeError: If critical security issues are detected
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()

    # Only enforce strict validation in production
    if environment not in ("production", "prod"):
        logger.info(f"Running in {environment} mode - security validation relaxed")
        return

    logger.info("Running in PRODUCTION mode - enforcing strict security validation")

    # Check SECRET_KEY is set
    secret_key = os.getenv("SECRET_KEY", "")
    if not secret_key:
        raise RuntimeError(
            "CRITICAL: SECRET_KEY environment variable must be set in production. "
            "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    # Check for weak secrets
    secret_lower = secret_key.lower()
    for pattern in WEAK_SECRET_PATTERNS:
        if pattern in secret_lower:
            raise RuntimeError(
                f"CRITICAL: SECRET_KEY contains weak pattern '{pattern}'. "
                "Generate a secure random key for production."
            )

    # Check SECRET_KEY length
    if len(secret_key) < 32:
        raise RuntimeError(
            f"CRITICAL: SECRET_KEY is too short ({len(secret_key)} chars). "
            "Must be at least 32 characters for production."
        )

    # Warn about InMemoryUserStore in production
    logger.warning(
        "SECURITY WARNING: InMemoryUserStore is being used in production. "
        "All user data will be lost on container restart. "
        "Configure a database-backed UserStore for production."
    )

    # Check CORS origins are not localhost in production
    cors_origins = os.getenv("CORS_ORIGINS", "")
    if "localhost" in cors_origins or "127.0.0.1" in cors_origins:
        logger.warning(
            "SECURITY WARNING: CORS_ORIGINS contains localhost in production. "
            "This should be set to actual frontend domains."
        )


# Run production security validation on startup
try:
    validate_production_security()
except RuntimeError as e:
    logger.error(f"Security validation failed: {e}")
    raise


# ============================================================
# Pydantic Models
# ============================================================

class ScanRequest(BaseModel):
    """Request model for security scanning."""
    text: str = Field(..., description="Text to scan for security threats", min_length=1)
    enable_injection_check: bool = Field(True, description="Enable prompt injection detection")
    enable_pii_check: bool = Field(True, description="Enable PII detection")
    enable_jailbreak_check: bool = Field(True, description="Enable jailbreak detection")
    enable_encoding_check: bool = Field(True, description="Enable encoding attack detection")
    redact_pii: bool = Field(True, description="Redact detected PII")
    redaction_mode: str = Field("full", description="Redaction mode: full, partial, token, mask, hash")


class ScanResponse(BaseModel):
    """Response model for security scan results."""
    is_safe: bool = Field(..., description="Overall safety status")
    threats_found: int = Field(..., description="Number of threats detected")
    checks: dict = Field(..., description="Detailed check results")
    sanitized_text: Optional[str] = Field(None, description="Sanitized version of input")


class PIIRequest(BaseModel):
    """Request model for PII detection."""
    text: str = Field(..., description="Text to scan for PII", min_length=1)
    redaction_mode: str = Field("full", description="Redaction mode: full, partial, token, mask, hash")


class PIIResponse(BaseModel):
    """Response model for PII detection results."""
    has_pii: bool = Field(..., description="Whether PII was detected")
    matches: List[dict] = Field(..., description="PII matches found")
    redacted_text: str = Field(..., description="Text with PII redacted")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="AIGuard API",
    description="""
## Security Guardrails for LLM Applications

Protect your LLM applications from:
- **Prompt Injection**: Detect and block malicious prompt injection attempts
- **Jailbreaking**: Identify DAN, developer mode, and persona jailbreaks
- **PII Leakage**: Detect and redact personally identifiable information
- **Encoding Attacks**: Catch Base64, hex, and Unicode evasion attempts
- **Output Filtering**: Filter responses for data leakage and toxicity

### Quick Start
1. **Scan Text**: POST /scan with your input text
2. **PII Check**: POST /pii to detect and redact PII
3. **Health Check**: GET /health for service status
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================================
# Request Size Limit Middleware
# ============================================================

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """
    Middleware to limit request body size.

    Prevents DoS attacks via large request bodies.
    """
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > MAX_REQUEST_BODY_SIZE:
                    logger.warning(
                        f"Request body too large: {length} bytes (max: {MAX_REQUEST_BODY_SIZE})"
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "error": "Request entity too large",
                            "detail": f"Request body exceeds maximum size of {MAX_REQUEST_BODY_SIZE} bytes",
                            "max_size": MAX_REQUEST_BODY_SIZE,
                        }
                    )
            except ValueError:
                pass

    return await call_next(request)


# ============================================================
# CORS Middleware
# ============================================================

origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")

# Explicit allowed headers (not wildcard for better security)
ALLOWED_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
    "x-api-key",
    "x-user-id",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=ALLOWED_HEADERS,
)

# Install security filter for logs
install_security_filter(logger)

# Register error handlers
register_error_handlers(app)

# Register rate limit exception handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# Get settings
settings = get_settings()


# ============================================================
# Global Detectors
# ============================================================

# Initialize detectors on startup (lazy loading)
_injection_detector: Optional[PromptInjectionDetector] = None
_pii_detector: Optional[PIIDetector] = None
_jailbreak_detector: Optional[JailbreakDetector] = None
_output_guard: Optional[OutputGuard] = None


def get_injection_detector() -> PromptInjectionDetector:
    """Get or create injection detector."""
    global _injection_detector
    if _injection_detector is None:
        _injection_detector = PromptInjectionDetector(
            embedding_model="all-MiniLM-L6-v2",
            similarity_threshold=0.75,
        )
    return _injection_detector


def get_pii_detector() -> PIIDetector:
    """Get or create PII detector."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector(redaction_mode="full")
    return _pii_detector


def get_jailbreak_detector() -> JailbreakDetector:
    """Get or create jailbreak detector."""
    global _jailbreak_detector
    if _jailbreak_detector is None:
        _jailbreak_detector = JailbreakDetector(threshold=0.80)
    return _jailbreak_detector


def get_output_guard() -> OutputGuard:
    """Get or create output guard."""
    global _output_guard
    if _output_guard is None:
        _output_guard = OutputGuard(pii_detector=get_pii_detector())
    return _output_guard


# ============================================================
# Authentication
# ============================================================

# In-memory user store for demo (replace with database in production)
user_store = InMemoryUserStore()


@app.post("/auth/register", tags=["Authentication"])
@limiter.limit("10/hour")
async def register(
    user_data: UserCreate,
    request: Request,
):
    """Register a new user."""
    try:
        user = await user_store.create_user(user_data)
        return {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/auth/login", tags=["Authentication"])
@limiter.limit("20/minute")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    request: Request = None,
):
    """Login and receive access token."""
    from shared.auth import authenticate_user, login_user

    user = await authenticate_user(username, password, user_store)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = await login_user(username, password, user_store)
    return token_data


@app.post("/auth/refresh", tags=["Authentication"])
@limiter.limit("30/minute")
async def refresh(
    refresh_token: str = Body(..., embed=True),
    request: Request = None,
):
    """Refresh access token."""
    from shared.auth import refresh_user_token

    token_data = await refresh_user_token(refresh_token, user_store)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


@app.get("/auth/me", tags=["Authentication"])
@limiter.limit("60/minute")
async def get_current_user_info(request: Request, current_user: TokenData = Depends(get_current_user)):
    """Get current user information."""
    user = await user_store.get_user_by_id(current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "is_active": user.is_active,
    }


# ============================================================
# Security Endpoints
# ============================================================

@app.post("/scan", response_model=ScanResponse, tags=["Security"])
@limiter.limit("30/minute")
async def scan_text(request: Request, scan_request: ScanRequest):
    """
    Scan text for security threats.

    Performs comprehensive security checks including:
    - Prompt injection detection
    - Jailbreak detection
    - PII detection (with optional redaction)
    - Encoding attack detection
    """
    results = {
        "is_safe": True,
        "threats_found": 0,
        "checks": {},
        "sanitized_text": scan_request.text,
    }

    sanitized = scan_request.text

    # Prompt Injection Detection
    if scan_request.enable_injection_check:
        detector = get_injection_detector()
        injection_result = detector.detect(sanitized)
        results["checks"]["injection"] = {
            "safe": injection_result.is_safe,
            "threat_type": injection_result.threat_type.value,
            "confidence": injection_result.confidence,
            "details": injection_result.details,
        }
        if not injection_result.is_safe:
            results["is_safe"] = False
            results["threats_found"] += 1
        sanitized = injection_result.sanitized_input

    # Jailbreak Detection
    if scan_request.enable_jailbreak_check:
        detector = get_jailbreak_detector()
        jailbreak_result = detector.detect(sanitized)
        results["checks"]["jailbreak"] = {
            "safe": jailbreak_result.is_safe,
            "threat_type": jailbreak_result.threat_type.value,
            "confidence": jailbreak_result.confidence,
            "details": jailbreak_result.details,
        }
        if not jailbreak_result.is_safe:
            results["is_safe"] = False
            results["threats_found"] += 1
        sanitized = jailbreak_result.sanitized_input

    # Encoding Attack Detection
    if scan_request.enable_encoding_check:
        detector = get_injection_detector()
        has_encoding, encoding_type = detector.check_encoding_attacks(sanitized)
        results["checks"]["encoding"] = {
            "safe": not has_encoding,
            "encoding_type": encoding_type if has_encoding else None,
        }
        if has_encoding:
            results["is_safe"] = False
            results["threats_found"] += 1

    # PII Detection
    if scan_request.enable_pii_check:
        detector = get_pii_detector()
        if scan_request.redact_pii:
            # Update redaction mode
            detector.redaction_mode = scan_request.redaction_mode
        has_pii, pii_matches = detector.detect(sanitized)
        redacted = detector.redact(sanitized)
        results["checks"]["pii"] = {
            "has_pii": has_pii,
            "matches": len(pii_matches),
            "redacted_text": redacted,
        }
        sanitized = redacted

    results["sanitized_text"] = sanitized

    return ScanResponse(**results)


@app.post("/pii", response_model=PIIResponse, tags=["Security"])
@limiter.limit("30/minute")
async def detect_pii(request: Request, pii_request: PIIRequest):
    """
    Detect and redact PII in text.

    Scans for common PII types including:
    - Social Security Numbers (SSN)
    - Credit card numbers
    - Email addresses
    - Phone numbers
    - IP addresses
    - Dates of birth
    """
    detector = get_pii_detector()
    detector.redaction_mode = pii_request.redaction_mode

    has_pii, pii_matches = detector.detect(pii_request.text)
    redacted = detector.redact(pii_request.text)

    return PIIResponse(
        has_pii=has_pii,
        matches=[
            {
                "type": match.pii_type,
                "start": match.start,
                "end": match.end,
                "match": match.match,
            }
            for match in pii_matches
        ],
        redacted_text=redacted,
    )


@app.get("/health", tags=["Health"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns the health status of the API and its components.
    Returns 503 if critical components are not ready.
    """
    components = {
        "api": "healthy",
        "injection_detector": "ready" if _injection_detector else "not_loaded",
        "pii_detector": "ready" if _pii_detector else "not_loaded",
        "jailbreak_detector": "ready" if _jailbreak_detector else "not_loaded",
    }

    # Check if all critical components are ready
    # For a security service, all detectors should be available
    all_ready = all(
        v in ("ready", "healthy") for v in components.values()
    )

    response_data = {
        "status": "healthy" if all_ready else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": components,
    }

    if not all_ready:
        logger.warning(f"Health check returning degraded status: {components}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response_data
        )

    return response_data


@app.get("/ready", tags=["Health"])
@limiter.limit("60/minute")
async def readiness_check(request: Request):
    """
    Readiness check endpoint for Kubernetes/container orchestration.

    Returns 200 only when all components are initialized and ready.
    Returns 503 if any component is not ready.
    """
    # Force initialization of detectors to check readiness
    try:
        get_injection_detector()
        get_pii_detector()
        get_jailbreak_detector()
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["Root"])
@limiter.limit("60/minute")
async def root(request: Request):
    """
    Root endpoint.

    Returns API information and available endpoints.
    """
    return {
        "name": "AIGuard API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "endpoints": {
            "scan": "/scan",
            "pii": "/pii",
            "health": "/health",
            "ready": "/ready",
        },
    }


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Run the FastAPI application.

    Usage:
        python -m src.api.main
    """
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
