# ============================================================
# StreamProcess-Pipeline: Authentication Module
# ============================================================
"""
Authentication and authorization utilities for the Stream Processing Pipeline.

This module provides:
- User models: User, UserCreate, Token, TokenData
- Role management: Role enum with permissions
- Token creation and validation: JWT-based authentication
- Dependency injection: get_current_user, require_role, require_admin
- User store: In-memory user management (replace with database in production)
"""

import asyncio
import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

# Security settings - should be overridden by environment variables in production
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


def get_secret_key() -> str:
    """Get the secret key from environment or use the default."""
    import os
    return os.getenv("JWT_SECRET_KEY", SECRET_KEY)


# ============================================================
# Role Management
# ============================================================

class Role(str, Enum):
    """
    User roles with hierarchical permissions.

    Roles:
        ADMIN: Full access to all resources
        OPERATOR: Can manage pipeline operations
        USER: Read-only access to most resources
        GUEST: Limited access to public endpoints
    """

    ADMIN = "admin"
    OPERATOR = "operator"
    USER = "user"
    GUEST = "guest"

    @classmethod
    def permissions(cls, role: "Role") -> List[str]:
        """
        Get the permissions for a given role.

        Args:
            role: The role to get permissions for

        Returns:
            List of permission strings
        """
        permissions_map = {
            cls.ADMIN: ["read", "write", "delete", "manage_users", "manage_system"],
            cls.OPERATOR: ["read", "write", "manage_pipelines"],
            cls.USER: ["read"],
            cls.GUEST: ["read_public"],
        }
        return permissions_map.get(role, [])


# ============================================================
# User Models
# ============================================================

class User(BaseModel):
    """
    User model representing an authenticated user.

    Attributes:
        id: Unique user identifier
        username: Username (unique)
        email: User email address
        role: User role determining permissions
        is_active: Whether the user account is active
        created_at: Timestamp when user was created
        updated_at: Timestamp when user was last updated
    """

    id: str
    username: str
    email: str
    role: Role
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    hashed_password: str = Field(default="", repr=False)

    class Config:
        use_enum_values = True


class UserCreate(BaseModel):
    """
    User creation model for registration.

    Attributes:
        username: Desired username
        email: User email address
        password: Plain text password (will be hashed)
        role: Initial role (defaults to USER)
    """

    username: str
    email: str
    password: str
    role: Role = Role.USER

    class Config:
        use_enum_values = True


class Token(BaseModel):
    """
    Token response model for login/refresh endpoints.

    Attributes:
        access_token: JWT access token
        refresh_token: JWT refresh token
        token_type: Token type (always "bearer")
        expires_in: Seconds until access token expires
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds


class TokenData(BaseModel):
    """
    Decoded token data from JWT.

    Attributes:
        user_id: User ID from the token
        username: Username from the token
        role: User role from the token
        exp: Token expiration timestamp
    """

    user_id: str
    username: str
    role: Role
    exp: Optional[datetime] = None

    class Config:
        use_enum_values = True


# ============================================================
# Password Hashing
# ============================================================

def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256.

    Note: In production, use bcrypt or argon2 instead.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return f"${salt}${hashed}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored hashed password

    Returns:
        True if password matches, False otherwise
    """
    try:
        salt, stored_hash = hashed_password.split("$")[1:]
        computed_hash = hashlib.sha256(f"{plain_password}{salt}".encode()).hexdigest()
        return secrets.compare_digest(computed_hash, stored_hash)
    except (ValueError, AttributeError):
        return False


# ============================================================
# Token Management
# ============================================================

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "type": "access",
    })

    return _encode_token(to_encode)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.

    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid.uuid4()),  # Unique token ID for revocation
    })

    return _encode_token(to_encode)


def _encode_token(data: Dict[str, Any]) -> str:
    """Encode a JWT token with the configured secret."""
    # Convert datetime to timestamp
    data_copy = data.copy()
    if 'exp' in data_copy and isinstance(data_copy['exp'], datetime):
        data_copy['exp'] = int(data_copy['exp'].timestamp())

    try:
        from jose import jwt
        return jwt.encode(data_copy, get_secret_key(), algorithm=ALGORITHM)
    except ImportError:
        # Fallback to simple base64 encoding if jwt is not available
        # This is NOT secure and should only be used for development
        import json
        import base64
        json_data = json.dumps(data_copy)
        return base64.b64encode(json_data.encode()).decode()


def verify_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token.

    Args:
        token: The JWT token to verify

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        from jose import jwt
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
    except ImportError:
        # Fallback for development without jwt
        import json
        import base64
        try:
            json_data = base64.b64decode(token.encode()).decode()
            payload = json.loads(json_data)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    username = payload.get("username")
    role = payload.get("role")
    exp = payload.get("exp")

    if user_id is None or username is None or role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check expiration if we have a timestamp
    if exp:
        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
        if exp_datetime < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        exp_datetime = None

    return TokenData(
        user_id=user_id,
        username=username,
        role=role,
        exp=exp_datetime,
    )


# ============================================================
# User Store
# ============================================================

class InMemoryUserStore:
    """
    In-memory user store for development and testing.

    In production, replace with a database-backed implementation.

    Attributes:
        users: Dictionary of users by ID
        username_index: Index of users by username
        email_index: Index of users by email
    """

    def __init__(self) -> None:
        """Initialize the user store with a default admin user."""
        self.users: Dict[str, User] = {}
        self.username_index: Dict[str, str] = {}
        self.email_index: Dict[str, str] = {}
        self._lock = None  # Will be created when needed

        # Create default admin user synchronously
        self._create_admin_sync()

    def _create_admin_sync(self) -> None:
        """Create the default admin user synchronously."""
        admin = User(
            id="admin",
            username="admin",
            email="admin@localhost",
            role=Role.ADMIN,
            hashed_password=hash_password("admin123"),
        )
        self.users[admin.id] = admin
        self.username_index[admin.username.lower()] = admin.id
        self.email_index[admin.email.lower()] = admin.id

    async def _get_lock(self) -> asyncio.Lock:
        """Get or create the async lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _add_user(self, user: User) -> None:
        """Add a user to all indexes."""
        self.users[user.id] = user
        self.username_index[user.username.lower()] = user.id
        self.email_index[user.email.lower()] = user.id

    async def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user.

        Args:
            user_data: User creation data

        Returns:
            Created user

        Raises:
            ValueError: If username or email already exists
        """
        async with self._lock:
            # Check for existing user
            if user_data.username.lower() in self.username_index:
                raise ValueError(f"Username '{user_data.username}' already exists")

            if user_data.email.lower() in self.email_index:
                raise ValueError(f"Email '{user_data.email}' already exists")

            # Create new user
            user = User(
                id=str(uuid.uuid4()),
                username=user_data.username,
                email=user_data.email,
                role=user_data.role,
                hashed_password=hash_password(user_data.password),
            )

            await self._add_user(user)
            logger.info(f"Created user: {user.username}")

            return user

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.

        Args:
            user_id: User ID to look up

        Returns:
            User if found, None otherwise
        """
        return self.users.get(user_id)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get a user by username.

        Args:
            username: Username to look up

        Returns:
            User if found, None otherwise
        """
        user_id = self.username_index.get(username.lower())
        if user_id:
            return self.users.get(user_id)
        return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.

        Args:
            email: Email to look up

        Returns:
            User if found, None otherwise
        """
        user_id = self.email_index.get(email.lower())
        if user_id:
            return self.users.get(user_id)
        return None

    async def verify_credentials(self, username: str, password: str) -> Optional[User]:
        """
        Verify user credentials.

        Args:
            username: Username to verify
            password: Password to verify

        Returns:
            User if credentials are valid, None otherwise
        """
        user = await self.get_user_by_username(username)
        if user and user.is_active and verify_password(password, user.hashed_password):
            return user
        return None


# ============================================================
# Authentication Dependencies
# ============================================================

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    FastAPI dependency to get the current authenticated user.

    Args:
        credentials: Bearer token credentials

    Returns:
        Token data for the current user

    Raises:
        HTTPException: If no credentials provided or token is invalid
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    return verify_token(token)


async def require_role(*roles: Role) -> Callable[[TokenData], Awaitable[TokenData]]:
    """
    Factory function to create a dependency that requires specific roles.

    Args:
        *roles: Required roles (user must have at least one)

    Returns:
        Dependency function

    Example:
        >>> @app.get("/admin")
        >>> async def admin_endpoint(user: TokenData = Depends(require_role(Role.ADMIN))):
        >>>     return {"message": "Welcome admin"}
    """

    async def role_dependency(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        user_role = Role(current_user.role)

        if user_role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in roles]}",
            )

        return current_user

    return role_dependency


# Pre-configured role dependencies
require_admin = require_role(Role.ADMIN)
require_operator = require_role(Role.ADMIN, Role.OPERATOR)
require_user = require_role(Role.ADMIN, Role.OPERATOR, Role.USER)


# ============================================================
# Authentication Flows
# ============================================================

async def authenticate_user(
    username: str,
    password: str,
    user_store: InMemoryUserStore,
) -> Optional[User]:
    """
    Authenticate a user with username and password.

    Args:
        username: Username to authenticate
        password: Password to verify
        user_store: User store to verify against

    Returns:
        User if authenticated, None otherwise
    """
    return await user_store.verify_credentials(username, password)


async def login_user(
    username: str,
    password: str,
    user_store: InMemoryUserStore,
) -> Optional[Token]:
    """
    Login a user and return tokens.

    Args:
        username: Username to login
        password: Password to verify
        user_store: User store to verify against

    Returns:
        Token response with access and refresh tokens, or None if failed
    """
    user = await authenticate_user(username, password, user_store)

    if not user:
        return None

    # Get role value (handle both Role enum and string)
    role_value = user.role.value if hasattr(user.role, 'value') else user.role

    # Create access token
    access_token = create_access_token({
        "sub": user.id,
        "username": user.username,
        "role": role_value,
    })

    # Create refresh token
    refresh_token = create_refresh_token({
        "sub": user.id,
        "username": user.username,
        "role": role_value,
    })

    logger.info(f"User logged in: {user.username}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
    )


async def refresh_user_token(
    refresh_token: str,
    user_store: InMemoryUserStore,
) -> Optional[Token]:
    """
    Refresh an access token using a refresh token.

    Args:
        refresh_token: The refresh token
        user_store: User store to verify user exists

    Returns:
        New token response, or None if refresh token is invalid
    """
    try:
        token_data = verify_token(refresh_token)

        # Verify user still exists
        user = await user_store.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            return None

        # Create new tokens
        access_token = create_access_token({
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
        })

        new_refresh_token = create_refresh_token({
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
        })

        logger.info(f"Token refreshed for user: {user.username}")

        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
        )

    except Exception as e:
        logger.warning(f"Token refresh failed: {e}")
        return None
