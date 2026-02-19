# ============================================================
# StreamProcess-Pipeline: Configuration Module
# ============================================================
"""
Configuration and secrets management for the Stream Processing Pipeline.

This module provides:
- Settings class with environment variable loading
- Type-safe configuration access
- Validation for required settings
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


logger = logging.getLogger(__name__)


# ============================================================
# Configuration Classes
# ============================================================

@dataclass(frozen=True)
class DatabaseSettings:
    """
    Database connection settings.

    Attributes:
        url: Database connection URL
        pool_size: Connection pool size
        max_overflow: Max overflow connections
        pool_timeout: Connection pool timeout in seconds
    """

    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        """
        Create database settings from environment variables.

        Returns:
            DatabaseSettings instance
        """
        return cls(
            url=os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/streamprocess"
            ),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
        )


@dataclass(frozen=True)
class RedisSettings:
    """
    Redis connection settings.

    Attributes:
        url: Redis connection URL
        db: Redis database number
        decode_responses: Whether to decode responses to strings
    """

    url: str
    db: int = 0
    decode_responses: bool = True

    @classmethod
    def from_env(cls) -> "RedisSettings":
        """
        Create Redis settings from environment variables.

        Returns:
            RedisSettings instance
        """
        return cls(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            db=int(os.getenv("REDIS_DB", "0")),
            decode_responses=os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true",
        )


@dataclass(frozen=True)
class APISettings:
    """
    API server settings.

    Attributes:
        host: API host address
        port: API port
        workers: Number of worker processes
        log_level: Logging level
        cors_origins: Allowed CORS origins
        rate_limit_enabled: Whether rate limiting is enabled
    """

    host: str
    port: int
    workers: int = 1
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=list)
    rate_limit_enabled: bool = True

    @classmethod
    def from_env(cls) -> "APISettings":
        """
        Create API settings from environment variables.

        Returns:
            APISettings instance
        """
        cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "1")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            cors_origins=cors_origins,
            rate_limit_enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
        )


@dataclass(frozen=True)
class CelerySettings:
    """
    Celery task queue settings.

    Attributes:
        broker_url: Message broker URL
        result_backend: Result backend URL
        task_routes: Task routing configuration
        task_soft_time_limit: Soft task timeout
        task_time_limit: Hard task timeout
    """

    broker_url: str
    result_backend: str
    task_routes: dict = field(default_factory=dict)
    task_soft_time_limit: int = 300
    task_time_limit: int = 600

    @classmethod
    def from_env(cls) -> "CelerySettings":
        """
        Create Celery settings from environment variables.

        Returns:
            CelerySettings instance
        """
        return cls(
            broker_url=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
            result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
            task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "300")),
            task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "600")),
        )


@dataclass(frozen=True)
class LLMAPIKey:
    """
    LLM API key configuration.

    Attributes:
        provider: API provider name (openai, anthropic, etc.)
        key: API key value
        base_url: Optional custom base URL
    """

    provider: str
    key: str
    base_url: Optional[str] = None


@dataclass(frozen=True)
class LLMSettings:
    """
    LLM (Large Language Model) settings.

    Attributes:
        provider: Default LLM provider
        model: Default model name
        api_keys: Dictionary of API keys by provider
        temperature: Default sampling temperature
        max_tokens: Default max tokens
        timeout: Request timeout in seconds
    """

    provider: str
    model: str
    api_keys: List[LLMAPIKey] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60

    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """
        Get API key for a provider.

        Args:
            provider: Provider name (uses default if None)

        Returns:
            API key if found, None otherwise
        """
        provider = provider or self.provider
        for key_config in self.api_keys:
            if key_config.provider.lower() == provider.lower():
                return key_config.key
        return None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """
        Create LLM settings from environment variables.

        Returns:
            LLMSettings instance
        """
        provider = os.getenv("LLM_PROVIDER", "openai")
        model = os.getenv("LLM_MODEL", "gpt-4")

        # Collect API keys from environment
        api_keys = []
        for key, value in os.environ.items():
            if key.endswith("_API_KEY") or key.endswith("_API_TOKEN"):
                provider_name = key.replace("_API_KEY", "").replace("_API_TOKEN", "").lower()
                if provider_name:
                    api_keys.append(LLMAPIKey(provider=provider_name, key=value))

        # Add explicit key if provided
        if llm_key := os.getenv("LLM_API_KEY"):
            api_keys.append(LLMAPIKey(provider=provider, key=llm_key))

        return cls(
            provider=provider,
            model=model,
            api_keys=api_keys,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        )


@dataclass(frozen=True)
class MonitoringSettings:
    """
    Monitoring and metrics settings.

    Attributes:
        enabled: Whether monitoring is enabled
        prometheus_port: Prometheus metrics port
        metrics_path: Metrics endpoint path
        tracing_enabled: Whether distributed tracing is enabled
    """

    enabled: bool
    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    tracing_enabled: bool = False

    @classmethod
    def from_env(cls) -> "MonitoringSettings":
        """
        Create monitoring settings from environment variables.

        Returns:
            MonitoringSettings instance
        """
        return cls(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            metrics_path=os.getenv("METRICS_PATH", "/metrics"),
            tracing_enabled=os.getenv("TRACING_ENABLED", "false").lower() == "true",
        )


@dataclass(frozen=True)
class SecuritySettings:
    """
    Security configuration settings.

    Attributes:
        jwt_secret_key: Secret key for JWT tokens
        jwt_algorithm: JWT signing algorithm
        access_token_expire_minutes: Access token lifetime
        refresh_token_expire_days: Refresh token lifetime
        enable_https: Whether to require HTTPS
    """

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    enable_https: bool = False

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        """
        Create security settings from environment variables.

        Returns:
            SecuritySettings instance
        """
        import secrets

        jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not jwt_secret:
            jwt_secret = secrets.token_urlsafe(32)
            logger.warning("Using auto-generated JWT secret key. Set JWT_SECRET_KEY in production!")

        return cls(
            jwt_secret_key=jwt_secret,
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            refresh_token_expire_days=int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")),
            enable_https=os.getenv("ENABLE_HTTPS", "false").lower() == "true",
        )


# ============================================================
# Main Settings Class
# ============================================================

@dataclass(frozen=True)
class Settings:
    """
    Main settings class combining all configuration sections.

    Attributes:
        environment: Environment name (dev, staging, prod)
        debug: Whether debug mode is enabled
        database: Database settings
        redis: Redis settings
        api: API settings
        celery: Celery settings
        llm: LLM settings
        monitoring: Monitoring settings
        security: Security settings
        project_root: Path to project root
        data_dir: Path to data directory
        log_dir: Path to log directory
    """

    environment: str
    debug: bool
    database: DatabaseSettings
    redis: RedisSettings
    api: APISettings
    celery: CelerySettings
    llm: LLMSettings
    monitoring: MonitoringSettings
    security: SecuritySettings
    project_root: Path
    data_dir: Path
    log_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Create settings from environment variables.

        Returns:
            Settings instance with all subsections loaded
        """
        environment = os.getenv("ENVIRONMENT", "development")
        debug = os.getenv("DEBUG", "true").lower() == "true"

        # Get project root (assumes we're in the project directory)
        project_root = Path.cwd()
        if not (project_root / "src").exists():
            # Try parent directory
            project_root = Path(__file__).parent.parent

        return cls(
            environment=environment,
            debug=debug,
            database=DatabaseSettings.from_env(),
            redis=RedisSettings.from_env(),
            api=APISettings.from_env(),
            celery=CelerySettings.from_env(),
            llm=LLMSettings.from_env(),
            monitoring=MonitoringSettings.from_env(),
            security=SecuritySettings.from_env(),
            project_root=project_root,
            data_dir=project_root / "data",
            log_dir=project_root / "logs",
        )

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() in ("production", "prod")

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() in ("development", "dev")


# ============================================================
# Settings Singleton
# ============================================================

_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get the application settings singleton.

    Args:
        reload: Force reload settings from environment

    Returns:
        Settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.api.port)
        8000
    """
    global _settings

    if _settings is None or reload:
        _settings = Settings.from_env()
        logger.info(f"Settings loaded (environment: {_settings.environment}, debug: {_settings.debug})")

    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment.

    Returns:
        Newly loaded Settings instance
    """
    return get_settings(reload=True)


# ============================================================
# Environment File Loading
# ============================================================

def load_env_file(path: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.

    This is a simple implementation for development use.
    In production, use proper environment variable management.

    Args:
        path: Path to .env file (default: project_root/.env)
    """
    if path is None:
        settings = get_settings()
        path = str(settings.project_root / ".env")

    env_path = Path(path)
    if not env_path.exists():
        logger.debug(f"No .env file found at {path}")
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            os.environ[key] = value

    logger.info(f"Environment loaded from {path}")
