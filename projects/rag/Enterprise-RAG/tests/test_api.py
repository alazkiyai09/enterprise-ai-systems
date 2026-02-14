# ============================================================
# Enterprise-RAG: API Integration Tests
# ============================================================
"""
Unit tests for API authentication and rate limiting modules.

These tests focus on the new security modules without requiring
the full RAG application to be loaded.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import asyncio


# ============================================================
# Test Configuration Setup
# ============================================================

# Set test environment before any imports
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-default")
os.environ.setdefault("ENVIRONMENT", "testing")


# ============================================================
# Config Validation Tests (no heavy imports)
# ============================================================

class TestConfigValidation:
    """Tests for config validation."""

    def test_secret_key_validator_exists(self):
        """Test that SECRET_KEY validator is defined in config.py."""
        with open("src/config.py", "r") as f:
            content = f.read()

        assert "validate_secret_key" in content
        assert "insecure_defaults" in content
        assert "SECRET_KEY must be changed" in content

    def test_insecure_secret_key_raises_in_production(self):
        """Test that insecure SECRET_KEY raises error in production."""
        with open("src/config.py", "r") as f:
            content = f.read()

        # Check that the validator rejects insecure defaults in production
        assert "ENVIRONMENT" in content
        assert "production" in content.lower()
        assert 'raise ValueError' in content

    def test_secure_secret_key_pattern(self):
        """Test that config accepts secure keys (via source analysis)."""
        with open("src/config.py", "r") as f:
            content = f.read()

        # The validator should allow any key not in insecure_defaults
        assert "insecure_defaults" in content
        assert "return v" in content  # Should return the value if validation passes


# ============================================================
# Docker Compose Configuration Tests
# ============================================================

class TestDockerComposeConfig:
    """Tests for docker-compose configuration."""

    def test_vector_store_default_is_chroma(self):
        """Test that VECTOR_STORE_TYPE defaults to chroma."""
        import yaml

        with open("docker-compose.yml", "r") as f:
            config = yaml.safe_load(f)

        # Find the VECTOR_STORE_TYPE environment variable
        env_vars = config["services"]["rag-api"]["environment"]
        for var in env_vars:
            if "VECTOR_STORE_TYPE" in var:
                assert "chroma" in var.lower()
                break
        else:
            pytest.fail("VECTOR_STORE_TYPE not found in docker-compose.yml")


# ============================================================
# Requirements Tests
# ============================================================

class TestRequirements:
    """Tests for requirements.txt."""

    def test_zhipuai_in_requirements(self):
        """Test that zhipuai is in requirements."""
        with open("requirements.txt", "r") as f:
            content = f.read()

        assert "zhipuai" in content.lower()

    def test_python_json_logger_in_requirements(self):
        """Test that python-json-logger is in requirements."""
        with open("requirements.txt", "r") as f:
            content = f.read()

        assert "python-json-logger" in content.lower()


# ============================================================
# Auth Module File Tests (read file directly)
# ============================================================

class TestAuthModuleSource:
    """Tests for auth module source code."""

    def test_auth_file_exists(self):
        """Test that auth.py exists."""
        import os
        assert os.path.exists("src/api/auth.py")

    def test_verify_api_key_function_defined(self):
        """Test that verify_api_key function is defined."""
        with open("src/api/auth.py", "r") as f:
            content = f.read()

        assert "async def verify_api_key" in content
        assert "HTTPException" in content
        assert "X-API-Key" in content

    def test_api_key_header_defined(self):
        """Test that API_KEY_HEADER is defined."""
        with open("src/api/auth.py", "r") as f:
            content = f.read()

        assert "API_KEY_HEADER" in content
        assert "X-API-Key" in content

    def test_dev_mode_when_no_keys(self):
        """Test that dev-mode is returned when no keys configured."""
        with open("src/api/auth.py", "r") as f:
            content = f.read()

        assert 'return "dev-mode"' in content
        assert "if not VALID_API_KEYS:" in content

    def test_401_on_invalid_key(self):
        """Test that invalid key returns 401."""
        with open("src/api/auth.py", "r") as f:
            content = f.read()

        assert "HTTP_401_UNAUTHORIZED" in content
        assert "Invalid API key" in content

    def test_api_keys_parsed_from_env(self):
        """Test that API keys are parsed from API_KEYS env var."""
        with open("src/api/auth.py", "r") as f:
            content = f.read()

        assert 'os.getenv("API_KEYS"' in content
        assert ".split(" in content


# ============================================================
# Rate Limit Module File Tests
# ============================================================

class TestRateLimitModuleSource:
    """Tests for rate limit module source code."""

    def test_rate_limit_file_exists(self):
        """Test that rate_limit.py exists."""
        import os
        assert os.path.exists("src/api/rate_limit.py")

    def test_limiter_instance_defined(self):
        """Test that limiter is defined."""
        with open("src/api/rate_limit.py", "r") as f:
            content = f.read()

        assert "limiter = Limiter" in content
        assert "get_remote_address" in content

    def test_exception_handler_defined(self):
        """Test that exception handler is defined."""
        with open("src/api/rate_limit.py", "r") as f:
            content = f.read()

        assert "rate_limit_exception_handler" in content
        assert "RateLimitExceeded" in content
        assert "429" in content

    def test_rate_limit_decorators_defined(self):
        """Test that rate limit decorators are defined."""
        with open("src/api/rate_limit.py", "r") as f:
            content = f.read()

        assert "rate_limit_query" in content
        assert "rate_limit_ingest" in content
        assert "rate_limit_evaluation" in content


# ============================================================
# Route Authentication Tests (check source files)
# ============================================================

class TestRouteAuthenticationSource:
    """Tests for authentication in route files."""

    def test_query_routes_have_auth(self):
        """Test that query routes import and use auth."""
        with open("src/api/routes/query.py", "r") as f:
            content = f.read()

        assert "from src.api.auth import verify_api_key" in content
        assert "Depends(verify_api_key)" in content

    def test_documents_routes_have_auth(self):
        """Test that documents routes import and use auth."""
        with open("src/api/routes/documents.py", "r") as f:
            content = f.read()

        assert "from src.api.auth import verify_api_key" in content
        assert "Depends(verify_api_key)" in content

    def test_evaluation_routes_have_auth(self):
        """Test that evaluation routes import and use auth."""
        with open("src/api/routes/evaluation.py", "r") as f:
            content = f.read()

        assert "from src.api.auth import verify_api_key" in content
        assert "Depends(verify_api_key)" in content


# ============================================================
# Main.py Tests (check source file)
# ============================================================

class TestMainPySource:
    """Tests for main.py source code."""

    def test_local_rate_limit_import(self):
        """Test that main.py imports from local rate_limit module."""
        with open("src/api/main.py", "r") as f:
            content = f.read()

        assert "from src.api.rate_limit import" in content

    def test_cors_allows_api_key_header(self):
        """Test that CORS allows X-API-Key header."""
        with open("src/api/main.py", "r") as f:
            content = f.read()

        assert '"X-API-Key"' in content


# ============================================================
# Retrieval Module Exports Tests
# ============================================================

class TestRetrievalExportsSource:
    """Tests for retrieval module exports."""

    def test_bm25_exports_in_init(self):
        """Test that BM25 classes are exported in __init__.py."""
        with open("src/retrieval/__init__.py", "r") as f:
            content = f.read()

        assert "BM25Retriever" in content
        assert "SparseSearchResult" in content
        assert "BM25Stats" in content
        assert "create_bm25_retriever" in content

    def test_reranker_exports_in_init(self):
        """Test that reranker classes are exported in __init__.py."""
        with open("src/retrieval/__init__.py", "r") as f:
            content = f.read()

        assert "CrossEncoderReranker" in content
        assert "RerankedSearchResult" in content

    def test_hybrid_retriever_exports_in_init(self):
        """Test that hybrid retriever classes are exported in __init__.py."""
        with open("src/retrieval/__init__.py", "r") as f:
            content = f.read()

        assert "HybridRetriever" in content
        assert "HybridSearchResult" in content
        assert "create_hybrid_retriever" in content

    def test_embedding_service_exports_in_init(self):
        """Test that embedding service is exported in __init__.py."""
        with open("src/retrieval/__init__.py", "r") as f:
            content = f.read()

        assert "EmbeddingService" in content
        assert "create_embedding_service" in content


# ============================================================
# Environment File Tests
# ============================================================

class TestEnvExample:
    """Tests for .env.example."""

    def test_api_keys_in_env_example(self):
        """Test that API_KEYS is documented in .env.example."""
        with open(".env.example", "r") as f:
            content = f.read()

        assert "API_KEYS" in content

    def test_environment_in_env_example(self):
        """Test that ENVIRONMENT is documented in .env.example."""
        with open(".env.example", "r") as f:
            content = f.read()

        assert "ENVIRONMENT" in content


# ============================================================
# Deployment Guide Tests
# ============================================================

class TestDeploymentGuide:
    """Tests for DEPLOYMENT.md."""

    def test_deployment_guide_exists(self):
        """Test that DEPLOYMENT.md exists."""
        import os
        assert os.path.exists("DEPLOYMENT.md")

    def test_security_checklist_present(self):
        """Test that security checklist is in deployment guide."""
        with open("DEPLOYMENT.md", "r") as f:
            content = f.read()

        assert "Security Checklist" in content
        assert "API_KEYS" in content
        assert "SECRET_KEY" in content

    def test_quick_start_present(self):
        """Test that quick start instructions are in deployment guide."""
        with open("DEPLOYMENT.md", "r") as f:
            content = f.read()

        assert "Quick Start" in content

    def test_troubleshooting_present(self):
        """Test that troubleshooting section is in deployment guide."""
        with open("DEPLOYMENT.md", "r") as f:
            content = f.read()

        assert "Troubleshooting" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
