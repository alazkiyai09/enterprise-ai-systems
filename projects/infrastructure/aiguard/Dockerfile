# ============================================================
# AIGuard Dockerfile - Multi-stage Build
# ============================================================
# Production-ready Docker image for AIGuard security service
# Optimizations:
# - Multi-stage build for minimal image size
# - CPU-only PyTorch for smaller footprint
# - Non-root user for security
# - Health checks using Python urllib (no curl dependency)
# ============================================================

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libmagic1 \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements-fixed.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set HuggingFace cache directory
ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface

# Download spaCy model (en_core_web_sm for NER)
RUN python -m spacy download en_core_web_sm --quiet

# Pre-download sentence-transformers model for injection detection
# Disable Xet storage to use regular HTTP download
ENV HF_HUB_DISABLE_XET=1
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ============================================================
# Stage 2: Production Runtime
# ============================================================
FROM python:3.11-slim-bookworm AS production

# Labels
LABEL maintainer="AIGuard Team"
LABEL description="AIGuard Security Guardrails API"
LABEL version="1.0.0"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 aiguard && \
    useradd --uid 1000 --gid aiguard --shell /bin/bash --create-home aiguard

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy spaCy models from builder
COPY --from=builder /opt/venv/lib/python3.11/site-packages/en_core_web_sm /opt/venv/lib/python3.11/site-packages/en_core_web_sm

# Copy HuggingFace model cache from builder
COPY --from=builder /opt/huggingface /app/cache
RUN chown -R aiguard:aiguard /app/cache

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aiguard:aiguard . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R aiguard:aiguard /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    # HuggingFace cache directory
    HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    # Disable telemetry
    HF_HUB_DISABLE_TELEMETRY=1 \
    # Application settings
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

# Switch to non-root user
USER aiguard

# Expose port
EXPOSE 8000

# Health check using Python urllib (not curl - not available in slim images)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
