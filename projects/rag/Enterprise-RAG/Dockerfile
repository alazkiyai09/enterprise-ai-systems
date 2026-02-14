# ============================================================
# Enterprise-RAG: Production Dockerfile
# ============================================================
# Multi-stage Dockerfile for production deployment

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
# Use extra-index-url to get torch from CPU index
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt \
    && find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type f -name "*.pyc" -delete 2>/dev/null || true


# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.11/site-packages:/app/src

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder
COPY --from=builder /install /install

# Copy application code
COPY ./src /app/src
COPY ./pyproject.toml /app/
COPY README.md /app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/documents /app/data/chroma /app/logs && \
    chmod 755 /app/data /app/data/documents /app/data/chroma /app/logs

# Create non-root user for running the application
RUN groupadd -r appuser && \
    useradd -m -u 1000 -r -g appuser appuser && \
    chown -R appuser:appuser /app /install

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
