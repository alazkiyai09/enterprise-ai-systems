# Multi-stage Dockerfile for LLMOps-Eval
#
# Build stage: Install dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Runtime stage: Minimal image
FROM python:3.11-slim

# Set labels
LABEL maintainer="AI Engineer"
LABEL description="LLMOps-Eval: Production-ready LLM evaluation system"
LABEL version="0.1.0"

# Create non-root user
RUN groupadd -r llmops && useradd -r -g llmops llmops

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /install /install

# Make sure scripts in /install are usable
ENV PATH=/install/bin:$PATH

# Copy application code
COPY src/ /app/src/
COPY data/ /app/data/
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/data/datasets \
             /app/data/results \
             /app/data/cache \
             /app/data/models \
             /app/reports \
             /app/prometheus

# Set ownership
RUN chown -R llmops:llmops /app /install

# Switch to non-root user
USER llmops

# Set environment variables
ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/install/bin:$PATH
# Add site-packages to Python path
ENV PYTHONPATH=/install/lib/python3.11/site-packages:/app:/app/src:$PYTHONPATH
# Set cache directory for HuggingFace
ENV TRANSFORMERS_CACHE=/app/data/cache
ENV HF_HOME=/app/data/cache

# Expose ports
EXPOSE 8000 8501

# Health check using Python urllib (available in all Python images)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
