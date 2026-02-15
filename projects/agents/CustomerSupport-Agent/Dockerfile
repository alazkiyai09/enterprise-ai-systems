# Dockerfile for CustomerSupport-Agent
# Multi-stage build for optimized production image with CPU-only PyTorch

# Builder stage
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all dependencies with CPU-only PyTorch index
# The --extra-index-url ensures torch is resolved from PyTorch CPU index
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Download NLP data for TextBlob
RUN python -c "from textblob import TextBlob; TextBlob('test')" || \
    python -m textblob.download_corpora || echo "TextBlob corpora download skipped"

# Runtime stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p data/knowledge_base data/chroma_db data/user_memory && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
