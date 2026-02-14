# Enterprise-RAG Deployment Guide

This guide covers production deployment of the Enterprise-RAG system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Security Checklist](#security-checklist)
5. [Configuration](#configuration)
6. [Monitoring & Health Checks](#monitoring--health-checks)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key (or Anthropic API key)
- At least 4GB RAM for the container

---

## Quick Start

### 1. Clone and Configure

```bash
cd Enterprise-RAG
cp .env.example .env
```

### 2. Set Required Environment Variables

Edit `.env` and set at minimum:

```bash
OPENAI_API_KEY=sk-your-actual-api-key
API_KEYS=your-secure-api-key-1,your-secure-api-key-2
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### 3. Run with Docker Compose

```bash
docker-compose up -d
```

### 4. Verify Deployment

```bash
curl http://localhost:8000/health
```

---

## Production Deployment

### Option 1: Docker Compose (Recommended for Single Server)

```bash
# Set production environment
export ENVIRONMENT=production

# Generate secure secrets
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Start services
docker-compose -f docker-compose.yml up -d
```

### Option 2: Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ENVIRONMENT=production
export OPENAI_API_KEY=your-key
export API_KEYS=your-api-keys

# Run with Gunicorn
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Option 3: Kubernetes (for Scale)

See `k8s/` directory for Kubernetes manifests (if available).

---

## Security Checklist

Before deploying to production, ensure ALL of the following:

### Critical (Must Do)

- [ ] **Change SECRET_KEY** - Generate a secure random key:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

- [ ] **Set API_KEYS** - Configure authentication:
  ```bash
  API_KEYS=key-$(openssl rand -hex 32)
  ```

- [ ] **Set ENVIRONMENT=production** - Enables security validation

- [ ] **Use HTTPS** - Configure SSL/TLS certificates

- [ ] **Review CORS_ORIGINS** - Only allow your actual frontend domains

### Important (Should Do)

- [ ] **Set up rate limiting** - Already enabled, review limits in `src/api/rate_limit.py`

- [ ] **Enable logging** - Set LOG_LEVEL=INFO or WARNING

- [ ] **Configure backup** - Back up ChromaDB data directory

- [ ] **Set resource limits** - Already configured in docker-compose.yml

### Recommended

- [ ] **Use a reverse proxy** - nginx, Traefik, or cloud load balancer

- [ ] **Enable monitoring** - Set up health check alerts

- [ ] **Review file upload limits** - MAX_FILE_SIZE setting

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `API_KEYS` | Recommended | - | Comma-separated API keys |
| `SECRET_KEY` | Yes | - | Secret for JWT tokens |
| `ENVIRONMENT` | Yes | development | Environment mode |
| `VECTOR_STORE_TYPE` | No | chroma | Vector database type |
| `LLM_MODEL` | No | gpt-4-turbo | LLM model to use |
| `EMBEDDING_MODEL` | No | all-MiniLM-L6-v2 | Embedding model |
| `LOG_LEVEL` | No | INFO | Logging level |
| `CORS_ORIGINS` | No | localhost:8501 | Allowed CORS origins |

*Required for most functionality

### Vector Store Options

1. **ChromaDB** (Default, Recommended for <1M vectors)
   ```bash
   VECTOR_STORE_TYPE=chroma
   CHROMA_PATH=./data/chroma
   ```

2. **Qdrant** (For larger scale)
   ```bash
   VECTOR_STORE_TYPE=qdrant
   QDRANT_HOST=qdrant
   QDRANT_PORT=6333
   ```

### LLM Providers

1. **OpenAI** (Default)
   ```bash
   LLM_MODEL=gpt-4-turbo
   OPENAI_API_KEY=sk-...
   ```

2. **Anthropic Claude**
   ```bash
   LLM_MODEL=claude-3-opus-20240229
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **GLM (Zhipu AI)**
   ```bash
   LLM_MODEL=glm-4
   ZHIPU_API_KEY=...
   ```

---

## Monitoring & Health Checks

### Health Endpoints

- `GET /health` - Component health status
- `GET /stats` - System statistics
- `GET /` - API info and available endpoints

### Example Health Check

```bash
curl http://localhost:8000/health | jq
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "components": {
    "api": "healthy",
    "rag_chain": "healthy",
    "vector_store": "healthy (100 chunks)",
    "embedding_service": "healthy"
  }
}
```

### Docker Health Checks

Docker Compose includes health checks that verify:
- API responds to `/health` endpoint
- Qdrant responds to `/health` endpoint (if using Qdrant)

---

## Troubleshooting

### Common Issues

#### 1. Import Errors on Startup

**Symptom**: `ImportError: cannot import name 'BM25Retriever'`

**Solution**: Ensure you have the latest code with all exports in `src/retrieval/__init__.py`.

#### 2. Authentication Fails

**Symptom**: `401 Unauthorized` with valid API key

**Solution**:
- Check `API_KEYS` is set correctly (comma-separated, no spaces)
- Verify you're sending `X-API-Key` header (not `Authorization`)
- Check logs for authentication errors

#### 3. Vector Store Errors

**Symptom**: `Vector store not initialized`

**Solution**:
- Check `VECTOR_STORE_TYPE` is set to `chroma` or `qdrant`
- For Qdrant, ensure the Qdrant container is running
- Check data directory permissions

#### 4. Rate Limiting Issues

**Symptom**: `429 Too Many Requests`

**Solution**:
- Reduce request frequency
- Check rate limit settings in `src/api/rate_limit.py`
- Ensure X-Forwarded-For header is set correctly behind proxy

#### 5. Memory Issues

**Symptom**: Container crashes, OOM errors

**Solution**:
- Increase Docker memory limit in docker-compose.yml
- Reduce `EMBEDDING_BATCH_SIZE`
- Enable `ENABLE_GPU=false` if no GPU available

### Logs

View container logs:
```bash
docker-compose logs -f rag-api
```

View application logs:
```bash
tail -f logs/enterprise_rag.log
```

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=DEBUG
```

---

## Upgrading

### Backup Before Upgrade

```bash
# Backup ChromaDB data
cp -r data/chroma data/chroma_backup

# Backup environment
cp .env .env.backup
```

### Upgrade Steps

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Restart with new code
docker-compose down && docker-compose up -d
```

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs for error messages
3. Open an issue with reproduction steps
