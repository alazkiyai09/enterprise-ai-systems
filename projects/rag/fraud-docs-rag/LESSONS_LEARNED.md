# Lessons Learned: FraudDocs-RAG Project

This document captures issues encountered during development and deployment, along with their solutions.

---

## Docker Deployment Issues

### Issue 1: Curl Not Available in Slim Images

**Date:** 2026-02-19

**Problem:**
Docker health checks using `curl` failed because curl is not available in Python slim images.

**Root Cause:**
The docker-compose.yml used `curl` for health checks:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

**Solution:**
Use bash's `/dev/tcp` or Python's `urllib` instead:
```yaml
# For services with bash
healthcheck:
  test: ["CMD-SHELL", "bash -c 'echo > /dev/tcp/localhost/8000' || exit 1"]

# For Python services
healthcheck:
  test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"]
```

**Lesson:**
Avoid using curl in health checks for slim Docker images. Use built-in tools like bash `/dev/tcp` or Python's urllib.

---

### Issue 2: Network Name Typo

**Date:** 2026-02-19

**Problem:**
`docker-compose up` failed with network-related errors.

**Root Cause:**
Network name had a typo with a space: `fr auddocs-network` instead of `frauddocs-network`.

**Solution:**
Fixed the typo in docker-compose.yml:
```yaml
networks:
  frauddocs-network:
    driver: bridge
```

**Lesson:**
Always validate docker-compose.yml syntax with `docker-compose config` before deployment.

---

### Issue 3: Missing DNS Configuration

**Date:** 2026-02-19

**Problem:**
Containers couldn't resolve external hostnames like `huggingface.co`.

**Root Cause:**
Docker containers use the host's DNS by default, which may not be configured correctly in all environments.

**Solution:**
Add explicit DNS configuration in docker-compose.yml:
```yaml
services:
  api:
    dns:
      - 8.8.8.8
      - 8.8.4.4
```

**Lesson:**
Configure DNS explicitly for containers that need external connectivity, especially for downloading models or calling external APIs.

---

### Issue 4: PyTorch GPU vs CPU Image Size

**Date:** 2026-02-19

**Problem:**
Docker image was too large (~4GB+) due to full PyTorch with CUDA support.

**Root Cause:**
Default PyTorch installation includes GPU (CUDA) support, which adds significant size.

**Solution:**
Use CPU-only PyTorch by specifying the extra index URL:
```dockerfile
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

**Result:**
Image size reduced from ~4GB to 2.72GB.

**Lesson:**
For CPU-only deployments, use PyTorch's CPU-only wheel to significantly reduce image size.

---

### Issue 5: HuggingFace Cache Directory Permissions

**Date:** 2026-02-19

**Problem:**
`PermissionError: [Errno 13] Permission denied` when trying to cache HuggingFace models.

**Root Cause:**
The HuggingFace cache directory wasn't created with proper ownership for the non-root user.

**Solution:**
Create the cache directory with proper ownership in Dockerfile:
```dockerfile
ENV TRANSFORMERS_CACHE=/app/data/cache \
    HF_HOME=/app/data/cache \
    SENTENCE_TRANSFORMERS_HOME=/app/data/cache

RUN mkdir -p /app/data/cache && \
    chown -R ${APP_USER}:${APP_USER} /app/data/cache
```

**Lesson:**
Always create and set permissions for cache directories when running as non-root user.

---

### Issue 6: Volume Mount Overriding Directory Permissions

**Date:** 2026-02-19

**Problem:**
Container failed to write to `/home/appuser/.cache` even after setting permissions in Dockerfile.

**Root Cause:**
A named volume was mounted at `/home/appuser/.cache`, which overrode the permissions set in the Dockerfile.

**Solution:**
Either:
1. Use a different cache path that isn't volume-mounted
2. Don't mount a volume at the cache location
3. Use an init container to fix permissions

**Lesson:**
Volume mounts override directory permissions from the Dockerfile. Plan volume mount locations carefully.

---

## LLM Integration Issues

### Issue 7: Llama-Index Model Name Validation

**Date:** 2026-02-19

**Problem:**
`ValueError: Unknown model 'glm-4-flash'. Please provide a valid OpenAI model name`

**Root Cause:**
Llama-index's `OpenAI` class validates model names against a hardcoded list of OpenAI models. Non-OpenAI models are rejected even when using a custom `base_url`.

**Solution:**
Create a custom LLM wrapper that bypasses the validation:
```python
from llama_index.core.llms.custom import CustomLLM

class OpenAICompatibleLLM(CustomLLM):
    model: str = "glm-4-flash"
    api_key: str = ""
    base_url: str = ""
    # ... implement chat(), complete(), etc.
```

**Lesson:**
When using OpenAI-compatible APIs with non-OpenAI models, use `CustomLLM` to bypass model name validation.

---

### Issue 8: Anthropic vs OpenAI API Format

**Date:** 2026-02-19

**Problem:**
API calls failed with `'NoneType' object is not subscriptable` when using z.ai endpoint.

**Root Cause:**
The z.ai API uses Anthropic's message format, not OpenAI's. Response structures are different:
- OpenAI: `response.choices[0].message.content`
- Anthropic: `response.content[0].text`

**Solution:**
Detect the API type and use the appropriate client:
```python
if "anthropic" in self.base_url or "z.ai" in self.base_url:
    import anthropic
    self._client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
else:
    from openai import OpenAI
    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
```

**Lesson:**
"OpenAI-compatible" APIs may use different message formats. Check the actual API documentation and use the appropriate SDK.

---

### Issue 9: GLM Model Names

**Date:** 2026-02-19

**Problem:**
`Error code: 400 - {'error': {'code': '1211', 'message': '模型不存在，请检查模型代码。'}}` (Model doesn't exist)

**Root Cause:**
Used incorrect model names like `glm-4-flash` or `glm-4-plus` which don't exist in the z.ai API.

**Solution:**
Use the correct model name for the API endpoint:
- z.ai API: `glm-5`
- ZhipuAI native API: `glm-4`, `glm-4-plus`, `glm-4-flash`, etc.

**Lesson:**
Different API endpoints for the same provider may support different model names. Always check the specific endpoint's documentation.

---

## Code Issues

### Issue 10: Slowapi Rate Limiter Parameter Order

**Date:** 2026-02-19

**Problem:**
`TypeError: parameter 'request' must be an instance of starlette.requests.Request`

**Root Cause:**
Slowapi's `@limiter.limit()` decorator requires the `Request` parameter to be:
1. Named exactly `request`
2. The first positional parameter in the function

**Solution:**
Ensure the request parameter is first and named correctly:
```python
# Before
@limiter.limit("30/minute")
async def query_knowledge_base(query_request: QueryRequest, http_request: Request):

# After
@limiter.limit("30/minute")
async def query_knowledge_base(request: Request, query_request: QueryRequest):
```

**Lesson:**
Slowapi requires the `Request` object as the first parameter named `request`. Check decorator requirements for parameter ordering.

---

### Issue 11: Pydantic Score Validation

**Date:** 2026-02-19

**Problem:**
`ValidationError: Input should be less than or equal to 1 [type=less_than_equal, input_value=7.954521179199219]`

**Root Cause:**
The `Source` model had score validation `le=1.0`, but ChromaDB returns distance scores which can be any value (higher = more different).

**Solution:**
Remove the constraint or normalize scores:
```python
# Before
score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)

# After
score: float = Field(..., description="Relevance score")
```

**Lesson:**
Different vector databases return scores in different formats (similarity vs distance). Don't assume scores are normalized to 0-1.

---

### Issue 12: Shared Module Symlink in Docker

**Date:** 2026-02-19

**Problem:**
`ModuleNotFoundError: No module named 'shared'` inside Docker container.

**Root Cause:**
A symlink created on the host doesn't work inside the container because the target path doesn't exist in the container's filesystem.

**Solution:**
Mount the shared modules as a volume in docker-compose.yml:
```yaml
volumes:
  - /path/to/shared:/app/src/shared:ro
```

**Lesson:**
Symlinks pointing outside the build context won't work in Docker. Use volume mounts instead for external dependencies.

---

## Best Practices Identified

### 1. Docker Health Checks
- Use bash `/dev/tcp` or Python urllib instead of curl
- Set appropriate `start_period` for slow-starting services
- Use `condition: service_healthy` for dependencies

### 2. LLM Integration
- Use `CustomLLM` for non-OpenAI models with llama-index
- Check API format (OpenAI vs Anthropic) before choosing SDK
- Verify model names against the specific API endpoint

### 3. Docker Compose
- Always run `docker-compose config` to validate syntax
- Configure DNS for external connectivity
- Use CPU-only PyTorch for smaller images
- Mount external dependencies as volumes, not symlinks

### 4. Pydantic Models
- Be careful with validation constraints on externally-sourced data
- Consider the data source format (distance vs similarity scores)

---

## References

- [Llama-Index CustomLLM Documentation](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
- [Docker Health Check Best Practices](https://docs.docker.com/engine/reference/builder/#healthcheck)
- [PyTorch CPU-only Installation](https://pytorch.org/get-started/locally/#cpu-version)
