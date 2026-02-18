# Lessons Learned: MultiModal-RAG Project

This document captures issues encountered during development and deployment, along with their solutions.

---

## Docker Deployment Issues

### Issue 1: Missing Module Exports

**Date:** 2026-02-19

**Problem:**
The application failed to start with `ImportError: cannot import name 'create_rag_chain' from 'src.generation'`. Similar errors occurred for other factory functions.

**Root Cause:**
Factory functions like `create_rag_chain`, `create_processor_from_settings`, `create_evaluator`, `create_embedding_service`, `create_hybrid_retriever`, and `CrossEncoderReranker` class were defined in their respective modules but not exported in the `__init__.py` files.

**Solution:**
Added missing exports to module `__init__.py` files:
- `src/generation/__init__.py` - Added `create_rag_chain`
- `src/ingestion/__init__.py` - Added `create_processor_from_settings`
- `src/evaluation/__init__.py` - Added `create_evaluator`
- `src/retrieval/__init__.py` - Added `create_embedding_service`, `create_hybrid_retriever`, `CrossEncoderReranker`

**Lesson:**
Always ensure all public APIs are exported in `__init__.py` files. Consider using `__all__` lists explicitly and verifying imports work from the package level.

---

### Issue 2: Property/Attribute Conflict in EmbeddingService

**Date:** 2026-02-19

**Problem:**
`AttributeError: property 'device' of 'EmbeddingService' object has no setter`

**Root Cause:**
The `EmbeddingService` class had a read-only `@property` for `device` (lines 398-403), but the `__init__` method tried to assign to it with `self.device = device` (line 176).

**Solution:**
Changed `self.device = device` to `self._device_config = device` and updated `_detect_device()` method to use `self._device_config` instead of `self.device`.

**Lesson:**
When using `@property` decorators, ensure instance variables use different names (e.g., `_device_config` vs `device`) to avoid conflicts. Properties without setters are read-only.

---

### Issue 3: Missing Configuration Settings for Remote ChromaDB

**Date:** 2026-02-19

**Problem:**
The application tried to use `PersistentClient` instead of `HttpClient` for ChromaDB, even when `CHROMA_HOST` environment variable was set.

**Root Cause:**
The `Settings` class in `src/config.py` didn't have `CHROMA_HOST` and `CHROMA_PORT` attributes defined. Pydantic's `extra="ignore"` setting caused these environment variables to be silently ignored.

**Solution:**
Added `CHROMA_HOST` and `CHROMA_PORT` fields to the `Settings` class:
```python
CHROMA_HOST: Optional[str] = Field(
    default=None,
    description="ChromaDB server host (None for local persistent mode)",
)
CHROMA_PORT: int = Field(
    default=8000,
    ge=1,
    le=65535,
    description="ChromaDB server port",
)
```

**Lesson:**
When using Pydantic settings with `extra="ignore"`, all environment variables that need to be accessed must be explicitly defined as fields in the settings class.

---

### Issue 4: Incorrect LLM Provider Detection

**Date:** 2026-02-19

**Problem:**
`ValueError: 'gpt' is not a valid LLMProvider`

**Root Cause:**
The code extracted the first part of `LLM_MODEL` (e.g., "gpt" from "gpt-4-turbo") and used it as the provider name. However, `LLMProvider` enum expects "openai", not "gpt".

**Solution:**
Replaced simple string split with proper model-to-provider mapping:
```python
llm_model_lower = settings.LLM_MODEL.lower()
if llm_model_lower.startswith("gpt") or llm_model_lower.startswith("o1") or llm_model_lower.startswith("o3"):
    llm_provider = "openai"
elif llm_model_lower.startswith("claude"):
    llm_provider = "anthropic"
elif llm_model_lower.startswith("glm"):
    llm_provider = "glm"
else:
    llm_provider = "openai"  # default
```

**Lesson:**
Model names don't always map directly to provider names. Create explicit mapping logic based on model name prefixes.

---

### Issue 5: Incorrect Request.scope() Usage in Query Route

**Date:** 2026-02-19

**Problem:**
`AttributeError: type object 'Request' has no attribute 'scope'`

**Root Cause:**
The query route imported `Request` from FastAPI and tried to call `Request.scope()["app"]` on the class itself, rather than on a request instance.

**Solution:**
Import and use the global `rag_chain` variable directly from `src.api.main`:
```python
from src.api.main import rag_chain
```

**Lesson:**
`Request.scope` is an instance attribute, not a class method. When accessing app state from routes, either:
1. Add a `Request` parameter to the function signature and use `request.app`
2. Use global variables initialized during lifespan
3. Use dependency injection

---

### Issue 6: chromadb-client HTTP-Only Mode

**Date:** 2026-02-19

**Problem:**
`RuntimeError: Chroma is running in http-only client mode, and can only be run with 'chromadb.api.fastapi.FastAPI'`

**Root Cause:**
The `requirements.txt` specified `chromadb-client==0.5.23` (HTTP-only client) instead of the full `chromadb` package. This package doesn't support `PersistentClient`.

**Solution:**
The code already had logic to use `HttpClient` when `CHROMA_HOST` is set. After fixing Issue #3 (adding settings), this worked correctly.

**Lesson:**
When using `chromadb-client` (HTTP-only), always connect to a remote ChromaDB server. For local/embedded ChromaDB, use the full `chromadb` package instead.

---

## ChromaDB Container Issues

### Issue 7: ChromaDB Health Check Timing

**Date:** 2026-02-19

**Problem:**
`dependency failed to start: container multimodal-rag-chromadb is unhealthy`

**Root Cause:**
The API container tried to start before ChromaDB was fully healthy. The health check interval was too aggressive.

**Solution:**
Increased `start_period` in docker-compose health check to 40s to give ChromaDB more time to initialize.

**Lesson:**
Database containers often need more time to initialize. Use appropriate `start_period` values in health checks, especially for containers that perform schema migrations or data initialization on startup.

---

## Best Practices Identified

### 1. Module Exports
- Always define `__all__` in `__init__.py` files
- Verify imports work at package level, not just module level
- Use explicit exports rather than wildcard imports

### 2. Pydantic Settings
- Define all environment variables as fields in Settings class
- Be aware that `extra="ignore"` silently drops undefined variables
- Use `Field()` with descriptions for documentation

### 3. Property Decorators
- Use underscore prefix for internal attributes (`_device` vs `device`)
- Remember that properties without setters are read-only
- Consider using `@property.setter` if mutation is needed

### 4. Docker Compose
- Set appropriate health check intervals and start periods
- Use `depends_on` with `condition: service_healthy` for dependencies
- Configure DNS servers (`8.8.8.8`) to avoid resolution issues

### 5. Error Messages
- Include enough context in error messages to diagnose issues
- Log the actual values being used (e.g., which provider/model was selected)

---

## References

- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [ChromaDB HTTP Client Guide](https://docs.trychroma.com/guides#using-the-python-http-only-client)
- [FastAPI Application State](https://fastapi.tiangolo.com/advanced/using-request-directly/)
