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

### Issue 8: Request.scope() Bug in Multiple Route Files

**Date:** 2026-02-19

**Problem:**
Multiple API routes failed with `AttributeError: type object 'Request' has no attribute 'scope'` during document ingestion, queries, and evaluation.

**Root Cause:**
The same `Request.scope()["app"]` pattern was used in multiple route files (documents.py, query.py, evaluation.py), affecting nearly all endpoints.

**Solution:**
Replaced all occurrences with direct imports from the main module:
```python
from src.api.main import rag_chain, document_processor, vector_store, rag_evaluator
```

**Files Fixed:**
- `src/api/routes/documents.py` - 3 occurrences
- `src/api/routes/query.py` - 3 occurrences
- `src/api/routes/evaluation.py` - 3 occurrences

**Lesson:**
When a pattern is repeated across multiple files, fix all occurrences systematically. Use grep to find all instances: `grep -r "Request.scope()" src/`

---

### Issue 9: Reserved LogRecord Attribute in Logging

**Date:** 2026-02-19

**Problem:**
`KeyError: "Attempt to overwrite 'filename' in LogRecord"`

**Root Cause:**
Python's logging module reserves certain attribute names (like `filename`, `lineno`, `levelname`) in LogRecord. Using these as keys in the `extra` parameter causes a conflict.

**Solution:**
Renamed the reserved attribute to avoid conflict:
```python
# Before
logger.info(..., extra={"filename": filename, ...})

# After
logger.info(..., extra={"doc_filename": filename, ...})
```

**Lesson:**
Avoid using reserved LogRecord attribute names in logging `extra` dicts. Common reserved names include: `filename`, `lineno`, `levelname`, `message`, `name`, `pathname`, `process`, `thread`.

---

### Issue 10: Lazy-Loaded Model Not Accessed via Property

**Date:** 2026-02-19

**Problem:**
`AttributeError: 'NoneType' object has no attribute 'encode'` during embedding generation.

**Root Cause:**
The `EmbeddingService` class uses lazy loading with a `model` property, but the `embed_texts` method directly accessed `self._model` instead of `self.model`, bypassing the lazy-loading mechanism.

**Solution:**
Changed direct attribute access to use the property:
```python
# Before
new_embeddings = self._model.encode(...)

# After
new_embeddings = self.model.encode(...)  # Triggers lazy loading
```

**Lesson:**
When implementing lazy loading with properties, always use the property accessor, not the underlying private attribute. The property ensures the resource is loaded before use.

---

### Issue 11: CrossEncoder Missing device Attribute

**Date:** 2026-02-19

**Problem:**
`AttributeError: 'CrossEncoder' object has no attribute 'device'`

**Root Cause:**
The reranker code tried to log the model's device with `self._model.device`, but the `CrossEncoder` class from sentence-transformers doesn't expose a `device` attribute directly.

**Solution:**
Use the configured device value instead of trying to read it from the model:
```python
# Before
"device": str(self._model.device),

# After
"device": str(self.device),  # Use configured device
```

**Lesson:**
Don't assume third-party library classes expose internal attributes. Check the library's API documentation or use values you control.

---

### Issue 12: File Extension Mismatch in Validation

**Date:** 2026-02-19

**Problem:**
Document upload failed with "Unsupported file type: .txt" even though "txt" was in the supported formats list.

**Root Cause:**
`Path.suffix` returns the extension with a dot (`.txt`), but the supported formats list contained extensions without dots (`['pdf', 'docx', 'txt']`).

**Solution:**
Strip the leading dot from the file extension before comparison:
```python
file_ext = Path(file.filename).suffix.lower().lstrip('.')
```

**Lesson:**
Be aware of the exact format returned by path manipulation functions. `Path.suffix` includes the dot; `Path.stem` does not include the extension.

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

## GLM-5 Integration Issues

### Issue 13: GLM API Model Name Mismatch

**Date:** 2026-02-19

**Problem:**
`Error code: 400 - {'error': {'code': '1211', 'message': 'Unknown Model, please check the model code.'}}`

**Root Cause:**
Used incorrect model name `glm-5-flash` which doesn't exist in GLM API. Different APIs use different model naming conventions.

**Solution:**
Use the correct model name `glm-5` as specified in the GLM documentation.

**Lesson:**
Always verify model names against the API documentation. Model names vary between providers and even between different endpoints of the same provider.

---

### Issue 14: GLM API Rate Limiting (Insufficient Balance)

**Date:** 2026-02-19

**Problem:**
`Error code: 429 - {'error': {'code': '1113', 'message': 'Insufficient balance or no resource package. Please recharge.'}}`

**Root Cause:**
The GLM API key had insufficient balance to make requests.

**Solution:**
Recharge the API key balance. This is expected behavior for paid APIs.

**Lesson:**
Always ensure API keys have sufficient balance/credits before testing. Monitor API usage and set up alerts for low balance.

---

### Issue 15: GLM API Integration Approaches

**Date:** 2026-02-19

**Problem:**
Multiple integration approaches exist for GLM API, causing confusion about which to use.

**Root Cause:**
GLM API supports multiple access methods:
1. LangChain `ChatZhipuAI` - requires `pyjwt` dependency
2. Anthropic-compatible API via `anthropic` SDK
3. Native ZhipuAI SDK

**Solution:**
Use the Anthropic-compatible API with `anthropic` SDK for simplicity:
```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-api-key",
    base_url="https://api.z.ai/api/anthropic",  # International
)

response = client.messages.create(
    model="glm-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

**Lesson:**
When a provider offers multiple SDK options, choose the one that:
1. Uses familiar patterns (Anthropic SDK is widely known)
2. Has fewer dependencies
3. Has better documentation

---

### Issue 16: Missing pyjwt Dependency for ChatZhipuAI

**Date:** 2026-02-19

**Problem:**
`ImportError: jwt package not found, please install it with pip install pyjwt`

**Root Cause:**
The `langchain_community.chat_models.ChatZhipuAI` requires `pyjwt` for JWT-based authentication with ZhipuAI's native API, but it wasn't listed in requirements.txt.

**Solution:**
Either:
1. Add `pyjwt` to requirements.txt if using ChatZhipuAI
2. Use Anthropic-compatible API instead (no pyjwt needed)

**Lesson:**
Third-party library wrappers often have hidden dependencies. Check the library's documentation for all required packages, or use the native SDK when available.

---

### Issue 17: CLIP vs Text Embedding Dimension Mismatch

**Date:** 2026-02-19

**Problem:**
`ValueError: operands could not be broadcast together with shapes (384,) (512,)`

**Root Cause:**
The multimodal retriever tried to combine CLIP image embeddings (512 dimensions) with text embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions):
```python
combined = (text_embedding + image_embedding) / 2  # Fails: different shapes
```

**Solution:**
Disable CLIP and use text-based image search only:
```python
retriever = create_retriever(enable_clip=False)
```

**Proper Solutions (TODO):**
1. Use CLIP for both text and image embeddings (consistent 512d)
2. Project both embeddings to a common dimension
3. Use separate indices for different modalities

**Lesson:**
When combining embeddings from different models, always verify dimensions match. Different embedding models produce different vector sizes:
- `all-MiniLM-L6-v2`: 384 dimensions
- `all-mpnet-base-v2`: 768 dimensions
- CLIP `ViT-B/32`: 512 dimensions

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
