"""
RAG Generation Chain for FraudDocs-RAG.

This module implements the generation component of the RAG pipeline, including:
- Multi-provider LLM support (Ollama, GLM-4, OpenAI)
- Financial domain-specific system prompting
- Response generation with source citations
- Context-aware question answering
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.custom import CustomLLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """
    Environment types for LLM selection.

    Attributes:
        DEVELOPMENT: Local development with Ollama
        DEMO: Demo environment with GLM-4
        PRODUCTION: Production with OpenAI
    """

    DEVELOPMENT = "development"
    DEMO = "demo"
    PRODUCTION = "production"


# Financial domain-specific system prompt
FINANCIAL_COMPLIANCE_SYSTEM_PROMPT = """You are a knowledgeable financial compliance and fraud detection expert specializing in anti-money laundering (AML), know-your-customer (KYC), fraud detection, and financial regulations.

## Your Role
You provide accurate, well-sourced information about:
- Anti-Money Laundering (AML) policies and procedures
- Know Your Customer (KYC) and Customer Due Diligence (CDD)
- Fraud detection and investigation protocols
- Financial regulations and compliance requirements
- Suspicious Activity Reports (SAR) and reporting obligations

## Guidelines
1. **Always cite your sources** using [1], [2], [3] notation corresponding to the numbered sources provided
2. **Be accurate and factual** - only use information from the provided context
3. **Never make up regulations or requirements** not present in the source documents
4. **Acknowledge uncertainty** - if the context doesn't contain enough information to answer confidently, say so
5. **Provide practical guidance** when possible, based on the retrieved documents
6. **Maintain professional tone** appropriate for financial industry professionals
7. **Organize your response clearly** with appropriate headings and bullet points

## Important Disclaimers
- Recommend consulting with legal counsel, compliance officers, or regulatory bodies for specific situations
- Note that regulations may vary by jurisdiction and change over time
- Advise users to verify current requirements with authoritative sources

When answering, reference the specific documents and sections that support your response."""


# Response template with citation formatting
RESPONSE_TEMPLATE = """Based on the retrieved documents, here is the answer to the question:

{response}

---
## Sources
{sources}

---
*Disclaimer: This information is provided for educational purposes. Always consult with qualified legal, compliance, or regulatory professionals for specific guidance on your situation.*"""


class RAGChain:
    """
    RAG generation chain with multi-provider LLM support.

    This class manages the generation component of the RAG pipeline, including:
    - Environment-aware LLM selection (Ollama, GLM-4, OpenAI)
    - Financial domain-specific prompting
    - Context-aware response generation
    - Source citation formatting

    Attributes:
        retriever: HybridRetriever instance for context retrieval
        environment: Current environment (development/demo/production)
        llm: Configured LLM instance
        system_prompt: Financial compliance system prompt

    Example:
        >>> from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever
        >>> retriever = HybridRetriever()
        >>> retriever.load_index()
        >>> rag_chain = RAGChain(retriever, environment="demo")
        >>> answer, sources = rag_chain.query(
        ...     "What are the SAR filing requirements?",
        ...     doc_type_filter="aml"
        ... )
        >>> print(answer)
        >>> print(f"Sources: {sources}")
    """

    # Environment-specific LLM configurations
    LLM_CONFIGS = {
        Environment.DEVELOPMENT: {
            "provider": "ollama",
            "model": "llama3.2:3b",
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 300,
        },
        Environment.DEMO: {
            "provider": "openai_compatible",
            "model": "glm-5",  # GLM-5 model
            "base_url": "https://api.z.ai/api/anthropic",  # Z.ai API endpoint
            "api_key_env": "ZHIPUAI_API_KEY",
            "temperature": 0.7,
            "max_tokens": 4096,
            "timeout": 300,
        },
        Environment.PRODUCTION: {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 4096,
            "timeout": 300,
        },
    }

    def __init__(
        self,
        retriever: Any,  # HybridRetriever - using Any to avoid circular import
        environment: str | Environment = Environment.DEVELOPMENT,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """
        Initialize the RAG chain.

        Args:
            retriever: HybridRetriever instance for context retrieval
            environment: Environment type (development/demo/production)
            system_prompt: Optional custom system prompt
            temperature: Optional override for LLM temperature
            max_tokens: Optional override for max tokens

        Raises:
            ValueError: If environment is invalid
            RuntimeError: If LLM initialization fails
            EnvironmentError: If required API keys are missing

        Example:
            >>> rag_chain = RAGChain(
            ...     retriever=retriever,
            ...     environment="demo"
            ... )
        """
        # Validate and set environment
        if isinstance(environment, str):
            try:
                self.environment = Environment(environment.lower())
            except ValueError:
                valid_envs = [e.value for e in Environment]
                raise ValueError(
                    f"Invalid environment: {environment}. "
                    f"Must be one of: {valid_envs}"
                )
        else:
            self.environment = environment

        self.retriever = retriever
        self.system_prompt = system_prompt or FINANCIAL_COMPLIANCE_SYSTEM_PROMPT

        # Get config for environment
        self.config = self.LLM_CONFIGS[self.environment].copy()

        # Apply overrides if provided
        if temperature is not None:
            self.config["temperature"] = temperature
        if max_tokens is not None:
            self.config["max_tokens"] = max_tokens

        # Initialize LLM
        self.llm = self._get_llm()

        logger.info(
            f"RAGChain initialized: environment={self.environment.value}, "
            f"provider={self.config['provider']}, model={self.config['model']}"
        )

    def _get_llm(self) -> LLM:
        """
        Initialize and return the appropriate LLM based on environment.

        Returns:
            Configured LLM instance

        Raises:
            RuntimeError: If LLM initialization fails
            EnvironmentError: If required API keys are missing
        """
        provider = self.config["provider"]

        try:
            if provider == "ollama":
                return self._init_ollama()
            elif provider == "openai":
                return self._init_openai()
            elif provider == "openai_compatible":
                return self._init_openai_compatible()
            else:
                raise ValueError(f"Unknown LLM provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise RuntimeError(f"LLM initialization failed: {e}") from e

    def _init_ollama(self) -> Ollama:
        """
        Initialize Ollama LLM for local development.

        Returns:
            Configured Ollama instance
        """
        logger.info(f"Initializing Ollama with model: {self.config['model']}")

        return Ollama(
            model=self.config["model"],
            temperature=self.config["temperature"],
            request_timeout=self.config["timeout"],
            context_window=4096,
        )

    def _init_openai(self) -> OpenAI:
        """
        Initialize OpenAI LLM for production.

        Returns:
            Configured OpenAI instance

        Raises:
            EnvironmentError: If OPENAI_API_KEY is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for production environment. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )

        logger.info(f"Initializing OpenAI with model: {self.config['model']}")

        return OpenAI(
            model=self.config["model"],
            api_key=api_key,
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            timeout=self.config["timeout"],
        )

    def _init_openai_compatible(self) -> "OpenAICompatibleLLM":
        """
        Initialize OpenAI-compatible API (GLM-4) for demo environment.

        Returns:
            Configured OpenAI-compatible LLM instance

        Raises:
            EnvironmentError: If ZHIPUAI_API_KEY is not set
        """
        api_key_env = self.config.get("api_key_env", "ZHIPUAI_API_KEY")
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} environment variable is required for demo environment. "
                f"Set it with: export {api_key_env}='your-key-here'"
            )

        logger.info(
            f"Initializing OpenAI-compatible API (GLM-4) at: {self.config['base_url']}"
        )

        # Use custom wrapper to bypass llama-index model validation
        return OpenAICompatibleLLM(
            model=self.config["model"],
            api_key=api_key,
            base_url=self.config["base_url"],
            temperature=self.config["temperature"],
            max_tokens=self.config["max_tokens"],
            timeout=self.config["timeout"],
        )

    def _format_context_with_numbers(self, context: str) -> tuple[str, list[str]]:
        """
        Format context string with numbered citation markers.

        Args:
            context: Raw context string from retriever

        Returns:
            Tuple of (formatted_context, list of source references)

        Example:
            >>> context = retriever.format_context(nodes)
            >>> formatted, sources = self._format_context_with_numbers(context)
            >>> print(f"Sources: {sources}")
        """
        # Split context into chunks
        lines = context.split("\n")
        formatted_lines = []
        sources = []

        current_source = None
        chunk_number = 0

        for line in lines:
            # Detect citation headers: [1. filename | Category: ...]
            if line.strip().startswith("[") and "| Category:" in line:
                chunk_number += 1

                # Extract source name
                try:
                    source_part = line.strip().split("|")[0].strip()
                    # Remove the leading number and bracket
                    source_name = source_part.split(".", 1)[1].strip().rstrip("]")
                    sources.append(source_name)
                except Exception:
                    sources.append(f"Source {chunk_number}")

                # Replace citation header with numbered marker
                formatted_line = line.split("]")[0] + f"] [{chunk_number}]"
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)

        formatted_context = "\n".join(formatted_lines)

        logger.debug(f"Formatted {len(sources)} sources with citation markers")

        return formatted_context, sources

    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create the complete prompt with system message and context.

        Args:
            question: User's question
            context: Retrieved context with citations

        Returns:
            Complete prompt string

        Example:
            >>> prompt = self._create_prompt(
            ...     "What are SAR requirements?",
            ...     context
            ... )
        """
        # Format context with numbered citations
        formatted_context, sources = self._format_context_with_numbers(context)

        prompt = f"""{self.system_prompt}

## Question
{question}

## Context
Below are relevant excerpts from financial documents:

{formatted_context}

## Instructions
Using only the information provided in the context above, answer the question:
1. Provide a clear, accurate answer based on the documents
2. Cite your sources using [1], [2], [3] notation
3. If the context doesn't contain enough information, acknowledge this limitation
4. Organize your answer with appropriate structure

Answer:"""

        return prompt

    def query(
        self,
        question: str,
        doc_type_filter: str | list[str] | None = None,
        use_rerank: bool = True,
        top_k: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Query the RAG system and generate a response with citations.

        Args:
            question: User's question
            doc_type_filter: Optional filter by document category
            use_rerank: Whether to apply cross-encoder reranking
            top_k: Optional override for number of retrieved documents

        Returns:
            Tuple of (answer_text, list_of_sources)

            Each source is a dict with keys:
            - file_name: str
            - category: str
            - score: float
            - text_preview: str

        Raises:
            RuntimeError: If query generation fails
            ValueError: If question is empty

        Example:
            >>> answer, sources = rag_chain.query(
            ...     "What are the SAR filing deadlines?",
            ...     doc_type_filter="aml"
            ... )
            >>> print(f"Answer: {answer}")
            >>> print(f"\\nSources used:")
            >>> for source in sources:
            ...     print(f"  - {source['file_name']} ({source['category']})")
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Processing query: {question[:100]}...")

        try:
            # Step 1: Retrieve relevant documents
            logger.info("Retrieving relevant documents...")

            # Override top_k if specified
            original_top_k = self.retriever.top_k
            if top_k is not None:
                self.retriever.top_k = top_k

            nodes = self.retriever.retrieve(
                query=question,
                doc_type_filter=doc_type_filter,
                use_rerank=use_rerank,
            )

            # Restore original top_k
            self.retriever.top_k = original_top_k

            if not nodes:
                logger.warning("No relevant documents found")
                return (
                    "I couldn't find any relevant documents to answer your question. "
                    "This could mean:\n"
                    "1. The question is outside the scope of the available documents\n"
                    "2. Try using different search terms\n"
                    "3. Remove any document type filters to search across all categories",
                    []
                )

            logger.info(f"Retrieved {len(nodes)} relevant documents")

            # Step 2: Format context
            context = self.retriever.format_context(
                nodes,
                include_scores=False,
            )

            # Step 3: Create prompt
            prompt = self._create_prompt(question, context)

            # Step 4: Generate response
            logger.info("Generating response with LLM...")

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]

            response = self.llm.chat(messages)

            answer_text = response.message.content or "No response generated"

            logger.info(f"Generated response ({len(answer_text)} characters)")

            # Step 5: Extract source information
            sources = []
            for node in nodes:
                metadata = node.node.metadata
                sources.append({
                    "file_name": metadata.get("file_name", "Unknown"),
                    "category": metadata.get("category", "general"),
                    "score": float(node.score) if node.score else 0.0,
                    "title": metadata.get("title", ""),
                    "text_preview": (node.node.text or "")[:200] + "..."
                    if node.node.text else ""
                })

            logger.info(f"Returning answer with {len(sources)} sources")

            return answer_text, sources

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process query: {e}") from e

    def query_stream(
        self,
        question: str,
        doc_type_filter: str | list[str] | None = None,
        use_rerank: bool = True,
    ):
        """
        Query the RAG system with streaming response.

        Args:
            question: User's question
            doc_type_filter: Optional filter by document category
            use_rerank: Whether to apply cross-encoder reranking

        Yields:
            Response chunks as they are generated

        Example:
            >>> for chunk in rag_chain.query_stream("Explain AML requirements"):
            ...     print(chunk, end="", flush=True)
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Processing streaming query: {question[:100]}...")

        try:
            # Retrieve relevant documents
            nodes = self.retriever.retrieve(
                query=question,
                doc_type_filter=doc_type_filter,
                use_rerank=use_rerank,
            )

            if not nodes:
                yield (
                    "I couldn't find any relevant documents to answer your question. "
                    "Please try different search terms or remove document type filters."
                )
                return

            # Format context
            context = self.retriever.format_context(nodes, include_scores=False)

            # Create prompt
            prompt = self._create_prompt(question, context)

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ]

            # Stream response
            logger.info("Streaming response from LLM...")
            for chunk in self.llm.stream_chat(messages):
                if chunk.delta:
                    yield chunk.delta

        except Exception as e:
            logger.error(f"Streaming query failed: {e}", exc_info=True)
            yield f"\n\nError: Failed to generate response - {str(e)}"

    def get_config(self) -> dict[str, Any]:
        """
        Get the current configuration of the RAG chain.

        Returns:
            Dictionary containing configuration details

        Example:
            >>> config = rag_chain.get_config()
            >>> print(f"Provider: {config['provider']}")
            >>> print(f"Model: {config['model']}")
        """
        return {
            "environment": self.environment.value,
            "provider": self.config["provider"],
            "model": self.config["model"],
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"],
            "timeout": self.config["timeout"],
        }


def main() -> None:
    """
    Demonstration of RAGChain usage.

    This main block demonstrates:
    - Environment-specific LLM initialization
    - Query processing with citations
    - Source extraction and formatting
    - Streaming responses
    """
    import sys

    from fraud_docs_rag.retrieval.hybrid_retriever import HybridRetriever
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("FraudDocs-RAG RAG Chain Demonstration")
    logger.info("=" * 70)

    try:
        # Determine environment from command line or default to development
        import argparse
        parser = argparse.ArgumentParser(description="RAG Chain Demo")
        parser.add_argument(
            "--env",
            choices=["development", "demo", "production"],
            default="development",
            help="Environment to run in"
        )
        args = parser.parse_args()

        environment = args.env

        logger.info(f"Running in environment: {environment}")

        # Initialize embedding model
        logger.info("Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
        )

        # Initialize retriever with sample data
        logger.info("Initializing retriever...")
        retriever = HybridRetriever(
            collection_name="demo_rag_chain",
            chroma_path="./data/chroma_db_rag_demo",
            top_k=5,
            rerank_top_n=3,
        )

        # Check if we can load existing index, otherwise build it
        if not retriever.load_index():
            logger.info("No existing index found, creating sample data...")

            # Create sample nodes
            sample_nodes = [
                TextNode(
                    text=(
                        "Suspicious Activity Reports (SAR) must be filed within 30 days "
                        "of detecting suspicious transaction activity. Financial institutions "
                        "are required to report any transactions that appear suspicious, "
                        "involve illegal funds, or have no business purpose. The SAR form "
                        "must be completed accurately and filed electronically through FinCEN."
                    ),
                    metadata={
                        "file_name": "aml_sar_requirements.pdf",
                        "category": "aml",
                        "title": "SAR Filing Requirements",
                    }
                ),
                TextNode(
                    text=(
                        "Customer Due Diligence (CDD) requires verification of customer "
                        "identity using reliable, independent source documents. Enhanced "
                        "Due Diligence (EDD) must be applied for high-risk customers, "
                        "including politically exposed persons (PEPs) and customers from "
                        "high-risk jurisdictions identified by FATF."
                    ),
                    metadata={
                        "file_name": "kyc_cdd_requirements.pdf",
                        "category": "kyc",
                        "title": "Customer Due Diligence Procedures",
                    }
                ),
                TextNode(
                    text=(
                        "Fraud detection systems must monitor for unusual transaction patterns, "
                        "including rapid movement of funds, structuring to avoid reporting "
                        "thresholds (smurfing), and transactions with high-risk jurisdictions. "
                        "All fraud alerts must be investigated by trained personnel within "
                        "24 hours of generation."
                    ),
                    metadata={
                        "file_name": "fraud_detection_manual.pdf",
                        "category": "fraud",
                        "title": "Fraud Detection Protocols",
                    }
                ),
                TextNode(
                    text=(
                        "Financial institutions must establish comprehensive AML programs "
                        "including: written policies and procedures, a designated BSA/AML "
                        "compliance officer, ongoing employee training, and independent "
                        "testing. The program must be approved by the board of directors "
                        "and updated regularly to reflect regulatory changes."
                    ),
                    metadata={
                        "file_name": "aml_compliance_program.pdf",
                        "category": "aml",
                        "title": "AML Program Requirements",
                    }
                ),
            ]

            # Add embeddings
            for node in sample_nodes:
                node.embedding = embed_model.get_text_embedding(node.text)

            # Build index
            retriever.build_index(sample_nodes)

        # Initialize RAG chain
        logger.info("Initializing RAG Chain...")
        rag_chain = RAGChain(
            retriever=retriever,
            environment=environment,
        )

        logger.info(f"Configuration: {rag_chain.get_config()}")
        print()

        # Example 1: Basic query
        logger.info("Example 1: Basic query")
        logger.info("-" * 70)
        question1 = "What are the requirements for filing Suspicious Activity Reports?"

        logger.info(f"Question: {question1}")
        answer1, sources1 = rag_chain.query(question1)

        print(f"\nAnswer:\n{answer1}\n")
        print("Sources:")
        for i, source in enumerate(sources1, start=1):
            print(f"  [{i}] {source['file_name']} ({source['category']}) - Score: {source['score']:.3f}")
        print()

        # Example 2: Filtered query
        logger.info("Example 2: Filtered query (KYC only)")
        logger.info("-" * 70)
        question2 = "What are the customer verification requirements?"

        logger.info(f"Question: {question2}")
        answer2, sources2 = rag_chain.query(question2, doc_type_filter="kyc")

        print(f"\nAnswer:\n{answer2}\n")
        print("Sources:")
        for i, source in enumerate(sources2, start=1):
            print(f"  [{i}] {source['file_name']} ({source['category']}) - Score: {source['score']:.3f}")
        print()

        # Example 3: Streaming response
        logger.info("Example 3: Streaming response")
        logger.info("-" * 70)
        question3 = "Explain the AML program requirements."

        logger.info(f"Question: {question3}")
        print(f"\nStreaming Answer:\n")

        for chunk in rag_chain.query_stream(question3):
            print(chunk, end="", flush=True)

        print("\n")

        logger.info("=" * 70)
        logger.info("Demonstration complete!")

        # Cleanup
        import shutil
        demo_chroma_path = Path("./data/chroma_db_rag_demo")
        if demo_chroma_path.exists():
            retriever.delete_index()
            shutil.rmtree(demo_chroma_path)
            logger.info("Cleaned up demo data")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        sys.exit(1)


class OpenAICompatibleLLM(CustomLLM):
    """
    Custom LLM wrapper for OpenAI-compatible and Anthropic-compatible APIs.
    Bypasses llama-index's strict model name validation.
    """

    model: str = "glm-5"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 300
    context_window: int = 128000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            # Use anthropic library for z.ai endpoint
            if "anthropic" in self.base_url or "z.ai" in self.base_url:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            else:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
        return self._client

    @property
    def is_anthropic(self):
        return "anthropic" in self.base_url or "z.ai" in self.base_url

    @property
    def metadata(self):
        from llama_index.core.llms import LLMMetadata
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    def chat(self, messages, **kwargs):
        """Sync chat completion."""
        from llama_index.core.llms import ChatResponse

        if self.is_anthropic:
            # Anthropic API format
            system_msg = ""
            chat_messages = []
            for msg in messages:
                role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                if role == "system":
                    system_msg = msg.content
                else:
                    chat_messages.append({"role": role, "content": msg.content})

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_msg if system_msg else "You are a helpful assistant.",
                messages=chat_messages,
            )

            content = response.content[0].text if response.content else ""
            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                ),
                raw=response.model_dump() if hasattr(response, 'model_dump') else {},
            )
        else:
            # OpenAI API format
            formatted_messages = [
                {"role": msg.role.value if hasattr(msg.role, 'value') else msg.role, "content": msg.content}
                for msg in messages
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.choices[0].message.content,
                ),
                raw=response.model_dump(),
            )

    def stream_chat(self, messages, **kwargs):
        """Stream chat completion."""
        from llama_index.core.llms import ChatResponse

        if self.is_anthropic:
            # Anthropic API format
            system_msg = ""
            chat_messages = []
            for msg in messages:
                role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                if role == "system":
                    system_msg = msg.content
                else:
                    chat_messages.append({"role": role, "content": msg.content})

            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_msg if system_msg else "You are a helpful assistant.",
                messages=chat_messages,
            ) as stream:
                content = ""
                for text in stream.text_stream:
                    content += text
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=content,
                        ),
                        delta=text,
                        raw={},
                    )
        else:
            # OpenAI API format
            formatted_messages = [
                {"role": msg.role.value if hasattr(msg.role, 'value') else msg.role, "content": msg.content}
                for msg in messages
            ]

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=content,
                        ),
                        delta=chunk.choices[0].delta.content,
                        raw=chunk.model_dump() if hasattr(chunk, 'model_dump') else {},
                    )

    def complete(self, prompt, **kwargs):
        """Sync completion (uses chat under the hood)."""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = self.chat(messages, **kwargs)
        from llama_index.core.llms import CompletionResponse
        return CompletionResponse(text=response.message.content, raw=response.raw)

    def stream_complete(self, prompt, **kwargs):
        """Stream completion (uses chat under the hood)."""
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        content = ""
        for partial_response in self.stream_chat(messages, **kwargs):
            content = partial_response.message.content
            from llama_index.core.llms import CompletionResponse
            yield CompletionResponse(
                text=content,
                delta=partial_response.delta if hasattr(partial_response, 'delta') else "",
                raw=partial_response.raw if hasattr(partial_response, 'raw') else {},
            )


if __name__ == "__main__":
    main()
