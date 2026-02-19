# ============================================================
# StreamProcess-Pipeline: Security Module
# ============================================================
"""
Security utilities for the Stream Processing Pipeline.

This module provides:
- SensitiveDataFilter: A logging filter to redact sensitive information
- install_security_filter: Function to install the filter on the root logger
"""

import logging
import re
from typing import Any, Dict, List, Optional


class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that redacts sensitive data from log messages.

    This filter helps prevent sensitive information (passwords, API keys,
    tokens, etc.) from appearing in logs by replacing them with a redacted
    placeholder.

    Attributes:
        patterns: Dictionary of regex patterns to match and redact
        replacement: String to replace matched patterns with
    """

    # Default sensitive patterns to redact
    DEFAULT_PATTERNS: Dict[str, str] = {
        # Passwords in various formats
        r"(?i)(password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?([^\"'\s,})]+)": r"\1='***REDACTED***'",
        # API keys and tokens
        r"(?i)(api[_-]?key|token|access[_-]?token|auth[_-]?token|bearer)[\"']?\s*[:=]\s*[\"']?([^\"'\s,})]+)": r"\1='***REDACTED***'",
        # JWT tokens (eyJ...)
        r"(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*)": "***JWT_REDACTED***",
        # UUID-like patterns (potential session IDs)
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b": "***UUID_REDACTED***",
        # Credit card numbers (basic pattern)
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b": "***CARD_REDACTED***",
        # Email addresses (optional, can be enabled)
        # r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b": "***EMAIL_REDACTED***",
    }

    def __init__(
        self,
        name: str = "",
        patterns: Optional[Dict[str, str]] = None,
        replacement: str = "***REDACTED***"
    ) -> None:
        """
        Initialize the SensitiveDataFilter.

        Args:
            name: Name of the filter (for logging.Filter compatibility)
            patterns: Custom regex patterns to redact (uses defaults if None)
            replacement: String to replace matched patterns with
        """
        super().__init__(name)
        self.patterns = patterns if patterns is not None else self.DEFAULT_PATTERNS
        self.replacement = replacement
        self._compiled_patterns: List[re.Pattern] = [
            re.compile(pattern) for pattern in self.patterns.keys()
        ]

    def add_pattern(self, pattern: str, replacement: Optional[str] = None) -> None:
        """
        Add a new pattern to the filter.

        Args:
            pattern: Regex pattern to match
            replacement: Custom replacement string (uses default if None)
        """
        self.patterns[pattern] = replacement or self.replacement
        self._compiled_patterns.append(re.compile(pattern))

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter the log record by redacting sensitive data.

        Args:
            record: The log record to filter

        Returns:
            True (always allows the record through, just modifies it)
        """
        # Redact from the message
        record.msg = self._redact(str(record.msg))

        # Redact from args if present
        if record.args:
            record.args = tuple(
                self._redact(str(arg)) if isinstance(arg, (str, bytes)) else arg
                for arg in record.args
            )

        return True

    def _redact(self, text: str) -> str:
        """
        Redact sensitive data from the given text.

        Args:
            text: Text to redact

        Returns:
            Text with sensitive data redacted
        """
        for pattern in self._compiled_patterns:
            text = pattern.sub(self.replacement, text)
        return text


# Global filter instance
_filter: Optional[SensitiveDataFilter] = None


def install_security_filter(
    level: int = logging.WARNING,
    custom_patterns: Optional[Dict[str, str]] = None
) -> SensitiveDataFilter:
    """
    Install the SensitiveDataFilter on the root logger.

    This function should be called during application initialization to
    ensure all log messages are filtered for sensitive data.

    Args:
        level: Minimum log level to apply filtering (default: WARNING)
        custom_patterns: Optional custom patterns to add to defaults

    Returns:
        The installed SensitiveDataFilter instance

    Example:
        >>> install_security_filter()
        <shared.security.SensitiveDataFilter object at ...>
    """
    global _filter

    if _filter is None:
        _filter = SensitiveDataFilter()

        if custom_patterns:
            for pattern, replacement in custom_patterns.items():
                _filter.add_pattern(pattern, replacement)

        # Install on root logger
        root_logger = logging.getLogger()
        root_logger.addFilter(_filter)

        # Install on all existing loggers
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.addFilter(_filter)

    return _filter


def get_security_filter() -> Optional[SensitiveDataFilter]:
    """
    Get the currently installed security filter.

    Returns:
        The installed SensitiveDataFilter or None if not installed
    """
    return _filter


def redact(text: str) -> str:
    """
    Convenience function to redact sensitive data from a string.

    Args:
        text: Text to redact

    Returns:
        Redacted text
    """
    if _filter is None:
        # Use a temporary filter if none is installed
        temp_filter = SensitiveDataFilter()
        return temp_filter._redact(text)
    return _filter._redact(text)
