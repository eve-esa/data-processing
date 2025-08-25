"""Core enums for the eve-pipeline."""

from enum import Enum


class ProcessorStatus(Enum):
    """Status of a processor operation."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class HashAlgorithm(Enum):
    """Supported hash algorithms for deduplication."""
    MD5 = "md5"
    SHA1 = "sha1"
    SHA256 = "sha256"


class ExtractionMethod(Enum):
    """PDF extraction methods."""
    NOUGAT = "nougat"
    MARKER = "marker"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
