"""Common utility functions for the eve-pipeline."""

import hashlib
import re
from pathlib import Path
from typing import Union

from eve_pipeline.core.enums import HashAlgorithm


class PathUtils:
    """Utility functions for path operations."""



    @staticmethod
    def ensure_path_exists(path: Union[str, Path], is_file: bool = False) -> Path:
        """Ensure path exists, creating parent directories if needed.

        Args:
            path: Path to ensure exists.
            is_file: Whether the path is a file (create parent dirs only).

        Returns:
            Path object.
        """
        path = Path(path)

        if is_file:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

        return path




class HashUtils:
    """Utility functions for hashing operations."""

    @staticmethod
    def calculate_hash(
        content: Union[str, bytes],
        algorithm: HashAlgorithm = HashAlgorithm.MD5,
    ) -> str:
        """Calculate hash of content.

        Args:
            content: Content to hash.
            algorithm: Hash algorithm to use.

        Returns:
            Hex digest of the hash.
        """
        if isinstance(content, str):
            content = content.encode('utf-8')

        if algorithm == HashAlgorithm.MD5:
            return hashlib.md5(content).hexdigest()
        elif algorithm == HashAlgorithm.SHA1:
            return hashlib.sha1(content).hexdigest()
        elif algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(content).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @staticmethod
    def normalize_content_for_hashing(
        content: str,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
    ) -> str:
        """Normalize content for consistent hashing.

        Args:
            content: Content to normalize.
            case_sensitive: Whether to preserve case.
            normalize_whitespace: Whether to normalize whitespace.

        Returns:
            Normalized content.
        """
        normalized = content

        if not case_sensitive:
            normalized = normalized.lower()

        if normalize_whitespace:
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = normalized.strip()

        return normalized





class RetryUtils:
    """Utility functions for retry mechanisms."""

    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-based).
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.

        Returns:
            Delay in seconds.
        """
        delay = base_delay * (2 ** attempt)
        return min(delay, max_delay)

    @staticmethod
    def should_retry(attempt: int, max_attempts: int) -> bool:
        """Check if should retry based on attempt count.

        Args:
            attempt: Current attempt number (0-based).
            max_attempts: Maximum number of attempts.

        Returns:
            True if should retry, False otherwise.
        """
        return attempt < max_attempts - 1
