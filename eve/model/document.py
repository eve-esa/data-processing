"""Unified Document object for the EVE pipeline."""

from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Document:
    """
    Unified document object that encapsulates content and metadata throughout the pipeline.

    This replaces the need to pass (Path, str) tuples and provides a consistent
    interface for document handling across all pipeline stages.
    """

    content: str
    file_path: Path
    file_format: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __dict__(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "file_path": str(self.file_path),
            "file_format": self.file_format,
            "metadata": self.metadata.copy(),
        }

    def __hash__(self):
        return hash(self.file_path)

    def __eq__(self, other):
        return isinstance(other, Document) and self.file_path == other.file_path

    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.file_path.name

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return self.file_path.suffix.lstrip(".")

    @property
    def content_length(self) -> int:
        """Get the length of the content."""
        return len(self.content)

    def is_empty(self) -> bool:
        """Check if the document content is empty."""
        return not self.content.strip()

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata entry."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value with optional default."""
        return self.metadata.get(key, default)

    def update_content(self, new_content: str) -> None:
        """Update the document content and track the change in metadata."""
        old_length = self.content_length
        self.content = new_content
        new_length = self.content_length

        # Track content changes in metadata
        changes = self.metadata.get("content_changes", [])
        changes.append(
            {
                "old_length": old_length,
                "new_length": new_length,
                "size_change": new_length - old_length,
            }
        )
        self.metadata["content_changes"] = changes

    @classmethod
    def from_path_and_content(
        cls, file_path: Path, content: str, **metadata
    ) -> "Document":
        """Create a Document from a file path and content string."""
        return cls(content=content, file_path=file_path, metadata=metadata)

    @classmethod
    def from_tuple(cls, path_content_tuple: tuple[Path, str], **metadata) -> "Document":
        """Create a Document from a (Path, str) tuple for backwards compatibility."""
        file_path, content = path_content_tuple
        return cls.from_path_and_content(file_path, content, **metadata)

    def to_tuple(self) -> tuple[Path, str]:
        """Convert to (Path, str) tuple for backwards compatibility."""
        return (self.file_path, self.content)

    def __str__(self) -> str:
        """String representation showing filename and content length."""
        return f"Document({self.filename}, {self.file_format} format)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Document(file_path={self.file_path}, format={self.file_format}, metadata_keys={list(self.metadata.keys())})"
