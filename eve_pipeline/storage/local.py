"""Local filesystem storage backend."""

import shutil
from pathlib import Path
from typing import Any, Optional

from eve_pipeline.storage.base import StorageBase


class LocalStorage(StorageBase):
    """Local filesystem storage backend."""

    def __init__(self, base_path: Optional[str] = None, **kwargs) -> None:
        """Initialize local storage.

        Args:
            base_path: Optional base path for relative operations.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.base_path = Path(base_path) if base_path else None

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base path if set.

        Args:
            path: Input path.

        Returns:
            Resolved Path object.
        """
        path_obj = Path(path)
        if self.base_path and not path_obj.is_absolute():
            return self.base_path / path_obj
        return path_obj

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        return self._resolve_path(path).exists()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text content from a file."""
        file_path = self._resolve_path(path)

        # Try multiple encodings if utf-8 fails
        encodings = [encoding, "utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for enc in encodings:
            try:
                with open(file_path, encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise Exception(f"Cannot decode file {file_path} with any supported encoding")

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text content to a file."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)

    def read_bytes(self, path: str) -> bytes:
        """Read binary content from a file."""
        file_path = self._resolve_path(path)
        with open(file_path, "rb") as f:
            return f.read()

    def write_bytes(self, path: str, content: bytes) -> None:
        """Write binary content to a file."""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(content)

    def list_files(self, path: str, pattern: Optional[str] = None) -> list[str]:
        """List files in a directory."""
        dir_path = self._resolve_path(path)

        if not dir_path.exists():
            return []

        if not dir_path.is_dir():
            return [str(dir_path)] if dir_path.is_file() else []

        files = []

        if pattern:
            # Use glob pattern if provided
            if pattern.startswith("**/"):
                # Recursive glob
                files.extend(str(p) for p in dir_path.rglob(pattern[3:]) if p.is_file())
            else:
                # Non-recursive glob
                files.extend(str(p) for p in dir_path.glob(pattern) if p.is_file())
        else:
            # List all files recursively
            files.extend(str(p) for p in dir_path.rglob("*") if p.is_file())

        return sorted(files)

    def delete(self, path: str) -> None:
        """Delete a file or directory."""
        file_path = self._resolve_path(path)

        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)

    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy a file."""
        src = self._resolve_path(src_path)
        dst = self._resolve_path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def get_metadata(self, path: str) -> dict[str, Any]:
        """Get file metadata."""
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return {}

        stat = file_path.stat()

        return {
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "permissions": oct(stat.st_mode)[-3:],
            "path": str(file_path),
        }

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory."""
        return self._resolve_path(path).is_dir()

    def mkdir(self, path: str, parents: bool = True, exist_ok: bool = True) -> None:
        """Create a directory.

        Args:
            path: Directory path to create.
            parents: Create parent directories if needed.
            exist_ok: Don't raise error if directory exists.
        """
        dir_path = self._resolve_path(path)
        dir_path.mkdir(parents=parents, exist_ok=exist_ok)
