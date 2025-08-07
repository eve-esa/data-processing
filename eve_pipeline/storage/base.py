"""Base storage interface for different storage backends."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging


class StorageBase(ABC):
    """Base class for storage backends."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize storage backend."""
        self.logger = logging.getLogger(f"eve_pipeline.storage.{self.__class__.__name__}")
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists.
        
        Args:
            path: Path to check.
            
        Returns:
            True if exists, False otherwise.
        """
        pass
    
    @abstractmethod
    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text content from a file.
        
        Args:
            path: Path to file.
            encoding: Text encoding.
            
        Returns:
            File content as string.
        """
        pass
    
    @abstractmethod
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text content to a file.
        
        Args:
            path: Path to file.
            content: Content to write.
            encoding: Text encoding.
        """
        pass
    
    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read binary content from a file.
        
        Args:
            path: Path to file.
            
        Returns:
            File content as bytes.
        """
        pass
    
    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> None:
        """Write binary content to a file.
        
        Args:
            path: Path to file.
            content: Content to write.
        """
        pass
    
    @abstractmethod
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory.
        
        Args:
            path: Directory path.
            pattern: Optional file pattern to match.
            
        Returns:
            List of file paths.
        """
        pass
    
    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file or directory.
        
        Args:
            path: Path to delete.
        """
        pass
    
    @abstractmethod
    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy a file.
        
        Args:
            src_path: Source path.
            dst_path: Destination path.
        """
        pass
    
    @abstractmethod
    def get_metadata(self, path: str) -> Dict[str, Any]:
        """Get file metadata.
        
        Args:
            path: Path to file.
            
        Returns:
            Dictionary with metadata.
        """
        pass
    
    def is_file(self, path: str) -> bool:
        """Check if path is a file.
        
        Args:
            path: Path to check.
            
        Returns:
            True if it's a file.
        """
        return self.exists(path) and not self.is_dir(path)
    
    def is_dir(self, path: str) -> bool:
        """Check if path is a directory.
        
        Args:
            path: Path to check.
            
        Returns:
            True if it's a directory.
        """
        # Default implementation - can be overridden by subclasses
        try:
            files = self.list_files(path)
            return True
        except Exception:
            return False
    
    @staticmethod
    def is_s3_path(path: str) -> bool:
        """Check if path is an S3 path.
        
        Args:
            path: Path to check.
            
        Returns:
            True if it's an S3 path.
        """
        return isinstance(path, str) and path.startswith("s3://")
    
    @staticmethod
    def is_local_path(path: str) -> bool:
        """Check if path is a local path.
        
        Args:
            path: Path to check.
            
        Returns:
            True if it's a local path.
        """
        return not StorageBase.is_s3_path(path)