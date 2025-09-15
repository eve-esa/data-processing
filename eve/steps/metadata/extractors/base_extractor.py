"""Base class for metadata extractors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from eve.model.document import Document
from eve.logging import get_logger


class BaseMetadataExtractor(ABC):
    """Abstract base class for metadata extractors."""

    def __init__(self, debug: bool = False):
        """
        Initialize the base metadata extractor.
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata, or None if extraction fails
        """
        pass

    def _clean_title(self, title: str) -> Optional[str]:
        """
        Clean and normalize a title string.
        
        Args:
            title: Raw title string
            
        Returns:
            Cleaned title or None if invalid
        """
        if not title or not isinstance(title, str):
            return None
            
        cleaned = title.strip()
        
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        if len(cleaned) < 3 or cleaned.isdigit():
            return None
            
        return cleaned

    def _extract_title_from_filename(self, file_path: Path) -> str:
        """
        Extract a readable title from filename.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Title derived from filename
        """
        title = file_path.stem
        
        title = title.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        while '  ' in title:
            title = title.replace('  ', ' ')
        
        return title.strip().title()
