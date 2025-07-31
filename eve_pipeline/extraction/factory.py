"""Extractor factory for creating appropriate extractors based on file type."""

from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from eve_pipeline.extraction.base import ExtractorBase
from eve_pipeline.extraction.pdf_extractor import PDFExtractor
from eve_pipeline.extraction.xml_extractor import XMLExtractor
from eve_pipeline.extraction.html_extractor import HTMLExtractor
from eve_pipeline.extraction.text_extractor import TextExtractor
from eve_pipeline.extraction.csv_extractor import CSVExtractor


class ExtractorFactory:
    """Factory for creating appropriate extractors based on file type."""
    
    # Default extractor mapping
    _EXTRACTORS: Dict[str, Type[ExtractorBase]] = {
        "pdf": PDFExtractor,
        "xml": XMLExtractor,
        "html": HTMLExtractor,
        "htm": HTMLExtractor,
        "txt": TextExtractor,
        "text": TextExtractor,
        "md": TextExtractor,
        "markdown": TextExtractor,
        "csv": CSVExtractor,
        "tsv": CSVExtractor,
    }
    
    def __init__(self) -> None:
        """Initialize factory."""
        self._custom_extractors: Dict[str, Type[ExtractorBase]] = {}
    
    def register_extractor(
        self, 
        file_extension: str, 
        extractor_class: Type[ExtractorBase]
    ) -> None:
        """Register a custom extractor for a file extension.
        
        Args:
            file_extension: File extension (without dot).
            extractor_class: Extractor class.
        """
        self._custom_extractors[file_extension.lower()] = extractor_class
    
    def get_extractor(
        self, 
        file_path: Union[str, Path],
        **kwargs
    ) -> Optional[ExtractorBase]:
        """Get appropriate extractor for file.
        
        Args:
            file_path: Path to file.
            **kwargs: Additional arguments for extractor initialization.
            
        Returns:
            Extractor instance or None if unsupported.
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip(".")
        
        # Check custom extractors first
        if extension in self._custom_extractors:
            extractor_class = self._custom_extractors[extension]
            return extractor_class(**kwargs)
        
        # Check default extractors
        if extension in self._EXTRACTORS:
            extractor_class = self._EXTRACTORS[extension]
            return extractor_class(**kwargs)
        
        return None
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be extracted.
        
        Args:
            file_path: Path to file.
            
        Returns:
            True if file can be extracted.
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip(".")
        return (extension in self._EXTRACTORS or 
                extension in self._custom_extractors)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file extensions.
        """
        return list(set(self._EXTRACTORS.keys()) | set(self._custom_extractors.keys()))
    
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Optional[str]:
        """Extract content from file using appropriate extractor.
        
        Args:
            file_path: Path to input file.
            output_path: Optional path for output file.
            **kwargs: Additional arguments for extractor.
            
        Returns:
            Extracted content as string, or None if failed.
        """
        extractor = self.get_extractor(file_path, **kwargs)
        if not extractor:
            return None
        
        try:
            result = extractor.process_file(file_path, output_path, **kwargs)
            return result.content if result.is_success else None
        except Exception:
            return None
    
    @classmethod
    def get_default_factory(cls) -> "ExtractorFactory":
        """Get default factory instance.
        
        Returns:
            Default ExtractorFactory instance.
        """
        return cls()


# Global default factory instance
default_factory = ExtractorFactory.get_default_factory()