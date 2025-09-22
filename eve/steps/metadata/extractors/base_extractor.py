"""Base class for metadata extractors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
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

    def _validate_document_format(self, document: Document, expected_format: str) -> bool:
        """
        Validate that the document has the expected format.
        
        Args:
            document: Document to validate
            expected_format: Expected file format
            
        Returns:
            True if format matches, False otherwise
        """
        if document.file_format != expected_format:
            self.logger.warning(f"Expected {expected_format} format, got {document.file_format}")
            return False
        return True

    def _map_metadata_fields(self, source_data: Dict[str, Any], field_mapping: Dict[str, str], 
                           target_metadata: Dict[str, Any], apply_cleaning: Optional[List[str]] = None) -> None:
        """
        Map fields from source data to target metadata with optional cleaning.
        
        Args:
            source_data: Source dictionary containing raw metadata
            field_mapping: Mapping of source_key -> target_key
            target_metadata: Target dictionary to update
            apply_cleaning: List of fields that should be cleaned with _clean_title
        """
        apply_cleaning = apply_cleaning or []
        
        for source_key, target_key in field_mapping.items():
            if source_key in source_data:
                value = source_data[source_key]
                if target_key in apply_cleaning and isinstance(value, str):
                    cleaned_value = self._clean_title(value)
                    if cleaned_value:
                        target_metadata[target_key] = cleaned_value
                else:
                    target_metadata[target_key] = value

    def _set_title_with_fallback(self, metadata: Dict[str, Any], extracted_title: Optional[str], 
                                document: Document, title_source: str = 'extracted') -> None:
        """
        Set title in metadata with filename fallback if no title is available.
        
        Args:
            metadata: Metadata dictionary to update
            extracted_title: Extracted title (can be None)
            document: Document object for filename fallback
            title_source: Source description for extracted title
        """
        if extracted_title:
            metadata['title'] = extracted_title
            metadata['title_source'] = title_source
        else:
            metadata['title'] = self._extract_title_from_filename(document.file_path)
            metadata['title_source'] = 'filename'

    def _process_authors(self, authors: Union[str, List[str]]) -> List[str]:
        """
        Process and normalize author information.
        
        Args:
            authors: Authors as string or list
            
        Returns:
            List of cleaned author names
        """
        if isinstance(authors, list):
            return [author.strip() for author in authors if author.strip()]
        elif isinstance(authors, str):
            # Try common separators
            for separator in [' and ', ', ', '; ']:
                if separator in authors:
                    return [author.strip() for author in authors.split(separator) if author.strip()]
            # Single author
            return [authors.strip()] if authors.strip() else []
        return []

    def _add_extraction_method(self, metadata: Dict[str, Any], method: str) -> None:
        """
        Add extraction method to metadata.
        
        Args:
            metadata: Metadata dictionary to update
            method: Extraction method name
        """
        if 'extraction_methods' not in metadata:
            metadata['extraction_methods'] = []
        if method not in metadata['extraction_methods']:
            metadata['extraction_methods'].append(method)

    def _finalize_metadata(self, metadata: Dict[str, Any], document: Document) -> Optional[Dict[str, Any]]:
        """
        Finalize metadata by adding debug logging and validation.
        
        Args:
            metadata: Metadata dictionary
            document: Document object
            
        Returns:
            Finalized metadata or None if empty
        """
        if not metadata:
            return None
            
        if self.debug:
            self.logger.debug(f"Extracted metadata for {document.filename}: {metadata}")
            
        return metadata

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
