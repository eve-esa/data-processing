"""
Base class for metadata extractors.

This module provides the foundation for all metadata extraction in the EVE pipeline.
It defines shared utilities and patterns that eliminate code duplication across
different extractor types (PDF, HTML, Scholar, etc.).

Key Features:
- Document format validation
- Standardized field mapping with optional cleaning
- Title extraction with filename fallback
- Author name processing and normalization
- Extraction method tracking
- Consistent metadata finalization and debug logging

Design Pattern:
All extractors inherit from BaseMetadataExtractor and implement the abstract
extract_metadata() method. They can then leverage the shared utility methods
to handle common operations consistently.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from eve.model.document import Document
from eve.logging import get_logger


class BaseMetadataExtractor(ABC):
    """
    Abstract base class for all metadata extractors.
    
    This class provides the common interface and shared utilities that all
    specific extractors (PDF, HTML, Scholar) inherit. It centralizes common
    patterns to reduce code duplication and ensure consistency.
    
    Shared Utilities Provided:
    - Document format validation (_validate_document_format)
    - Field mapping with cleaning (_map_metadata_fields) 
    - Title handling with fallback (_set_title_with_fallback)
    - Author processing (_process_authors)
    - Extraction method tracking (_add_extraction_method)
    - Metadata finalization (_finalize_metadata)
    - Text cleaning (_clean_title)
    - Filename-to-title conversion (_extract_title_from_filename)
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the base metadata extractor.
        
        Sets up logging and debug configuration that will be inherited
        by all specific extractor implementations.
        
        Args:
            debug: Enable debug logging for detailed extraction information
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a document.
        
        This is the main interface that all concrete extractors must implement.
        Each extractor will have its own specific logic for extracting metadata
        from its supported document format, but should use the shared utilities
        provided by this base class for common operations.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata, or None if extraction fails
            
        Note:
            Implementations should call self._finalize_metadata() before returning
            to ensure consistent debug logging and validation.
        """
        pass

    def _validate_document_format(self, document: Document, expected_format: str) -> bool:
        """
        Validate that the document has the expected format.
        
        This replaces the common pattern found in all extractors:
        if document.file_format != "expected":
            self.logger.warning(f"Expected format, got {document.file_format}")
            return None
        
        Args:
            document: Document to validate
            expected_format: Expected file format (e.g., "pdf", "html")
            
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
        
        This eliminates the repetitive pattern of:
        if 'field' in source_data:
            metadata['field'] = source_data['field']
        
        Instead, you can define a mapping dictionary and this method handles
        the iteration and optional cleaning automatically.
        
        Args:
            source_data: Source dictionary containing raw metadata
            field_mapping: Mapping of source_key -> target_key
                          e.g., {'title': 'title', 'pub_year': 'year'}
            target_metadata: Target dictionary to update with mapped fields
            apply_cleaning: List of target field names that should be cleaned 
                          with _clean_title (typically used for title fields)
                          
        Example:
            field_mapping = {'title': 'title', 'author': 'authors', 'year': 'year'}
            self._map_metadata_fields(bib_data, field_mapping, metadata, ['title'])
        """
        apply_cleaning = apply_cleaning or []
        
        for source_key, target_key in field_mapping.items():
            if source_key in source_data:
                value = source_data[source_key]
                # Apply text cleaning for specified fields (usually titles)
                if target_key in apply_cleaning and isinstance(value, str):
                    cleaned_value = self._clean_title(value)
                    if cleaned_value:  # Only set if cleaning was successful
                        target_metadata[target_key] = cleaned_value
                else:
                    target_metadata[target_key] = value

    def _set_title_with_fallback(self, metadata: Dict[str, Any], extracted_title: Optional[str], 
                                document: Document, title_source: str = 'extracted') -> None:
        """
        Set title in metadata with filename fallback if no title is available.
        
        This centralizes the common pattern of:
        if not metadata.get('title'):
            metadata['title'] = self._extract_title_from_filename(document.file_path)
            metadata['title_source'] = 'filename'
        else:
            metadata['title_source'] = 'extracted'
        
        Args:
            metadata: Metadata dictionary to update
            extracted_title: Extracted title (can be None or empty)
            document: Document object for filename fallback
            title_source: Source description for extracted title 
                         (e.g., 'html_tag', 'pdf_metadata', 'extracted')
                         
        Sets:
            metadata['title']: The title (extracted or from filename)
            metadata['title_source']: Where the title came from
        """
        if extracted_title:
            metadata['title'] = extracted_title
            metadata['title_source'] = title_source
        else:
            # Fallback to filename-based title
            metadata['title'] = self._extract_title_from_filename(document.file_path)
            metadata['title_source'] = 'filename'

    def _process_authors(self, authors: Union[str, List[str]]) -> List[str]:
        """
        Process and normalize author information from various formats.
        
        Handles the common patterns found in Scholar extractor:
        - List of author strings
        - Single string with various separators (and, comma, semicolon)
        - Single author string
        
        This eliminates duplicate author processing logic across extractors.
        
        Args:
            authors: Authors as string or list from source metadata
            
        Returns:
            List of cleaned author names (empty list if invalid input)
            
        Examples:
            ['John Doe', 'Jane Smith'] -> ['John Doe', 'Jane Smith']
            'John Doe and Jane Smith' -> ['John Doe', 'Jane Smith'] 
            'John Doe, Jane Smith' -> ['John Doe', 'Jane Smith']
            'John Doe' -> ['John Doe']
        """
        if isinstance(authors, list):
            # Already a list, just clean up empty entries
            return [author.strip() for author in authors if author.strip()]
        elif isinstance(authors, str):
            # Try common separators used in academic publications
            for separator in [' and ', ', ', '; ']:
                if separator in authors:
                    return [author.strip() for author in authors.split(separator) if author.strip()]
            # Single author case
            return [authors.strip()] if authors.strip() else []
        return []

    def _add_extraction_method(self, metadata: Dict[str, Any], method: str) -> None:
        """
        Add extraction method to metadata tracking.
        
        Replaces the repetitive pattern:
        metadata['extraction_methods'] = []
        if condition:
            metadata['extraction_methods'].append('method')
        
        This method handles the list initialization and prevents duplicates.
        
        Args:
            metadata: Metadata dictionary to update
            method: Extraction method name (e.g., 'pdf2bib', 'html_parsing', 'scholar')
            
        Maintains:
            metadata['extraction_methods']: List of methods used for this document
        """
        if 'extraction_methods' not in metadata:
            metadata['extraction_methods'] = []
        if method not in metadata['extraction_methods']:
            metadata['extraction_methods'].append(method)

    def _finalize_metadata(self, metadata: Dict[str, Any], document: Document) -> Optional[Dict[str, Any]]:
        """
        Finalize metadata by adding debug logging and validation.
        
        Replaces the common end-of-extraction pattern:
        if self.debug:
            self.logger.debug(f"Extracted metadata for {document.filename}: {metadata}")
        return metadata if metadata else None
        
        This ensures consistent debug output and handles empty metadata gracefully.
        
        Args:
            metadata: Metadata dictionary to finalize
            document: Document object for logging context
            
        Returns:
            Finalized metadata dictionary, or None if empty
            
        Note:
            All concrete extractors should call this method before returning
            to ensure consistent behavior and debug logging.
        """
        if not metadata:
            return None
            
        if self.debug:
            self.logger.debug(f"Extracted metadata for {document.filename}: {metadata}")
            
        return metadata

    def _clean_title(self, title: str) -> Optional[str]:
        """
        Clean and normalize a title string.
        
        This utility handles common title cleaning operations:
        - Removes leading/trailing whitespace
        - Converts newlines and carriage returns to spaces
        - Collapses multiple spaces into single spaces
        - Filters out invalid titles (too short, numeric-only)
        
        Used by field mapping when apply_cleaning includes 'title' fields.
        
        Args:
            title: Raw title string from extracted metadata
            
        Returns:
            Cleaned title string, or None if title is invalid
            
        Examples:
            "  Title\nwith\nspaces  " -> "Title with spaces"
            "123" -> None (numeric only)
            "ab" -> None (too short)
            "" -> None (empty)
        """
        if not title or not isinstance(title, str):
            return None
            
        # Remove leading/trailing whitespace
        cleaned = title.strip()
        
        # Convert newlines and carriage returns to spaces
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        
        # Collapse multiple spaces into single spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        # Filter out invalid titles
        if len(cleaned) < 3 or cleaned.isdigit():
            return None
            
        return cleaned

    def _extract_title_from_filename(self, file_path: Path) -> str:
        """
        Extract a readable title from filename as fallback.
        
        This is used when no title can be extracted from document content.
        Converts common filename patterns into readable titles:
        - Replaces underscores, hyphens, and dots with spaces
        - Converts to title case
        - Collapses multiple spaces
        
        Args:
            file_path: Path to the file
            
        Returns:
            Human-readable title derived from filename
            
        Examples:
            "research_paper_2023.pdf" -> "Research Paper 2023"
            "my-document.html" -> "My Document"
            "file.with.dots.txt" -> "File With Dots"
        """
        # Get filename without extension
        title = file_path.stem
        
        # Replace common separators with spaces
        title = title.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Collapse multiple spaces
        while '  ' in title:
            title = title.replace('  ', ' ')
        
        # Convert to title case and clean up
        return title.strip().title()
