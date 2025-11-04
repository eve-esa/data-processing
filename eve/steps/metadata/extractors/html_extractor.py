from typing import Dict, Any, Optional
from urllib.parse import urlparse

from eve.model.document import Document
from eve.common.regex_patterns import extract_html_title
from eve.logging import get_logger

class HtmlMetadataExtractor():
    """
    Metadata extractor for HTML files and web pages.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the HTML metadata extractor.
        
        The HTML extractor relies on regex patterns defined in eve.common.regex_patterns
        for parsing HTML content efficiently without requiring a full HTML parser.
        
        Args:
            debug: Enable debug logging for detailed extraction information
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)
    
    def _clean_title(self, title: str) -> Optional[str]:
        """
        Clean and normalize a title string.
        
        Args:
            title: Raw title string from extracted metadata
            
        Returns:
            Cleaned title string, or None if title is invalid
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
    
    def _extract_content_with_tags(self, document: Document) -> Optional[str]:
        with open(document.file_path, 'r', encoding = 'utf-8') as file:
            html_content = file.read()

        document.content = html_content
        return document

    def _extract_title_from_html(self, html_content: str) -> Optional[str]:
        """
        Extract title from HTML <title> tag using regex patterns.
        
        Args:
            html_content: Raw HTML content as string
            
        Returns:
            Cleaned title string from <title> tag, or None if not found/invalid
        """
        # Use regex pattern to extract title content
        title = extract_html_title(html_content)
        
        if title:
            # Apply standard title cleaning (whitespace, length validation, etc.)
            cleaned_title = self._clean_title(title)
            
            if cleaned_title:
                self.logger.debug(f"Extracted title from HTML: {cleaned_title}")
                return cleaned_title

        return None

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from an HTML document using multi-source approach.
        
        Args:
            document: HTML document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata with fields:
            - title: Page title (from various sources, with title_source indicator)
            - title_source: Source of title ('html_tag', 'meta_tag', 'filename')
            - url: Source URL if available
            - domain: Domain name from URL
            - scheme: URL scheme (http/https)
            - content_length: Length of HTML content
            - has_content: Boolean indicating content exists
            - extraction_methods: List containing 'html_parsing'
            
            Returns None if document format is invalid
        """

        metadata = {}

        document = self._extract_content_with_tags(document) # do this because extraction from previous step removes tag

        extracted_title = self._extract_title_from_html(document.content)
        metadata['title'] = extracted_title
        metadata['content_length'] = len(document.content)

        return metadata
