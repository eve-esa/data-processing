from typing import Dict, Any, Optional
from urllib.parse import urlparse

from eve.model.document import Document
from eve.steps.metadata.extractors.base_extractor import BaseMetadataExtractor
from eve.common.regex_patterns import (
    extract_html_title,
    extract_html_meta_tags,
    extract_json_ld_count
)


class HtmlMetadataExtractor(BaseMetadataExtractor):
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
        super().__init__(debug)
    
    def _extract_content_with_tags(self, document: Document) -> Optional[str]:
        with open(document.file_path, 'r', encoding = 'utf-8') as file:
            html_content = file.read()

        document.content = html_content
        return document

    def _extract_title_from_html(self, html_content: str) -> Optional[str]:
        """
        Extract title from HTML <title> tag using regex patterns.
        
        This method uses regex patterns from eve.common.regex_patterns to
        efficiently extract the title without parsing the entire HTML document.
        The extracted title is cleaned using the base class _clean_title method.
        
        Args:
            html_content: Raw HTML content as string
            
        Returns:
            Cleaned title string from <title> tag, or None if not found/invalid
            
        Note:
            The regex patterns handle various HTML formatting including:
            - Nested tags within title
            - Whitespace normalization
            - Character encoding issues
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

    def _extract_meta_tags(self, html_content: str) -> Dict[str, str]:
        """
        Extract metadata from HTML meta tags using regex patterns.
        
        Parses various types of meta tags including:
        - Standard meta tags: name="description", name="keywords", etc.
        - OpenGraph tags: property="og:title", property="og:description", etc.
        - Twitter Card tags: name="twitter:title", name="twitter:description", etc.
        - Other proprietary meta tag formats
        
        Args:
            html_content: Raw HTML content as string
            
        Returns:
            Dictionary containing extracted meta tag information with keys like:
            - description: Meta description content
            - keywords: Meta keywords content
            - author: Meta author content
            - og_title: OpenGraph title
            - og_description: OpenGraph description
            - twitter_title: Twitter Card title
            - etc.
            
        Note:
            The regex patterns in eve.common.regex_patterns handle the parsing
            and return a standardized dictionary format.
        """
        return extract_html_meta_tags(html_content)

    def _extract_structured_data(self, html_content: str) -> Dict[str, Any]:
        """
        Extract structured data information from HTML.
        
        Currently detects:
        - JSON-LD scripts (schema.org structured data)
        - Count of structured data elements found
        
        Future enhancements could include:
        - Microdata parsing
        - RDFa parsing
        - Schema.org type detection
        
        Args:
            html_content: Raw HTML content as string
            
        Returns:
            Dictionary containing structured data information:
            - json_ld_count: Number of JSON-LD script tags found
            
        Note:
            This provides metadata about the presence of structured data
            rather than parsing the actual structured data content.
        """
        structured_data = {}
        
        # Count JSON-LD structured data scripts
        json_ld_count = extract_json_ld_count(html_content)
        if json_ld_count > 0:
            structured_data['json_ld_count'] = json_ld_count

        return structured_data

    def _get_url_info(self, document: Document) -> Dict[str, str]:
        """
        Extract URL information from document metadata.
        
        Attempts to find URL information in the document's metadata fields
        and parse it to extract useful components like domain and scheme.
        This is useful for web-scraped content where the source URL provides
        context about the content.
        
        Args:
            document: Document object with potential URL metadata
            
        Returns:
            Dictionary containing URL components:
            - url: Full URL string
            - domain: Domain name (e.g., 'example.com')
            - scheme: URL scheme (e.g., 'https', 'http')
            
            Returns empty dict if no URL found or parsing fails.
            
        Note:
            Looks for URL in metadata fields: 'url' or 'source_url'
        """
        url_info = {}
        
        # Try to find URL in document metadata
        url = document.get_metadata('url') or document.get_metadata('source_url')
        
        if url and isinstance(url, str):
            try:
                # Parse URL to extract components
                parsed = urlparse(url)
                url_info.update({
                    'url': url,                    # Full URL
                    'domain': parsed.netloc,       # Domain name
                    'scheme': parsed.scheme        # http/https
                })
            except Exception as e:
                self.logger.debug(f"Failed to parse URL {url}: {e}")
        
        return url_info

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from an HTML document using multi-source approach.
        
        Args:
            document: HTML document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata with fields:
            - title: Page title (from various sources, with title_source indicator)
            - title_source: Source of title ('html_tag', 'meta_tag', 'filename')
            - meta_tags: Dictionary of all meta tag content
            - structured_data: Information about structured data presence
            - url: Source URL if available
            - domain: Domain name from URL
            - scheme: URL scheme (http/https)
            - content_length: Length of HTML content
            - has_content: Boolean indicating content exists
            - extraction_methods: List containing 'html_parsing'
            
            Returns None if document format is invalid
        """
        # Step 1: Validate document format
        if not self._validate_document_format(document, "html"):
            return None

        metadata = {}

        document = self._extract_content_with_tags(document) # do this because extraction from previous step removes tag

        # Step 2: Extract title from HTML <title> tag
        extracted_title = self._extract_title_from_html(document.content)
        title_source = 'html_tag' if extracted_title else None
        
        # Step 3: Extract meta tags (OpenGraph, Twitter Cards, standard meta tags)
        meta_data = self._extract_meta_tags(document.content)
        if meta_data:
            metadata['meta_tags'] = meta_data
            
            # Step 4: Try alternative title sources if HTML title not found
            if not extracted_title:
                # Try OpenGraph title first, then Twitter Card title
                alt_title = meta_data.get('og_title') or meta_data.get('twitter_title')
                if alt_title:
                    cleaned_alt_title = self._clean_title(alt_title)
                    if cleaned_alt_title:
                        extracted_title = cleaned_alt_title
                        title_source = 'meta_tag'

        # Step 5: Set title with filename fallback using base class utility
        self._set_title_with_fallback(metadata, extracted_title, document, title_source or 'extracted')

        # Step 6: Extract structured data information
        structured_data = self._extract_structured_data(document.content)
        if structured_data:
            metadata['structured_data'] = structured_data

        # Step 7: Extract URL information from document metadata
        url_info = self._get_url_info(document)
        if url_info:
            # Merge URL info directly into metadata
            metadata.update(url_info)

        # Step 8: Add content analysis information
        content_length = len(document.content)
        metadata['content_length'] = content_length
        metadata['has_content'] = content_length > 0

        # Step 9: Track extraction method used
        self._add_extraction_method(metadata, 'html_parsing')

        # Step 10: Finalize with debug logging and validation
        return self._finalize_metadata(metadata, document)
