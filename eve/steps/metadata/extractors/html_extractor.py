"""HTML metadata extractor."""

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
    """Metadata extractor for HTML files."""

    def __init__(self, debug: bool = False):
        """
        Initialize the HTML metadata extractor.
        
        Args:
            debug: Enable debug logging
        """
        super().__init__(debug)

    def _extract_title_from_html(self, html_content: str) -> Optional[str]:
        """
        Extract title from HTML content using regex.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Extracted title or None if not found
        """
        title = extract_html_title(html_content)
        
        if title:
            cleaned_title = self._clean_title(title)
            
            if cleaned_title:
                self.logger.debug(f"Extracted title from HTML: {cleaned_title}")
                return cleaned_title

        return None

    def _extract_meta_tags(self, html_content: str) -> Dict[str, str]:
        """
        Extract metadata from HTML meta tags.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Dictionary containing extracted meta tag information
        """
        return extract_html_meta_tags(html_content)

    def _extract_structured_data(self, html_content: str) -> Dict[str, Any]:
        """
        Extract structured data (JSON-LD, microdata) from HTML.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Dictionary containing structured data
        """
        structured_data = {}
        
        json_ld_count = extract_json_ld_count(html_content)
        if json_ld_count > 0:
            structured_data['json_ld_count'] = json_ld_count

        return structured_data

    def _get_url_info(self, document: Document) -> Dict[str, str]:
        """
        Extract URL information if available in document metadata.
        
        Args:
            document: Document object
            
        Returns:
            Dictionary containing URL information
        """
        url_info = {}
        
        url = document.get_metadata('url') or document.get_metadata('source_url')
        
        if url and isinstance(url, str):
            try:
                parsed = urlparse(url)
                url_info.update({
                    'url': url,
                    'domain': parsed.netloc,
                    'scheme': parsed.scheme
                })
            except Exception as e:
                self.logger.debug(f"Failed to parse URL {url}: {e}")
        
        return url_info

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from an HTML document.
        
        Args:
            document: HTML document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata
        """
        if document.file_format != "html":
            self.logger.warning(f"Expected HTML format, got {document.file_format}")
            return None

        metadata = {}

        title = self._extract_title_from_html(document.content)
        if title:
            metadata['title'] = title
            metadata['title_source'] = 'html_tag'
        else:
            metadata['title'] = self._extract_title_from_filename(document.file_path)
            metadata['title_source'] = 'filename'

        meta_data = self._extract_meta_tags(document.content)
        if meta_data:
            metadata['meta_tags'] = meta_data
            
            if metadata['title_source'] == 'filename':
                alt_title = meta_data.get('og_title') or meta_data.get('twitter_title')
                if alt_title:
                    cleaned_alt_title = self._clean_title(alt_title)
                    if cleaned_alt_title:
                        metadata['title'] = cleaned_alt_title
                        metadata['title_source'] = 'meta_tag'

        structured_data = self._extract_structured_data(document.content)
        if structured_data:
            metadata['structured_data'] = structured_data

        url_info = self._get_url_info(document)
        if url_info:
            metadata.update(url_info)

        content_length = len(document.content)
        metadata['content_length'] = content_length
        metadata['has_content'] = content_length > 0

        metadata['extraction_methods'] = ['html_parsing']

        if self.debug:
            self.logger.debug(f"Extracted metadata for {document.filename}: {metadata}")

        return metadata if metadata else None
