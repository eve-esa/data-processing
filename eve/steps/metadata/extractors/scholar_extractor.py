"""
Google Scholar metadata extractor using scholarly package.

This extractor searches Google Scholar to find metadata for academic papers by using
progressively longer text snippets from the document content until a unique result is found.

Extracted metadata includes:
- Bibliographic information (title, authors, year, venue, abstract)
- Citation count and publication URLs
- Scholar-specific information
"""

import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

from eve.model.document import Document
from eve.steps.metadata.extractors.base_extractor import BaseMetadataExtractor


class ScholarMetadataExtractor(BaseMetadataExtractor):
    """Metadata extractor using Google Scholar search via scholarly package."""

    def __init__(self, debug: bool = False):
        """
        Initialize the Google Scholar metadata extractor.
        
        Args:
            debug: Enable debug logging
        """
        super().__init__(debug)
        
        # Configuration for search behavior
        self.min_query_length = 1000
        self.max_query_length = 8000
        self.increment_size = 1000
        self.max_iterations = 5

    def _extract_text_snippet(self, document: Document, length: int) -> str:
        """
        Extract a text snippet of specified length from document content.
        
        Args:
            document: Document to extract text from
            length: Desired length of text snippet
            
        Returns:
            Cleaned text snippet for search query
        """
        content = document.content.strip()
        
        if not content:
            self.logger.error("No content available for text snippet extraction")
            return ""
        
        snippet = content[:length]
        
        lines = snippet.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = ' '.join(line.split())
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        result = ' '.join(cleaned_lines)
        
        if len(result) == length and length < len(content):
            last_space = result.rfind(' ')
            if last_space > length * 0.8:
                result = result[:last_space]
        
        return result.strip()

    async def _search_scholar(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Search Google Scholar with the given query.
        
        Args:
            query: Search query string
            
        Returns:
            List of search results or None if search fails
        """
        try:
            from scholarly import scholarly
            
            self.logger.debug(f"Searching Google Scholar with query length: {len(query)}")
            self.logger.info(f"Query text preview: {query[:200]}{'...' if len(query) > 200 else ''}")
            
            def do_search():
                search_query = scholarly.search_pubs(query)
                results = []
                try:
                    for i, result in enumerate(search_query):
                        if i >= 3:
                            break
                        results.append(dict(result))
                except StopIteration:
                    pass
                return results
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, do_search)
            
            self.logger.debug(f"Google Scholar returned {len(results)} results")
            for i, result in enumerate(results):
                self.logger.debug(f"Result {i+1}:")
                bib = result.get('bib', {})
                self.logger.debug(f"  Title: {bib.get('title', 'N/A')}")
            
            return results
            
        except ImportError:
            self.logger.error("scholarly package not available")
            return None
        except Exception as e:
            self.logger.error(f"Google Scholar search failed: {str(e)}")
            return None

    def _extract_scholar_metadata(self, scholar_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from Google Scholar result.
        
        Args:
            scholar_result: Raw result from scholarly search
            
        Returns:
            Cleaned metadata dictionary
        """
        metadata = {}
        
        bib = scholar_result.get('bib', {})
        
        # Map bibliographic fields
        bib_mapping = {
            'title': 'title',
            'pub_year': 'year',
            'venue': 'venue', 
            'abstract': 'abstract'
        }
        self._map_metadata_fields(bib, bib_mapping, metadata, apply_cleaning=['title'])
        
        # Convert year to string if present
        if 'year' in metadata:
            metadata['year'] = str(metadata['year'])
            
        # Process authors
        if bib.get('author'):
            metadata['authors'] = self._process_authors(bib['author'])
                
        # Map scholar-specific fields
        scholar_mapping = {
            'num_citations': 'citation_count',
            'pub_url': 'publication_url',
            'eprint_url': 'pdf_url',
            'filled': 'scholar_filled'
        }
        self._map_metadata_fields(scholar_result, scholar_mapping, metadata)
        
        # Convert citation count to int if present
        if 'citation_count' in metadata:
            metadata['citation_count'] = int(metadata['citation_count'])
        
        return metadata

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a document using Google Scholar search.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata or None if extraction fails
        """
        if not document.content.strip():
            self.logger.warning("No content available for Google Scholar search")
            return None

        query_length = self.min_query_length
        unique_result = None
        search_iterations = 0
        
        while search_iterations < self.max_iterations and query_length <= self.max_query_length:
            search_iterations += 1
            
            query = self._extract_text_snippet(document, query_length)
            
            if not query:
                self.logger.warning("Could not extract meaningful text for Google Scholar search")
                return None
            
            self.logger.debug(f"Scholar search iteration {search_iterations}, query length: {len(query)}")
            
            results = await self._search_scholar(query)
            
            if results is None:
                self.logger.error("Google Scholar search failed")
                return None
            
            if len(results) == 0:
                self.logger.info("No results found in Google Scholar")
                return None
            
            if len(results) == 1:
                unique_result = results[0]
                self.logger.info(f"Found unique Google Scholar result after {search_iterations} iterations with query length {query_length}")
                bib = unique_result.get('bib', {})
                self.logger.info(f"Unique result title: {bib.get('title', 'N/A')}")
                break
            
            if len(results) > 1:
                self.logger.info(f"Found {len(results)} results in iteration {search_iterations}, increasing query length to {query_length + self.increment_size}")
                for i, result in enumerate(results[:2]):
                    bib = result.get('bib', {})
                    self.logger.debug(f"  Multiple result {i+1}: {bib.get('title', 'N/A')}")
                query_length += self.increment_size
                continue
        
        if unique_result is None:
            if search_iterations >= self.max_iterations:
                self.logger.warning(f"Could not find unique result after {self.max_iterations} iterations")
            else:
                self.logger.warning("Could not find unique result - query too long")
            return None
        
        try:
            metadata = self._extract_scholar_metadata(unique_result)
            
            self.logger.info(f"Successfully extracted metadata from unique Google Scholar result")
            if self.debug:
                self.logger.debug(f"Raw Scholar result keys: {list(unique_result.keys())}")
                self.logger.debug(f"Extracted metadata keys: {list(metadata.keys()) if metadata else 'None'}")
            
            return self._finalize_metadata(metadata, document)
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from Scholar result: {str(e)}")
            return None
