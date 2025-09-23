import asyncio
from typing import Dict, Any, Optional, List

from eve.model.document import Document
from eve.steps.metadata.extractors.base_extractor import BaseMetadataExtractor


class ScholarMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor using Google Scholar search with iterative query refinement.
    
    This extractor implements a sophisticated search strategy to find academic papers
    by progressively refining search queries until a unique result is found.

    """

    def __init__(self, debug: bool = False):
        """
        Initialize the Google Scholar metadata extractor.
        
        Configures the iterative search parameters that control how the extractor
        balances between search precision and recall. Smaller increments provide
        more precision but require more API calls.
        
        Args:
            debug: Enable debug logging for detailed search information
        """
        super().__init__(debug)
        
        # Configuration for iterative search behavior
        self.min_query_length = 1000    # Start with 1KB of text
        self.max_query_length = 8000    # Max 8KB of text (API limits)
        self.increment_size = 1000      # Expand by 1KB each iteration
        self.max_iterations = 5         # Prevent infinite loops

    def _extract_text_snippet(self, document: Document, length: int) -> str:
        """
        Extract a cleaned text snippet of specified length from document content.
        
        This method creates search queries by extracting meaningful text from the
        document content. The text is cleaned and normalized to improve search
        quality on Google Scholar.
        
        Text Processing Steps:
        1. Extract substring of desired length from document start
        2. Split into lines and clean each line (normalize whitespace)
        3. Filter out empty lines
        4. Join lines back into single search string
        5. Ensure word boundaries (avoid cutting words in half)
        
        Args:
            document: Document to extract text from
            length: Desired length of text snippet (in characters)
            
        Returns:
            Cleaned text snippet optimized for Google Scholar search.
            Returns empty string if no content available.
        """
        content = document.content.strip()
        
        if not content:
            self.logger.error("No content available for text snippet extraction")
            return ""
        
        # Extract initial snippet of desired length
        snippet = content[:length]
        
        # Clean and normalize text line by line
        lines = snippet.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Normalize whitespace (collapse multiple spaces, tabs, etc.)
            cleaned_line = ' '.join(line.split())
            if cleaned_line.strip():  # Only keep non-empty lines
                cleaned_lines.append(cleaned_line)
        
        # Join all cleaned lines into single search string
        result = ' '.join(cleaned_lines)
        
        # Ensure word boundaries - truncate at last complete word if needed
        if len(result) == length and length < len(content):
            last_space = result.rfind(' ')
            # Only truncate if we find a space in the last 20% of the text
            if last_space > length * 0.8:
                result = result[:last_space]
        
        return result.strip()

    async def _search_scholar(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Search Google Scholar with the given query using async execution.
        
        This method handles the Google Scholar API call asynchronously to avoid
        blocking the event loop during potentially slow network requests.
        
        Args:
            query: Search query string (cleaned text snippet from document)
            
        Returns:
            List of search result dictionaries with fields:
            - bib: Bibliographic information (title, authors, year, etc.)
            - num_citations: Citation count
            - pub_url: Publication URL
            - eprint_url: PDF URL if available
            - filled: Whether full metadata was retrieved
            
            Returns None if search fails due to network, API, or dependency issues.
        """
        try:
            from scholarly import scholarly, ProxyGenerator
            
            self.logger.debug(f"Searching Google Scholar with query length: {len(query)}")
            self.logger.info(f"Query text preview: {query[:200]}{'...' if len(query) > 200 else ''}")
            
            def do_search():
                """
                Inner function to perform synchronous Scholar search.
                This runs in a thread pool to maintain async compatibility.
                """
                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)
                search_query = scholarly.search_pubs(query)
                results = []
                try:
                    # Limit to top 3 results for efficiency
                    for i, result in enumerate(search_query):
                        if i >= 3:  # Only need a few results for uniqueness check
                            break
                        results.append(dict(result))  # Convert to dict for JSON serialization
                except StopIteration:
                    # Normal termination when fewer than 3 results available
                    pass
                return results
            
            # Execute search in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, do_search)
            
            # Log search results for debugging
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
        Extract and normalize metadata from a Google Scholar search result.
        
        This method processes the raw Scholar API response and converts it into
        the standardized metadata format used by the EVE pipeline. It handles
        various data types and applies necessary cleaning and normalization.
        
        Processing Steps:
        1. Extract bibliographic data from 'bib' field
        2. Apply title cleaning using base class utilities
        3. Normalize author information to list format
        4. Extract Scholar-specific fields (citations, URLs)
        5. Apply type conversions (year to string, citations to int)
        
        Args:
            scholar_result: Raw result dictionary from scholarly.search_pubs()
                          Contains 'bib' field with bibliographic data and
                          top-level fields for citations, URLs, etc.
        """
        metadata = {}
        
        # Extract nested bibliographic information
        bib = scholar_result.get('bib', {})
        
        # Map standard bibliographic fields with title cleaning
        bib_mapping = {
            'title': 'title',           # Paper title
            'pub_year': 'year',         # Publication year  
            'venue': 'venue',           # Journal/conference name
            'abstract': 'abstract'      # Paper abstract
        }
        # Apply cleaning to title field to ensure quality
        self._map_metadata_fields(bib, bib_mapping, metadata, apply_cleaning=['title'])
        
        # Ensure year is stored as string for consistency
        if 'year' in metadata:
            metadata['year'] = str(metadata['year'])
            
        # Process authors using base class utility for normalization
        if bib.get('author'):
            metadata['authors'] = self._process_authors(bib['author'])
                
        # Map Scholar-specific fields from top-level result
        scholar_mapping = {
            'num_citations': 'citation_count',     # Citation count
            'pub_url': 'publication_url',          # Link to paper
            'eprint_url': 'pdf_url',              # Link to PDF
            'filled': 'scholar_filled'             # Full metadata retrieved?
        }
        self._map_metadata_fields(scholar_result, scholar_mapping, metadata)
        
        # Ensure citation count is stored as integer
        if 'citation_count' in metadata:
            metadata['citation_count'] = int(metadata['citation_count'])
        
        return metadata

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a document using iterative Google Scholar search.
        
        This method implements a sophisticated search strategy to find academic papers
        by progressively refining search queries until a unique result is found.
        
        Args:
            document: Document to extract metadata from (must have text content)
            
        """
        # Validate document has content for search
        if not document.content.strip():
            self.logger.warning("No content available for Google Scholar search")
            return None

        # Initialize iterative search variables
        query_length = self.min_query_length  # Start with 1KB
        unique_result = None
        search_iterations = 0
        
        # Iterative search loop: expand query until unique result found
        while search_iterations < self.max_iterations and query_length <= self.max_query_length:
            search_iterations += 1
            
            # Extract text snippet of current length for search query
            query = self._extract_text_snippet(document, query_length)
            
            if not query:
                self.logger.warning("Could not extract meaningful text for Google Scholar search")
                return None
            
            self.logger.debug(f"Scholar search iteration {search_iterations}, query length: {len(query)}")
            
            # Perform Scholar search with current query
            results = await self._search_scholar(query)
            
            # Handle search failure
            if results is None:
                self.logger.error("Google Scholar search failed")
                return None
            
            # Handle no results found
            if len(results) == 0:
                self.logger.info("No results found in Google Scholar")
                return None
            
            # SUCCESS: Unique result found!
            if len(results) == 1:
                unique_result = results[0]
                self.logger.info(f"Found unique Google Scholar result after {search_iterations} iterations with query length {query_length}")
                bib = unique_result.get('bib', {})
                self.logger.info(f"Unique result title: {bib.get('title', 'N/A')}")
                break
            
            # AMBIGUOUS: Multiple results found, need longer query
            if len(results) > 1:
                self.logger.info(f"Found {len(results)} results in iteration {search_iterations}, increasing query length to {query_length + self.increment_size}")
                # Log first few results for debugging
                for i, result in enumerate(results[:2]):
                    bib = result.get('bib', {})
                    self.logger.debug(f"  Multiple result {i+1}: {bib.get('title', 'N/A')}")
                
                # Expand query length for next iteration
                query_length += self.increment_size
                continue
        
        # Check if search was successful
        if unique_result is None:
            if search_iterations >= self.max_iterations:
                self.logger.warning(f"Could not find unique result after {self.max_iterations} iterations")
            else:
                self.logger.warning("Could not find unique result - query too long")
            return None
        
        # Extract metadata from the unique Scholar result
        try:
            metadata = self._extract_scholar_metadata(unique_result)
            
            self.logger.info(f"Successfully extracted metadata from unique Google Scholar result")
            if self.debug:
                self.logger.debug(f"Raw Scholar result keys: {list(unique_result.keys())}")
                self.logger.debug(f"Extracted metadata keys: {list(metadata.keys()) if metadata else 'None'}")
            
            # Finalize using base class utility
            return self._finalize_metadata(metadata, document)
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from Scholar result: {str(e)}")
            return None
