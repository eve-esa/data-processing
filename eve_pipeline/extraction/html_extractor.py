"""HTML content extraction using Trafilatura."""

from pathlib import Path
from typing import Any, Dict, Optional

from eve_pipeline.extraction.base import ExtractorBase


class HTMLExtractor(ExtractorBase):
    """HTML content extractor using Trafilatura."""
    
    def __init__(
        self,
        include_comments: bool = False,
        include_tables: bool = True,
        include_links: bool = True,
        **kwargs,
    ) -> None:
        """Initialize HTML extractor.
        
        Args:
            include_comments: Whether to include comments.
            include_tables: Whether to include table content.
            include_links: Whether to include link URLs.
            **kwargs: Additional configuration.
        """
        super().__init__(
            supported_formats=["html", "htm"],
            output_format="markdown",
            **kwargs,
        )
        self.include_comments = include_comments
        self.include_tables = include_tables
        self.include_links = include_links
        
        # Initialize trafilatura
        self._initialize_trafilatura()
    
    def _initialize_trafilatura(self) -> None:
        """Initialize Trafilatura extractor."""
        try:
            import trafilatura
            self.trafilatura = trafilatura
            self.logger.info("Initialized Trafilatura HTML extractor")
        except ImportError as e:
            raise ImportError(f"Trafilatura not available: {e}")
    
    def extract_content(self, file_path: Path) -> str:
        """Extract content from HTML file.
        
        Args:
            file_path: Path to HTML file.
            
        Returns:
            Extracted content as markdown.
        """
        try:
            # Read HTML content with encoding detection
            html_content = self._read_html_file(file_path)
            
            # Extract using Trafilatura
            extracted_text = self.trafilatura.extract(
                html_content,
                include_comments=self.include_comments,
                include_tables=self.include_tables,
                include_links=self.include_links,
                output_format="markdown",
            )
            
            if not extracted_text:
                # Fallback to basic extraction
                extracted_text = self._fallback_extraction(html_content)
            
            if not extracted_text:
                raise RuntimeError("No content extracted from HTML")
            
            # Add header
            header = self._create_markdown_header(file_path)
            return header + extracted_text
            
        except Exception as e:
            self.logger.error(f"HTML extraction failed: {e}")
            raise
    
    def _read_html_file(self, file_path: Path) -> str:
        """Read HTML file with encoding detection.
        
        Args:
            file_path: Path to HTML file.
            
        Returns:
            HTML content as string.
        """
        # Try multiple encodings
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise RuntimeError(f"Cannot decode HTML file {file_path}")
    
    def _fallback_extraction(self, html_content: str) -> str:
        """Fallback extraction using BeautifulSoup.
        
        Args:
            html_content: HTML content.
            
        Returns:
            Extracted text content.
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available for fallback extraction")
            return ""
        except Exception as e:
            self.logger.warning(f"Fallback extraction failed: {e}")
            return ""
    
    def get_html_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get HTML metadata.
        
        Args:
            file_path: Path to HTML file.
            
        Returns:
            Dictionary with metadata.
        """
        try:
            html_content = self._read_html_file(file_path)
            
            # Extract metadata using trafilatura
            metadata = self.trafilatura.extract_metadata(html_content)
            
            result = {
                "file_size": file_path.stat().st_size,
                "content_length": len(html_content),
            }
            
            if metadata:
                result.update({
                    "title": metadata.title or "",
                    "author": metadata.author or "",
                    "date": str(metadata.date) if metadata.date else "",
                    "description": metadata.description or "",
                    "sitename": metadata.sitename or "",
                    "url": metadata.url or "",
                })
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to extract HTML metadata: {e}")
            return {"file_size": file_path.stat().st_size}
    
    def extract_links(self, file_path: Path) -> list[str]:
        """Extract all links from HTML file.
        
        Args:
            file_path: Path to HTML file.
            
        Returns:
            List of URLs found in the HTML.
        """
        try:
            from bs4 import BeautifulSoup
            import urllib.parse
            
            html_content = self._read_html_file(file_path)
            soup = BeautifulSoup(html_content, "html.parser")
            
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Convert relative URLs to absolute if possible
                if href.startswith("http"):
                    links.append(href)
                elif href.startswith("/"):
                    # Need base URL to convert relative links
                    links.append(href)
                elif href.startswith("#"):
                    # Skip anchor links
                    continue
                else:
                    links.append(href)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            self.logger.warning(f"Failed to extract links: {e}")
            return []