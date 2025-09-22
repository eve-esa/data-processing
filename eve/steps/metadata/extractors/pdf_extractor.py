"""
PDF metadata extractor using pdf2bib and pdfplumber.

This module handles metadata extraction from PDF documents using multiple approaches:

1. **Primary Method - pdf2bib**: Attempts to extract DOI and bibliographic metadata
   - Uses the pdf2bib library to search for DOI patterns in PDF content
   - Can identify articles and fetch metadata from DOI resolution
   - Provides rich bibliographic data when successful

2. **Fallback Method - Direct PDF Reading**: Extracts basic metadata when pdf2bib fails
   - Uses pdfplumber as primary PDF reader (more reliable)
   - Falls back to PyPDF2 if pdfplumber fails
   - Extracts title from PDF metadata fields or first page content

3. **Final Fallback - Filename**: Uses base class functionality when all else fails

The extractor follows a graceful degradation strategy, attempting the most sophisticated
methods first and falling back to simpler approaches as needed.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from eve.model.document import Document
from eve.steps.metadata.extractors.base_extractor import BaseMetadataExtractor


class PdfMetadataExtractor(BaseMetadataExtractor):
    """
    Metadata extractor for PDF files using multiple extraction strategies.
    
    Extraction Strategy:
    1. Attempt pdf2bib for DOI-based bibliographic metadata
    2. Try direct PDF reading (pdfplumber -> PyPDF2 fallback)
    3. Use filename as title fallback (handled by base class)
    
    Extracted Metadata Fields:
    - title: Document title (from metadata, content, or filename)
    - authors: List of author names (from bibliographic data)
    - year: Publication year
    - journal: Journal or publication venue
    - doi: Digital Object Identifier
    - identifier: Various identifier types (DOI, arXiv, etc.)
    - bibtex: BibTeX citation format
    - extraction_methods: List of methods used ['pdf2bib', 'pdf_reader']
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the PDF metadata extractor.
        
        Sets up pdf2bib configuration to minimize verbose output during extraction.
        
        Args:
            debug: Enable debug logging for detailed extraction information
        """
        super().__init__(debug)
        
        # Configure pdf2bib to reduce verbose output
        import pdf2bib
        pdf2bib.config.set('verbose', False)

    async def _extract_bibtex_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata using pdf2bib library (primary extraction method).
        
        pdf2bib attempts to:
        1. Scan PDF content for DOI patterns
        2. Resolve DOI to fetch bibliographic metadata
        3. Return structured metadata including BibTeX format
        
        This is the preferred method as it can provide rich, accurate metadata
        for academic papers with valid DOIs.
        
        Args:
            file_path: Path to the PDF file to analyze
            
        Returns:
            Dictionary containing structured bibtex metadata with fields:
            - identifier: DOI or other identifier found
            - identifier_type: Type of identifier (e.g., 'doi')
            - metadata: Nested dict with title, author, year, journal, etc.
            - bibtex: Formatted BibTeX citation
            - validation_info: Validation details from DOI resolution
            - method: Method used for extraction
            
            Returns None if no DOI found or extraction fails
        """
        try:
            import pdf2bib
            
            # Attempt to extract DOI and fetch metadata
            bib_data = pdf2bib.pdf2bib(file_path)
            
            if bib_data is None:
                self.logger.debug(f"No bibtex data found in {file_path}")
                return None
                
            self.logger.debug(f"Successfully extracted bibtex data from {file_path}")
            return bib_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract bibtex metadata from {file_path}: {str(e)}")
            return None

    async def _extract_title_from_pdf(self, file_path: str) -> Optional[str]:
        """
        Extract title from PDF using direct PDF reading (fallback method).
        
        This method attempts title extraction when pdf2bib fails:
        
        Strategy 1 - pdfplumber (preferred):
        1. Check PDF metadata fields for title
        2. If no metadata title, scan first 5 lines of first page
        3. Return first line that looks like a title (>10 chars after cleaning)
        
        Strategy 2 - PyPDF2 (fallback):
        1. Same approach as pdfplumber but using PyPDF2 library
        2. Used when pdfplumber fails or is unavailable
        
        Args:
            file_path: Path to the PDF file to analyze
            
        Returns:
            Extracted and cleaned title string, or None if extraction fails
            
        Note:
            This method relies on heuristics and may not work well for
            PDFs with complex layouts or embedded titles.
        """
        # Attempt 1: Use pdfplumber (more reliable for text extraction)
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                # First try: Check PDF metadata for title field
                if pdf.metadata and pdf.metadata.get('Title'):
                    title = pdf.metadata['Title']
                    cleaned_title = self._clean_title(title)
                    if cleaned_title:
                        self.logger.debug(f"Extracted title from PDF metadata (pdfplumber): {cleaned_title}")
                        return cleaned_title
                
                # Second try: Extract from first page content
                if len(pdf.pages) > 0:
                    first_page = pdf.pages[0]
                    text = first_page.extract_text()
                    
                    if text:
                        # Check first 5 lines for title-like content
                        lines = text.split('\n')[:5]
                        for line in lines:
                            cleaned_line = self._clean_title(line.strip())
                            # Heuristic: title should be substantial (>10 chars)
                            if cleaned_line and len(cleaned_line) > 10:
                                self.logger.debug(f"Extracted title from first page (pdfplumber): {cleaned_line}")
                                return cleaned_line
                                
        except Exception as e:
            self.logger.debug(f"pdfplumber failed for {file_path}: {str(e)}, trying PyPDF2 fallback")
            
        # Attempt 2: Use PyPDF2 as fallback
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # First try: Check PDF metadata for title field
                if pdf_reader.metadata and pdf_reader.metadata.get('/Title'):
                    title = pdf_reader.metadata['/Title']
                    cleaned_title = self._clean_title(title)
                    if cleaned_title:
                        self.logger.debug(f"Extracted title from PDF metadata (PyPDF2): {cleaned_title}")
                        return cleaned_title
                
                # Second try: Extract from first page content
                if len(pdf_reader.pages) > 0:
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    
                    if text:
                        # Check first 5 lines for title-like content
                        lines = text.split('\n')[:5]
                        for line in lines:
                            cleaned_line = self._clean_title(line.strip())
                            # Heuristic: title should be substantial (>10 chars)
                            if cleaned_line and len(cleaned_line) > 10:
                                self.logger.debug(f"Extracted title from first page (PyPDF2): {cleaned_line}")
                                return cleaned_line
                                
        except Exception as e:
            self.logger.error(f"Both pdfplumber and PyPDF2 failed for {file_path}: {str(e)}")
            
        return None

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a PDF document using multi-strategy approach.
        
        Extraction Workflow:
        1. **Format Validation**: Ensure document is PDF format
        2. **Primary Extraction**: Attempt pdf2bib for DOI-based metadata
        3. **Metadata Mapping**: Map pdf2bib results to standard fields
        4. **Title Fallback**: Try direct PDF reading if no title from pdf2bib
        5. **Final Fallback**: Use filename if all title extraction fails
        6. **Method Tracking**: Record which extraction methods were used
        7. **Finalization**: Apply debug logging and validation
        
        Args:
            document: PDF document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata with fields:
            - title: Document title (various sources, with title_source indicator)
            - authors: List of author names (from bibliographic data)
            - year: Publication year
            - journal: Publication venue
            - doi: Digital Object Identifier
            - identifier: Document identifier (DOI, arXiv, etc.)
            - bibtex: BibTeX citation format
            - extraction_methods: List of methods used
            
            Returns None if document format is invalid
        """
        # Step 1: Validate document format
        if not self._validate_document_format(document, "pdf"):
            return None

        file_path = str(document.file_path)
        metadata = {}

        # Step 2: Primary extraction using pdf2bib (DOI-based approach)
        bibtex_data = await self._extract_bibtex_metadata(file_path)
        if bibtex_data:
            # Step 3a: Map top-level pdf2bib fields (identifier info, bibtex, etc.)
            direct_mapping = {
                'identifier': 'identifier',           # DOI, arXiv ID, etc.
                'identifier_type': 'identifier_type', # 'doi', 'arxiv', etc.
                'validation_info': 'validation_info', # DOI resolution details
                'method': 'method',                   # pdf2bib extraction method
                'bibtex': 'bibtex'                   # Formatted BibTeX citation
            }
            self._map_metadata_fields(bibtex_data, direct_mapping, metadata)
            
            # Step 3b: Map nested bibliographic metadata
            bib_metadata = bibtex_data.get('metadata')
            if bib_metadata and isinstance(bib_metadata, dict):
                bib_mapping = {
                    'title': 'title',       # Paper title
                    'year': 'year',         # Publication year
                    'journal': 'journal',   # Journal/venue name
                    'doi': 'doi'           # DOI (may differ from identifier)
                }
                # Apply title cleaning to ensure quality
                self._map_metadata_fields(bib_metadata, bib_mapping, metadata, apply_cleaning=['title'])
                
                # Step 3c: Handle author information (may need special processing)
                if 'author' in bib_metadata:
                    metadata['authors'] = bib_metadata['author']

        # Step 4: Title extraction fallback (when pdf2bib doesn't provide title)
        extracted_title = metadata.get('title')
        if not extracted_title:
            # Try direct PDF reading for title
            extracted_title = await self._extract_title_from_pdf(file_path)
            
        # Step 5: Set title with filename fallback using base class utility
        self._set_title_with_fallback(metadata, extracted_title, document)

        # Step 6: Track extraction methods used
        self._add_extraction_method(metadata, 'pdf_reader')  # Always add PDF reading
        if bibtex_data:
            self._add_extraction_method(metadata, 'pdf2bib')  # Add if pdf2bib succeeded

        # Step 7: Finalize with debug logging and validation
        return self._finalize_metadata(metadata, document)
