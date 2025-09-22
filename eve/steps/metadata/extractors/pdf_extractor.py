"""PDF metadata extractor using pdf2bib and pdfplumber."""

from typing import Dict, Any, Optional
from pathlib import Path

from eve.model.document import Document
from eve.steps.metadata.extractors.base_extractor import BaseMetadataExtractor


class PdfMetadataExtractor(BaseMetadataExtractor):
    """Metadata extractor for PDF files using pdf2bib and pdfplumber."""

    def __init__(self, debug: bool = False):
        """
        Initialize the PDF metadata extractor.
        
        Args:
            debug: Enable debug logging
        """
        super().__init__(debug)
        
        import pdf2bib
        pdf2bib.config.set('verbose', False)

    async def _extract_bibtex_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata using pdf2bib library.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing bibtex metadata or None if extraction fails
        """
        try:
            import pdf2bib
            
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
        Extract title from PDF using pdfplumber or PyPDF2 as fallback.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted title or None if extraction fails
        """
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                if pdf.metadata and pdf.metadata.get('Title'):
                    title = pdf.metadata['Title']
                    cleaned_title = self._clean_title(title)
                    if cleaned_title:
                        self.logger.debug(f"Extracted title from PDF metadata (pdfplumber): {cleaned_title}")
                        return cleaned_title
                
                if len(pdf.pages) > 0:
                    first_page = pdf.pages[0]
                    text = first_page.extract_text()
                    
                    if text:
                        lines = text.split('\n')[:5]
                        for line in lines:
                            cleaned_line = self._clean_title(line.strip())
                            if cleaned_line and len(cleaned_line) > 10:
                                self.logger.debug(f"Extracted title from first page (pdfplumber): {cleaned_line}")
                                return cleaned_line
                                
        except Exception as e:
            self.logger.debug(f"pdfplumber failed for {file_path}: {str(e)}, trying PyPDF2 fallback")
            
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata and pdf_reader.metadata.get('/Title'):
                    title = pdf_reader.metadata['/Title']
                    cleaned_title = self._clean_title(title)
                    if cleaned_title:
                        self.logger.debug(f"Extracted title from PDF metadata (PyPDF2): {cleaned_title}")
                        return cleaned_title
                
                if len(pdf_reader.pages) > 0:
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    
                    if text:
                        lines = text.split('\n')[:5]
                        for line in lines:
                            cleaned_line = self._clean_title(line.strip())
                            if cleaned_line and len(cleaned_line) > 10:
                                self.logger.debug(f"Extracted title from first page (PyPDF2): {cleaned_line}")
                                return cleaned_line
                                
        except Exception as e:
            self.logger.error(f"Both pdfplumber and PyPDF2 failed for {file_path}: {str(e)}")
            
        return None

    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a PDF document.
        
        Args:
            document: PDF document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata
        """
        if not self._validate_document_format(document, "pdf"):
            return None

        file_path = str(document.file_path)
        metadata = {}

        # Extract bibtex metadata
        bibtex_data = await self._extract_bibtex_metadata(file_path)
        if bibtex_data:
            # Map direct bibtex fields
            direct_mapping = {
                'identifier': 'identifier',
                'identifier_type': 'identifier_type', 
                'validation_info': 'validation_info',
                'method': 'method',
                'bibtex': 'bibtex'
            }
            self._map_metadata_fields(bibtex_data, direct_mapping, metadata)
            
            # Process bibliographic metadata
            bib_metadata = bibtex_data.get('metadata')
            if bib_metadata and isinstance(bib_metadata, dict):
                bib_mapping = {
                    'title': 'title',
                    'year': 'year', 
                    'journal': 'journal',
                    'doi': 'doi'
                }
                self._map_metadata_fields(bib_metadata, bib_mapping, metadata, apply_cleaning=['title'])
                
                # Process authors
                if 'author' in bib_metadata:
                    metadata['authors'] = bib_metadata['author']

        # Handle title extraction with fallback
        extracted_title = metadata.get('title')
        if not extracted_title:
            extracted_title = await self._extract_title_from_pdf(file_path)
            
        self._set_title_with_fallback(metadata, extracted_title, document)

        # Add extraction methods
        self._add_extraction_method(metadata, 'pdf_reader')
        if bibtex_data:
            self._add_extraction_method(metadata, 'pdf2bib')

        return self._finalize_metadata(metadata, document)
