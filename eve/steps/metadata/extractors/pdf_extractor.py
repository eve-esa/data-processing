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
        if document.file_format != "pdf":
            self.logger.warning(f"Expected PDF format, got {document.file_format}")
            return None

        file_path = str(document.file_path)
        metadata = {}

        bibtex_data = await self._extract_bibtex_metadata(file_path)
        if bibtex_data:
            metadata.update({
                'identifier': bibtex_data.get('identifier'),
                'identifier_type': bibtex_data.get('identifier_type'),
                'validation_info': bibtex_data.get('validation_info'),
                'method': bibtex_data.get('method'),
                'bibtex': bibtex_data.get('bibtex')
            })
            
            bib_metadata = bibtex_data.get('metadata')
            if bib_metadata and isinstance(bib_metadata, dict):
                if 'title' in bib_metadata:
                    metadata['title'] = self._clean_title(bib_metadata['title'])
                if 'author' in bib_metadata:
                    metadata['authors'] = bib_metadata['author']
                if 'year' in bib_metadata:
                    metadata['year'] = bib_metadata['year']
                if 'journal' in bib_metadata:
                    metadata['journal'] = bib_metadata['journal']
                if 'doi' in bib_metadata:
                    metadata['doi'] = bib_metadata['doi']

        if not metadata.get('title'):
            title = await self._extract_title_from_pdf(file_path)
            if title:
                metadata['title'] = title

        if not metadata.get('title'):
            metadata['title'] = self._extract_title_from_filename(document.file_path)
            metadata['title_source'] = 'filename'
        else:
            metadata['title_source'] = 'extracted'

        metadata['extraction_methods'] = []
        if bibtex_data:
            metadata['extraction_methods'].append('pdf2bib')
        metadata['extraction_methods'].append('pdf_reader')

        if self.debug:
            self.logger.debug(f"Extracted metadata for {document.filename}: {metadata}")

        return metadata if metadata else None
