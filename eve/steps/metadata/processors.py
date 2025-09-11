"""Metadata extraction processors for different file formats."""

import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from eve.model.document import Document

try:
    import pdf2bib
    PDF2BIB_AVAILABLE = True
    pdf2bib.config.set('verbose', False)
except ImportError:
    PDF2BIB_AVAILABLE = False



class MetadataProcessor(ABC):
    """Abstract base class for metadata processors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from a document."""
        pass


class PDFMetadataProcessor(MetadataProcessor):
    """Metadata processor for PDF files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def _get_bib_data(self, filepath: Path) -> Dict[str, Any]:
        """Extract bibliographic data using pdf2bib."""
        if not PDF2BIB_AVAILABLE:
            return {}
        
        try:
            bib = pdf2bib.pdf2bib(str(filepath))
            return bib if bib is not None else {}
        except Exception as e:
            # Log error but don't fail the entire process
            return {'pdf2bib_error': str(e)}
    
    def _extract_title_pdftitle(self, filepath: Path) -> Optional[str]:
        """Extract title using pdftitle command."""
        try:
            result = subprocess.run(
                ['pdftitle', '-p', str(filepath), '--replace-missing-char', 'true'],
                capture_output=True,
                text=True,
                timeout=30  # Add timeout to prevent hanging
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            return None
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # pdftitle not available or failed
            return None
    
    def _create_temp_file(self, document: Document) -> Path:
        """Create a temporary file from document content if needed."""
        if document.file_path.exists():
            return document.file_path
        
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"temp_pdf_{hash(str(document.file_path))}.pdf"
        
        if hasattr(document, 'binary_content') and document.binary_content:
            temp_file.write_bytes(document.binary_content)
            return temp_file
        
        raise FileNotFoundError(f"Cannot access PDF file: {document.file_path}")
    
    async def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from a PDF document."""
        temp_file = None
        try:
            filepath = self._create_temp_file(document)
            temp_file = filepath if filepath != document.file_path else None
            
            metadata = {
                'file_format': 'pdf',
                'extraction_methods': []
            }
            
            # Try to extract bibliographic data using pdf2bib
            bib_data = self._get_bib_data(filepath)
            if bib_data:
                metadata.update({
                    'identifier': bib_data.get('identifier'),
                    'identifier_type': bib_data.get('identifier_type'),
                    'validation_info': bib_data.get('validation_info'),
                    'method': bib_data.get('method'),
                    'metadata': bib_data.get('metadata'),
                    'bibtex': bib_data.get('bibtex'),
                })
                metadata['extraction_methods'].append('pdf2bib')
                
                if bib_data.get('metadata'):
                    return metadata
            
            title = self._extract_title_pdftitle(filepath)
            if title:
                if not metadata.get('metadata'):
                    metadata['metadata'] = {'title': title}
                metadata['extraction_methods'].append('pdftitle')
            
            if not metadata.get('metadata') and not any(metadata.get(k) for k in ['identifier', 'bibtex']):
                metadata['metadata'] = None
                metadata['extraction_methods'] = ['none']
            
            return metadata
            
        except Exception as e:
            return {
                'file_format': 'pdf',
                'error': str(e),
                'extraction_methods': ['error']
            }
        finally:
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass


class HTMLMetadataProcessor(MetadataProcessor):
    """Metadata processor for HTML files."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.download_remote = config.get('download_remote_html', False)
    
    
    def _extract_html_title(self, html_content: str) -> Optional[str]:
        """Extract title from HTML content."""
        try:
            match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                title = ' '.join(title.split())
                return title if title else None
            return None
        except Exception:
            return None
    
    
    async def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from an HTML document."""
        try:
            metadata = {
                'file_format': 'html',
                'extraction_methods': []
            }
            
            html_content = None
            
            if document.content:
                html_content = document.content
                metadata['extraction_methods'].append('document_content')
            
            # Try to read from file if it exists and we don't have content
            if not html_content and document.file_path.exists():
                try:
                    html_content = document.file_path.read_text(encoding='utf-8', errors='ignore')
                    metadata['extraction_methods'].append('file_read')
                except Exception:
                    pass
            
            title = None
            if html_content:
                title = self._extract_html_title(html_content)
            
            # Store results
            metadata.update({
                'title': title,
                'filepath': str(document.file_path),
            })
            
            if not title:
                metadata['extraction_methods'] = ['none']
            
            return metadata
            
        except Exception as e:
            return {
                'file_format': 'html',
                'error': str(e),
                'extraction_methods': ['error']
            }

