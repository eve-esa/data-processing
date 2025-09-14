from typing import List

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.extraction.htmls import HtmlExtractor
from eve.steps.extraction.xmls import XmlExtractor
from eve.steps.extraction.pdfs import PdfExtractor

class ExtractionStep(PipelineStep):
    async def _html_extraction(self, document: Document) -> Document:
        html_extractor = HtmlExtractor(document)
        text = await html_extractor.extract_text()
        return text
    
    async def _pdf_extraction(self, document: Document, url: str) -> Document:
        pdf_extractor = PdfExtractor(document, url)
        text = await pdf_extractor.extract_text()
        return text
    
    async def _xml_extraction(self, document: Document) -> Document:
        xml_extractor = XmlExtractor(document)
        text = await xml_extractor.extract_text()
        return text

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute text extraction on input files or documents.
        
        Args:
            input_data: List of file paths or Document objects to extract text from
            
        Returns:
            List of Document objects with extracted text
        """
        format = self.config.get("format", None)  # write a wrapper to find out the extension
        if not format:
            unique_formats = set()

        unique_formats = {document.file_format for document in documents}
        
        self.logger.info(f"Extracting text from {unique_formats} files. File count: {len(documents)}")

        result = []
        for document in documents:
            try:
                if document.file_format == "html":
                    document_with_text = await self._html_extraction(document)
                elif document.file_format == "pdf":
                    url = self.config.get("url", None)
                    if not url:
                        self.logger.error("No URL provided for PDF extraction service")
                    document_with_text = await self._pdf_extraction(document, url)
                elif document.file_format == "xml":
                    document_with_text = await self._xml_extraction(document)
                else:
                    self.logger.error(f"Unsupported format: {document.file_format}")
                    continue
                
                if document_with_text and hasattr(document_with_text, 'content_length') and document_with_text.content_length > 1:
                    result.append(document_with_text)
                    self.logger.info(f"Successfully extracted {document_with_text.content_length} characters from {document_with_text.filename}")
                else:
                    self.logger.warning(f"No text extracted from {document.filename}")
            except Exception as e:
                self.logger.error(f"Failed to extract text from {document.filename}: {str(e)}")
                continue
        return result