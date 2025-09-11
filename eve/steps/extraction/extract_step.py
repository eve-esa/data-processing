from typing import List, Union, Tuple
from pathlib import Path

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.extraction.htmls import HtmlExtractor
from eve.steps.extraction.xmls import XmlExtractor
from eve.steps.extraction.pdfs import PdfExtractor

class ExtractionStep(PipelineStep):
    async def _html_extraction(self, document: Document) -> Document:
        html_extractor = HtmlExtractor(document)
        text = await html_extractor.extract_text()
        # Update the document with extracted text
        document.update_content(text)
        return document
    
    async def _pdf_extraction(self, document: Document, url: str) -> Document:
        pdf_extractor = PdfExtractor(document, url)
        text = await pdf_extractor.extract_text()
        # Update the document with extracted text
        document.update_content(text)
        return document
    
    async def _xml_extraction(self, document: Document) -> Document:
        xml_extractor = XmlExtractor(document)
        text = await xml_extractor.extract_text()
        # Update the document with extracted text
        document.update_content(text)
        return document

    async def execute(self, input_data: Union[List[Document], List[str], List[Path], List[Tuple[Path, str]]]) -> List[Document]:
        """Execute text extraction on input files or documents.
        
        Args:
            input_data: List of file paths, Document objects, or tuples to extract text from
            
        Returns:
            List of Document objects with extracted text
        """
        # Convert input to Document objects if needed
        documents = []
        if input_data and isinstance(input_data[0], str):
            # Convert string paths to Documents
            documents = [Document.from_path_and_content(Path(item), "") for item in input_data]
        elif input_data and isinstance(input_data[0], Path):
            # Convert Path objects to Documents
            documents = [Document.from_path_and_content(item, "") for item in input_data]
        elif input_data and isinstance(input_data[0], tuple):
            # Convert tuples to Documents  
            documents = [Document.from_tuple(item) for item in input_data]
        else:
            # Already Document objects
            documents = input_data or []
        
        config_format = self.config.get("format", None)  # Override format if provided in config
        
        if config_format:
            # Use the format specified in config for all documents
            extraction_formats = {config_format}
        else:
            # Use the individual document formats
            extraction_formats = {document.file_format for document in documents}
        
        self.logger.info(f"Extracting text from {extraction_formats} files. File count: {len(documents)}")

        result = []
        for document in documents:
            try:
                # Determine which format to use for extraction
                format_to_use = config_format if config_format else document.file_format
                
                if format_to_use == "html":
                    document_with_text = await self._html_extraction(document)
                elif format_to_use == "pdf":
                    url = self.config.get("url", None)
                    document_with_text = await self._pdf_extraction(document, url)
                elif format_to_use == "xml":
                    document_with_text = await self._xml_extraction(document)
                else:
                    self.logger.error(f"Unsupported format: {format_to_use}")
                    raise ValueError(f"Unsupported format: {format_to_use}")
                
                result.append(document_with_text)
                if document_with_text.content_length > 1:
                    self.logger.info(f"Successfully extracted {document_with_text.content_length} characters from {document_with_text.filename}")
                else:
                    self.logger.warning(f"No text extracted from {document_with_text.filename}")
            except Exception as e:
                self.logger.error(f"Failed to extract text from {document.filename}: {str(e)}")
                continue
        return result