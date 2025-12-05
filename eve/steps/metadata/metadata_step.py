"""
Metadata extraction step for the EVE pipeline.
"""

import asyncio
import json
from typing import List
from pathlib import Path

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.metadata.extractors.pdf_extractor import PdfMetadataExtractor
from eve.steps.metadata.extractors.html_extractor import HtmlMetadataExtractor

class MetadataStep(PipelineStep):
    """
    Metadata extraction step that extracts metadata from PDF and HTML documents.
    """

    def __init__(self, config: dict):
        """
        Initialize the metadata extraction step.
        
        Args:
            config: Configuration dictionary with options:
                - enabled_formats: List of file formats to process (pdf, html, txt, md)
                - fallback_to_filename: Use filename as title fallback
                - debug: Enable debug logging
                - export_metadata: Whether to export metadata to JSON file
                - metadata_destination: Directory to save metadata file
                - metadata_filename: Name of the metadata JSON file
                  Note: Text formats (txt, md) automatically enable this feature
        """
        super().__init__(config)
        
        self.enabled_formats = set(config.get("enabled_formats", ["pdf", "html", "txt", "md"]))
        self.fallback_to_filename = config.get("fallback_to_filename", True)
        self.export_metadata = config.get("export_metadata", True)
        self.metadata_destination = Path(config.get("metadata_destination", "./output"))
        self.metadata_filename = config.get("metadata_filename", "metadata.jsonl")
        
        self.extractors = {
            "pdf": PdfMetadataExtractor(debug=self.debug),
            "html": HtmlMetadataExtractor(debug=self.debug)
        }

    async def _extract_metadata_for_document(self, document: Document) -> Document:
        """
        Extract metadata for a single document using appropriate extractor.
        Args:
            document: Document to extract metadata from
            
        Returns:
            Document with metadata added to the metadata field:
            - extracted_title: Document title
            - extracted_authors: List of authors
            - extracted_year: Publication year
            - extracted_metadata: Full metadata dictionary
            - extraction_error: Error message (if extraction failed)
        """
        if document.file_format not in self.enabled_formats:
            self.logger.debug(f"Skipping metadata extraction for unsupported format: {document.file_format}")
            return document
            
        metadata = None
        
        try:
            if document.file_format in ["txt", "md"]:
                if not document.content.strip() and document.file_path.exists():
                    try:
                        with open(document.file_path, 'r', encoding='utf-8') as f:
                            document.update_content(f.read())
                        self.logger.info(f"Loaded content for text file {document.filename}: {len(document.content)} characters")
                    except Exception as e:
                        self.logger.error(f"Failed to load content for {document.filename}: {str(e)}")
                        return document
            
            elif document.file_format in self.extractors:
                extractor = self.extractors[document.file_format]
                metadata = await extractor.extract_metadata(document)
            else:
                self.logger.warning(f"No extractor available for format: {document.file_format}")
            
            if metadata:
                document.add_metadata("extracted_metadata", metadata)
                
                self.logger.info(f"Successfully extracted metadata from {document.filename}")
                if self.debug:
                    self.logger.debug(f"Extracted metadata keys: {list(metadata.keys())}")
            else:
                self.logger.warning(f"No metadata extracted from {document.filename}")
                
                if self.fallback_to_filename:
                    title = document.file_path.stem.replace("_", " ").replace("-", " ")
                    document.add_metadata("title", title)
                    self.logger.info(f"Using filename as title fallback: {title}")
                    
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {document.filename}: {str(e)}")
            
            if self.fallback_to_filename:
                title = document.file_path.stem.replace("_", " ").replace("-", " ")
                document.add_metadata("title", title)
            document.add_metadata("extraction_error", str(e))
        
        return document

    async def execute(self, documents: List[Document]) -> List[Document]:
        """
        Execute metadata extraction on input documents.
        
        Args:
            documents: List of Document objects to extract metadata from
            
        Returns:
            List of Document objects with metadata added
        """
        if not documents:
            self.logger.warning("No input documents provided to metadata step")
            return []
        
        supported_documents = [
            doc for doc in documents 
            if doc.file_format in self.enabled_formats
        ]
        
        unsupported_documents = [
            doc for doc in documents 
            if doc.file_format not in self.enabled_formats
        ]
        
        if unsupported_documents:
            self.logger.info(f"Skipping {len(unsupported_documents)} documents with unsupported formats")
        
        if not supported_documents:
            self.logger.warning("No documents with supported formats found")
            return documents
        
        self.logger.info(f"Extracting metadata from {len(supported_documents)} documents")
        
        tasks = [
            self._extract_metadata_for_document(document) 
            for document in supported_documents
        ]
        
        processed_supported = await asyncio.gather(*tasks, return_exceptions=True)
        
        result_supported = []
        for i, result in enumerate(processed_supported):
            if isinstance(result, Exception):
                self.logger.error(f"Exception processing {supported_documents[i].filename}: {result}")
                result_supported.append(supported_documents[i])
            else:
                result_supported.append(result)
        
        final_result = result_supported + unsupported_documents
        
        successful_count = sum(1 for doc in result_supported if doc.get_metadata("extracted_metadata"))
        self.logger.info(f"Successfully extracted metadata from {successful_count}/{len(supported_documents)} supported documents")
        
        if self.export_metadata:
            await self._export_metadata_to_json(final_result)
        
        return final_result

    async def _export_metadata_to_json(self, documents: List[Document]) -> None:
        """
        Export extracted metadata to a JSON file.
        
        Args:
            documents: List of processed documents with metadata
        """
        if not self.metadata_destination.exists():
            self.logger.info(f"Creating metadata destination directory: {self.metadata_destination}")
            self.metadata_destination.mkdir(parents=True, exist_ok=True)
        
        metadata_file = self.metadata_destination / self.metadata_filename

        for document in documents:
            doc_metadata = {
                "filename": document.filename,
                "file_path": str(document.file_path),
                "file_format": document.file_format,
                "content_length": document.content_length,
                "has_extracted_metadata": bool(document.get_metadata("extracted_metadata"))
            }
            if document.metadata:
                for key, value in document.metadata.items():
                    doc_metadata[key] = value

            metadata_file = self.metadata_destination / self.metadata_filename
            
            try:
                with open(metadata_file, 'a', encoding='utf-8') as f:
                    json.dump(doc_metadata, f, ensure_ascii=False, default=str)
                    f.write('\n')
                
                self.logger.info(f"Exported metadata to: {metadata_file}")
                self.logger.info(f"Metadata exported for {len(documents)} documents ({sum(1 for doc in documents if doc.get_metadata('extracted_metadata'))} with extracted metadata)")
                
            except Exception as e:
                self.logger.error(f"Failed to export metadata to {metadata_file}: {str(e)}")

