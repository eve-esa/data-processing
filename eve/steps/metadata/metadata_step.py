"""Metadata extraction step for the EVE pipeline."""

import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.metadata.extractors.pdf_extractor import PdfMetadataExtractor
from eve.steps.metadata.extractors.html_extractor import HtmlMetadataExtractor


class MetadataStep(PipelineStep):
    """
    Metadata extraction step that extracts bibliographic and content metadata 
    from PDF and HTML documents.
    
    Supported formats:
    - PDF: Uses pdf2bib for DOI extraction and pdftitle for title extraction
    - HTML: Extracts title from HTML title tags
    
    Configuration options:
    - enabled_formats: List of formats to process (default: ["pdf", "html"])
    - fallback_to_filename: Whether to use filename as title fallback (default: True)
    - debug: Enable debug logging (default: False)
    - export_metadata: Whether to export metadata to JSON file (default: True)
    - metadata_destination: Directory to save metadata file (default: "./output")
    - metadata_filename: Name of the metadata JSON file (default: "metadata.json")
    """

    def __init__(self, config: dict):
        """
        Initialize the metadata extraction step.
        
        Args:
            config: Configuration dictionary with options:
                - enabled_formats: List of file formats to process
                - fallback_to_filename: Use filename as title fallback
                - debug: Enable debug logging
                - export_metadata: Whether to export metadata to JSON file
                - metadata_destination: Directory to save metadata file
                - metadata_filename: Name of the metadata JSON file
        """
        super().__init__(config)
        
        self.enabled_formats = set(config.get("enabled_formats", ["pdf", "html"]))
        self.fallback_to_filename = config.get("fallback_to_filename", True)
        self.export_metadata = config.get("export_metadata", True)
        self.metadata_destination = Path(config.get("metadata_destination", "./output"))
        self.metadata_filename = config.get("metadata_filename", "metadata.json")
        
        self.extractors = {
            "pdf": PdfMetadataExtractor(debug=self.debug),
            "html": HtmlMetadataExtractor(debug=self.debug)
        }

    async def _extract_metadata_for_document(self, document: Document) -> Document:
        """
        Extract metadata for a single document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Document with metadata added to the metadata field
        """
        if document.file_format not in self.enabled_formats:
            self.logger.debug(f"Skipping metadata extraction for unsupported format: {document.file_format}")
            return document
            
        if document.file_format not in self.extractors:
            self.logger.warning(f"No extractor available for format: {document.file_format}")
            return document
            
        try:
            extractor = self.extractors[document.file_format]
            metadata = await extractor.extract_metadata(document)
            
            if metadata:
                for key, value in metadata.items():
                    if value is not None:
                        document.add_metadata(f"extracted_{key}", value)
                
                document.add_metadata("extracted_metadata", metadata)
                
                self.logger.info(f"Successfully extracted metadata from {document.filename}")
                if self.debug:
                    self.logger.debug(f"Extracted metadata keys: {list(metadata.keys())}")
            else:
                self.logger.warning(f"No metadata extracted from {document.filename}")
                
                if self.fallback_to_filename:
                    title = document.file_path.stem.replace("_", " ").replace("-", " ")
                    document.add_metadata("extracted_title", title)
                    self.logger.info(f"Using filename as title fallback: {title}")
                    
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {document.filename}: {str(e)}")
            
            if self.fallback_to_filename:
                title = document.file_path.stem.replace("_", " ").replace("-", " ")
                document.add_metadata("extracted_title", title)
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
        
        final_result.sort(key=lambda doc: documents.index(doc) if doc in documents else len(documents))
        
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

        metadata_collection = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(documents),
                "supported_formats": list(self.enabled_formats)
            },
            "documents": []
        }

        for document in documents:
            doc_metadata = {
                "filename": document.filename,
                "file_path": str(document.file_path),
                "file_format": document.file_format,
                "content_length": document.content_length,
                "has_extracted_metadata": bool(document.get_metadata("extracted_metadata"))
            }
            
            if document.metadata:
                extracted_metadata = {}
                for key, value in document.metadata.items():
                    if key.startswith('extracted_'):
                        clean_key = key.replace('extracted_', '')
                        extracted_metadata[clean_key] = value
                
                if extracted_metadata:
                    doc_metadata["extracted_metadata"] = extracted_metadata
                
                doc_metadata["full_metadata"] = document.metadata

            metadata_collection["documents"].append(doc_metadata)

        metadata_file = self.metadata_destination / self.metadata_filename
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_collection, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Exported metadata to: {metadata_file}")
            self.logger.info(f"Metadata exported for {len(documents)} documents ({sum(1 for doc in documents if doc.get_metadata('extracted_metadata'))} with extracted metadata)")
            
        except Exception as e:
            self.logger.error(f"Failed to export metadata to {metadata_file}: {str(e)}")

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.enabled_formats)
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about available extractors."""
        return {
            "supported_formats": list(self.enabled_formats),
            "available_extractors": list(self.extractors.keys()),
            "fallback_enabled": self.fallback_to_filename,
            "debug_enabled": self.debug,
            "export_metadata": self.export_metadata,
            "metadata_destination": str(self.metadata_destination),
            "metadata_filename": self.metadata_filename
        }
