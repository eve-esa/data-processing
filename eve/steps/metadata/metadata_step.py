"""
Metadata extraction step for the EVE pipeline.
"""

import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.metadata.extractors.pdf_extractor import PdfMetadataExtractor
from eve.steps.metadata.extractors.html_extractor import HtmlMetadataExtractor
from eve.steps.metadata.extractors.scholar_extractor import ScholarMetadataExtractor


class MetadataStep(PipelineStep):
    """
    Metadata extraction step that extracts bibliographic and content metadata 
    from PDF, HTML, and text documents.
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
                - enable_scholar_search: Whether to use Google Scholar for additional metadata
                  Note: Text formats (txt, md) automatically enable this feature
        """
        super().__init__(config)
        
        self.enabled_formats = set(config.get("enabled_formats", ["pdf", "html", "txt", "md"]))
        self.fallback_to_filename = config.get("fallback_to_filename", True)
        self.export_metadata = config.get("export_metadata", True)
        self.metadata_destination = Path(config.get("metadata_destination", "./output"))
        self.metadata_filename = config.get("metadata_filename", "metadata.json")
        self.enable_scholar_search = config.get("enable_scholar_search", False)
        
        self.extractors = {
            "pdf": PdfMetadataExtractor(debug=self.debug),
            "html": HtmlMetadataExtractor(debug=self.debug)
        }
        
        if self.enable_scholar_search or any(fmt in self.enabled_formats for fmt in ["txt", "md"]):
            self.scholar_extractor = ScholarMetadataExtractor(debug=self.debug)
            if not self.enable_scholar_search:
                self.logger.info("Enabling Google Scholar search automatically for text format support")

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
            - scholar_*: Google Scholar fields (if enabled)
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
                
                if hasattr(self, 'scholar_extractor'):
                    metadata = await self.scholar_extractor.extract_metadata(document)
                    if metadata:
                        self.logger.info(f"Successfully extracted metadata from text document {document.filename} using Google Scholar")
                else:
                    self.logger.warning("Scholar extractor not available for text format extraction. Enable scholar search to extract metadata from text files.")
            
            elif document.file_format in self.extractors:
                extractor = self.extractors[document.file_format]
                metadata = await extractor.extract_metadata(document)
            else:
                self.logger.warning(f"No extractor available for format: {document.file_format}")
            
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
            
            if (self.enable_scholar_search and hasattr(self, 'scholar_extractor')):
                
                try:
                    scholar_metadata = await self.scholar_extractor.extract_metadata(document)
                    
                    if scholar_metadata:
                        for key, value in scholar_metadata.items():
                            if value is not None:
                                document.add_metadata(f"scholar_{key}", value)
                        
                        document.add_metadata("scholar_metadata", scholar_metadata)
                        self.logger.info(f"Successfully extracted Google Scholar metadata from {document.filename}")
                        
                        if self.debug:
                            self.logger.debug(f"Scholar metadata keys: {list(scholar_metadata.keys())}")
                    else:
                        self.logger.info(f"No Google Scholar metadata found for {document.filename}")
                        
                except Exception as scholar_error:
                    self.logger.warning(f"Google Scholar extraction failed for {document.filename}: {str(scholar_error)}")
                    
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
            print(document.metadata)
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
        info = {
            "supported_formats": list(self.enabled_formats),
            "available_extractors": list(self.extractors.keys()),
            "fallback_enabled": self.fallback_to_filename,
            "debug_enabled": self.debug,
            "export_metadata": self.export_metadata,
            "metadata_destination": str(self.metadata_destination),
            "metadata_filename": self.metadata_filename,
            "scholar_search_enabled": self.enable_scholar_search
        }
        
        if self.enable_scholar_search:
            info["available_extractors"].append("google_scholar")
            
        return info
