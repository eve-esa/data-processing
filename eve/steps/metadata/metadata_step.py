"""Metadata extraction step for the EVE pipeline."""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.metadata.processors import PDFMetadataProcessor, HTMLMetadataProcessor

try:
    from eve.steps.metadata.db_logger import DBLogger, Status
    DB_LOGGER_AVAILABLE = True
except ImportError:
    DB_LOGGER_AVAILABLE = False


class MetadataExtractionStep(PipelineStep):
    """Extract metadata from documents and store in database or JSON file."""
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None):
        super().__init__(config, name)
        
        self.processors = {
            'pdf': PDFMetadataProcessor(config),
            'html': HTMLMetadataProcessor(config),
        }
        
        self.use_database = all([
            os.getenv("DB_HOST"),
            os.getenv("DB_USER"), 
            os.getenv("DB_PASSWORD"),
            os.getenv("DB_NAME")
        ]) and DB_LOGGER_AVAILABLE
        
        self.json_output_path = config.get('json_output_path', 'metadata_extraction_results.json')
        
        self.supported_formats = {'pdf', 'html'}
        
        self.logger.info(f"MetadataExtractionStep initialized. Using database: {self.use_database}")
        
    async def _extract_metadata_for_document(self, document: Document) -> Dict[str, Any]:
        """Extract metadata for a single document."""
        
        if document.file_format not in self.supported_formats:
            self.logger.info(f"Skipping {document.filename} - format '{document.file_format}' not supported for metadata extraction")
            return {
                'filepath': str(document.file_path),
                'file_format': document.file_format,
                'status': 'skipped',
                'reason': 'unsupported_format'
            }
        
        try:
            processor = self.processors.get(document.file_format)
            if not processor:
                raise ValueError(f"No processor available for format: {document.file_format}")
            
            self.logger.info(f"Extracting metadata from {document.filename}")
            metadata = await processor.extract_metadata(document)
            
            metadata.update({
                'filepath': str(document.file_path),
                'file_format': document.file_format,
                'status': 'success'
            })
            
            document.add_metadata('extracted_metadata', metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {document.filename}: {str(e)}")
            return {
                'filepath': str(document.file_path),
                'file_format': document.file_format,
                'status': 'error',
                'error': str(e)
            }
    
    async def _store_metadata_database(self, metadata_results: List[Dict[str, Any]]):
        """Store metadata results in database using DBLogger."""
        if not self.use_database:
            self.logger.warning("Database not available, cannot store to database")
            return
            
        schema = {
            'filepath': ('str', True),
            'file_format': ('str', False),
            'status': ('str', False),
            'identifier': ('longstr', False),
            'identifier_type': ('longstr', False),
            'validation_info': ('longstr', False),
            'method': ('longstr', False),
            'metadata': ('json', False),
            'bibtex': ('longstr', False),
            'title': ('longstr', False),
            'url': ('longstr', False),
            'error': ('longstr', False),
            'reason': ('longstr', False),
        }
        
        for result in metadata_results:
            try:
                db_logger = DBLogger('metadata_extraction_pipeline', schema=schema)
                
                db_data = {}
                for key, value in result.items():
                    if key in schema:
                        db_data[key] = value
                
                filepath = result.get('filepath', '')
                file_id = Path(filepath).stem if filepath else None
                
                status = Status.SUCCESS if result.get('status') == 'success' else Status.ERROR
                db_logger.start_logger(id=file_id, data=db_data, status=status)
                
                self.logger.debug(f"Stored metadata for {filepath} in database")
                
            except Exception as e:
                self.logger.error(f"Failed to store metadata in database: {str(e)}")
    
    async def _store_metadata_json(self, metadata_results: List[Dict[str, Any]]):
        """Store metadata results in JSON file."""
        try:
            output_path = Path(self.json_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            existing_data = []
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self.logger.warning(f"Could not read existing JSON file {output_path}, starting fresh")
                    existing_data = []
            
            existing_filepaths = {item.get('filepath') for item in existing_data}
            for result in metadata_results:
                if result.get('filepath') not in existing_filepaths:
                    existing_data.append(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Stored metadata for {len(metadata_results)} files in {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to store metadata in JSON file: {str(e)}")
    
    async def execute(self, input_data: List[Document]) -> List[Document]:
        """Execute metadata extraction on input documents.
        
        Args:
            input_data: List of Document objects to extract metadata from
            
        Returns:
            List of Document objects (unchanged, metadata stored separately)
        """
        documents = input_data or []
        
        if not documents:
            self.logger.warning("No documents provided for metadata extraction")
            return documents
        
        supported_documents = [
            doc for doc in documents 
            if doc.file_format in self.supported_formats
        ]
        
        self.logger.info(f"Extracting metadata from {len(supported_documents)} documents "
                        f"(out of {len(documents)} total documents)")
        
        metadata_tasks = [
            self._extract_metadata_for_document(doc) 
            for doc in supported_documents
        ]
        
        if metadata_tasks:
            metadata_results = await asyncio.gather(*metadata_tasks, return_exceptions=True)
            
            valid_results = []
            for i, result in enumerate(metadata_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Exception during metadata extraction: {str(result)}")
                    doc = supported_documents[i]
                    valid_results.append({
                        'filepath': str(doc.file_path),
                        'file_format': doc.file_format,
                        'status': 'error',
                        'error': str(result)
                    })
                else:
                    valid_results.append(result)
            
            if self.use_database:
                await self._store_metadata_database(valid_results)
            
            await self._store_metadata_json(valid_results)
            
            success_count = sum(1 for r in valid_results if r.get('status') == 'success')
            error_count = sum(1 for r in valid_results if r.get('status') == 'error')
            skipped_count = sum(1 for r in valid_results if r.get('status') == 'skipped')
            
            self.logger.info(f"Metadata extraction completed: {success_count} successful, "
                           f"{error_count} errors, {skipped_count} skipped")
        
        return documents
