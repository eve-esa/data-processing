"""
PII removal step for the EVE pipeline.

This step removes personally identifiable information (PII) from documents
using various configurable approaches including local models and remote servers.
"""

from typing import List, Union, Tuple, Dict, Any
from pathlib import Path

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.utils import normalize_to_documents
from eve.steps.pii.pii_processors import (
    LocalPresidioProcessor,
    RemoteServerProcessor
)


class PIIStep(PipelineStep):
    """
    Pipeline step for PII removal from documents.
    
    Supports two processing modes:
    - local_presidio: Uses Presidio with Flair models (high accuracy)
    - remote_server: Uses remote PII removal service
    
    Example config:
    ```yaml
    - name: pii
      config:
        method: "local_presidio"  # or "remote_server"
        entities: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
        score_threshold: 0.35
        # For remote_server method:
        server_url: "http://localhost:8000"
        # For local_presidio method:
        model_name: "flair/ner-english-large"
        debug: false
    ```
    """

    def __init__(self, config: dict):
        """
        Initialize the PII step.
        
        Args:
            config: Configuration dictionary with processing parameters.
        """
        super().__init__(config, name="PIIStep")
        
        # Extract configuration parameters
        self.method = config.get("method", "local_presidio")
        self.entities = config.get("entities", ["PERSON", "EMAIL_ADDRESS"])
        self.score_threshold = config.get("score_threshold", 0.35)
        self.debug = config.get("debug", False)
        
        # Method-specific parameters
        self.server_url = config.get("server_url", "http://localhost:8000")
        self.timeout = config.get("timeout", 120)
        self.model_name = config.get("model_name", "flair/ner-english-large")
        
        # Initialize processor based on method
        self.processor = self._create_processor()

    def _create_processor(self):
        """Create the appropriate PII processor based on configuration."""
        if self.method == "local_presidio":
            return LocalPresidioProcessor(
                entities=self.entities,
                score_threshold=self.score_threshold,
                model_name=self.model_name,
                debug=self.debug
            )
        elif self.method == "remote_server":
            return RemoteServerProcessor(
                server_url=self.server_url,
                entities=self.entities,
                score_threshold=self.score_threshold,
                timeout=self.timeout,
                debug=self.debug
            )
        else:
            raise ValueError(f"Unsupported PII method: {self.method}. "
                           f"Supported methods: local_presidio, remote_server")


    async def execute(self, input_data: Union[List[Document], List[Tuple[Path, str]]]) -> List[Document]:
        """
        Execute the PII removal step on input data.
        
        Args:
            input_data: List of Documents or list of tuples containing (file_path, content).
            
        Returns:
            List of Documents with PII removed.
        """
        # Convert input to Document objects using utility function
        documents = normalize_to_documents(input_data)
        
        self.logger.info(f"Executing PII removal step on {len(documents)} documents using method: {self.method}")
        
        if not documents:
            self.logger.warning("No input data provided to PII step")
            return []
        
        result = []
        processed_count = 0
        failed_count = 0
        total_entities_found = 0
        
        for document in documents:
            try:
                if self.debug:
                    self.logger.debug(f"Processing document: {document.filename}")
                
                # Process the document
                processed_document = await self.processor.process(document)
                
                # Track statistics
                entities_found = processed_document.get_metadata('pii_entities_found', 0)
                total_entities_found += entities_found
                
                if processed_document.get_metadata('pii_processed', False):
                    processed_count += 1
                    if self.debug:
                        processing_time = processed_document.get_metadata('pii_processing_time', 0)
                        self.logger.debug(f"Successfully processed {document.filename} "
                                        f"(found {entities_found} entities in {processing_time:.2f}s)")
                else:
                    failed_count += 1
                    self.logger.warning(f"PII processing may have failed for {document.filename}")
                
                result.append(processed_document)
                
            except Exception as e:
                failed_count += 1
                self.logger.error(f"Failed to process {document.filename}: {str(e)}")
                # Add original document to result even if processing failed
                result.append(document)
        
        # Log summary statistics
        self.logger.info(f"PII removal completed - Processed: {processed_count}, "
                        f"Failed: {failed_count}, Total entities found: {total_entities_found}")
        
        if self.method == "remote_server" and failed_count > 0:
            self.logger.info(f"Note: If using remote_server method, ensure the PII server is running at {self.server_url}")
        
        return result


def create_pii_step(config: Dict[str, Any]) -> PIIStep:
    """
    Factory function to create a PII step with the given configuration.
    
    Args:
        config: Configuration dictionary for the PII step.
        
    Returns:
        Configured PIIStep instance.
    """
    return PIIStep(config)
