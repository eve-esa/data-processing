"""
PII removal processing components for the EVE pipeline.

This module provides PII removal using Microsoft Presidio with support for both
local processing and remote server-based processing.
"""

import time
import aiohttp
from typing import Optional, List, Dict, Any

from eve.logging import get_logger
from eve.model.document import Document
from eve.common.regex_patterns import EMAIL_PATTERN


def anonymize_text(text: str, entities: List[Dict[str, Any]]) -> str:
    """Apply anonymization to text based on detected entities.
    
    Args:
        text: Original text
        entities: List of detected PII entities
        
    Returns:
        Anonymized text
    """
    # Sort entities by start position in reverse order to avoid index shifting
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0), reverse=True)
    
    anonymized_text = text
    for entity in sorted_entities:
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        entity_type = entity.get('entity_type', entity.get('label', 'PII'))
        
        # Create placeholder
        placeholder = f"[{entity_type.upper()}]"
        
        # Replace the entity text with placeholder
        anonymized_text = anonymized_text[:start] + placeholder + anonymized_text[end:]
    
    return anonymized_text


class LocalPresidioProcessor:
    """Local PII processor using Presidio with Flair models."""
    
    def __init__(self, 
                 entities: Optional[List[str]] = None,
                 score_threshold: float = 0.35,
                 model_name: str = "flair/ner-english-large",
                 debug: bool = False):
        """Initialize the local Presidio processor.
        
        Args:
            entities: List of entity types to detect (default: ["PERSON", "EMAIL_ADDRESS"])
            score_threshold: Minimum confidence score for detection
            model_name: Flair model to use for NER
            debug: Enable debug output
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)
        self.entities = entities or ["PERSON", "EMAIL_ADDRESS"]
        self.score_threshold = score_threshold
        self.model_name = model_name
        self._analyzer = None
        
    async def _initialize_analyzer(self):
        """Lazy initialization of Presidio analyzer."""
        if self._analyzer is not None:
            return
            
        try:
            from presidio_analyzer import AnalyzerEngine, RecognizerResult
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer import RecognizerRegistry, EntityRecognizer, AnalysisExplanation
            from presidio_analyzer.nlp_engine import NlpArtifacts
            from flair.models import SequenceTagger
            from flair.data import Sentence
            
            self.logger.info("Initializing Presidio analyzer with Flair model...")
            start_time = time.time()
            
            # Load Flair model
            flair_model = SequenceTagger.load(self.model_name)
            
            # Setup NLP engine
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            
            # Setup registry with Flair recognizer
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            registry.remove_recognizer("SpacyRecognizer")
            
            # Custom Flair recognizer
            class FlairRecognizer(EntityRecognizer):
                def __init__(self, model: SequenceTagger):
                    self.model = model
                    super().__init__(
                        supported_entities=["PERSON"],
                        supported_language="en",
                        name="Flair Analytics",
                    )

                def load(self) -> None:
                    pass

                def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None) -> List[RecognizerResult]:
                    results = []
                    sentences = Sentence(text)
                    self.model.predict(sentences)

                    for entity in (entities or self.supported_entities):
                        if entity not in self.supported_entities:
                            continue

                        for ent in sentences.get_spans("ner"):
                            if ent.labels[0].value in ["PER", "PERSON"]:
                                explanation = AnalysisExplanation(
                                    recognizer=self.__class__.__name__,
                                    original_score=round(ent.score, 2),
                                    textual_explanation=f"Identified as {ent.labels[0].value} by Flair's NER"
                                )
                                
                                result = RecognizerResult(
                                    entity_type="PERSON",
                                    start=ent.start_position,
                                    end=ent.end_position,
                                    score=round(ent.score, 2),
                                    analysis_explanation=explanation,
                                )
                                results.append(result)

                    return results
            
            flair_recognizer = FlairRecognizer(flair_model)
            registry.add_recognizer(flair_recognizer)
            
            # Create analyzer
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            
            init_time = time.time() - start_time
            self.logger.info(f"Presidio analyzer initialized in {init_time:.2f}s")
            
        except ImportError as e:
            self.logger.error(f"Failed to import required libraries for Presidio: {e}")
            self.logger.error("Please install: pip install presidio-analyzer presidio-anonymizer flair spacy")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Presidio analyzer: {e}")
            raise

    def _analyze_with_presidio(self, text: str) -> List[Dict[str, Any]]:
        """Analyze text using Presidio and return detected entities."""
        analyze_results = self._analyzer.analyze(
            text=text,
            entities=self.entities,
            language="en",
            score_threshold=self.score_threshold,
            return_decision_process=False
        )
        
        entities_found = []
        for result in analyze_results:
            entities_found.append({
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start:result.end]
            })
        
        return entities_found
    
    def _detect_emails_with_regex(self, text: str, existing_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect additional emails using regex pattern."""
        additional_emails = []
        
        for match in EMAIL_PATTERN.finditer(text):
            start, end = match.span()
            # Check if this email is already detected
            if not any(e['start'] <= start < e['end'] or e['start'] < end <= e['end'] 
                      for e in existing_entities):
                additional_emails.append({
                    "entity_type": "EMAIL_ADDRESS",
                    "start": start,
                    "end": end,
                    "score": 1.0,
                    "text": match.group()
                })
        
        return additional_emails
    
    def _update_document_with_results(self, document: Document, anonymized_text: str, 
                                    entities_found: List[Dict[str, Any]], processing_time: float) -> None:
        """Update document with PII processing results."""
        document.update_content(anonymized_text)
        document.add_metadata('pii_processed', True)
        document.add_metadata('pii_entities_found', len(entities_found))
        document.add_metadata('pii_processing_time', processing_time)
        document.add_metadata('pii_method', 'local_presidio')
    
    def _log_debug_info(self, document: Document, entities_found: List[Dict[str, Any]], processing_time: float) -> None:
        """Log debug information about PII processing."""
        if self.debug:
            self.logger.info(f"Found {len(entities_found)} PII entities in {document.filename}")
            for entity in entities_found:
                self.logger.debug(f"  {entity['entity_type']}: {entity['text']} (score: {entity['score']:.2f})")
        
        self.logger.info(f"{document.filename} - PII processing completed in {processing_time:.2f}s")

    async def process(self, document: Document) -> Document:
        """Process document for PII removal using local Presidio."""
        if self.debug:
            self.logger.info(f"Processing {document.filename} for PII removal (local)")
        
        if document.is_empty():
            self.logger.warning(f"{document.filename} - Empty content in PII processing")
            return document
        
        try:
            start_time = time.time()
            
            # Ensure analyzer is initialized
            await self._initialize_analyzer()
            
            # Analyze text for PII using Presidio
            entities_found = self._analyze_with_presidio(document.content)
            
            # Detect additional emails with regex
            additional_emails = self._detect_emails_with_regex(document.content, entities_found)
            entities_found.extend(additional_emails)
            
            # Anonymize the text
            anonymized_text = anonymize_text(document.content, entities_found)
            
            # Calculate processing time and update document
            processing_time = time.time() - start_time
            self._update_document_with_results(document, anonymized_text, entities_found, processing_time)
            
            # Log debug information
            self._log_debug_info(document, entities_found, processing_time)
            
            return document
            
        except Exception as e:
            self.logger.error(f"{document.filename} - PII processing failed: {str(e)}")
            return document


class RemoteServerProcessor:
    """Remote PII processor using a server-based API (LitServe)."""
    
    def __init__(self,
                 server_url: str = "http://localhost:8000",
                 entities: Optional[List[str]] = None,
                 score_threshold: float = 0.35,
                 timeout: int = 120,
                 debug: bool = False):
        """Initialize the remote server processor.
        
        Args:
            server_url: URL of the PII removal server
            entities: List of entity types to detect
            score_threshold: Minimum confidence score for detection
            timeout: Request timeout in seconds
            debug: Enable debug output
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)
        self.server_url = server_url.rstrip('/')
        self.predict_url = f"{self.server_url}/predict"
        self.entities = entities or ["PERSON", "EMAIL_ADDRESS"]
        self.score_threshold = score_threshold
        self.timeout = timeout

    async def _make_request(self, text: str) -> Dict[str, Any]:
        """Make async request to PII removal server."""
        payload = {
            "text": text,
            "entities": self.entities,
            "score_threshold": self.score_threshold,
            "return_analysis": True
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.predict_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Server request failed with status {response.status}: {error_text}")

    def _validate_server_response(self, result: Dict[str, Any]) -> None:
        """Validate server response and raise exception if invalid."""
        if not result.get('success', False):
            error_msg = result.get('error_message', 'Unknown error')
            raise Exception(f"Server processing failed: {error_msg}")
    
    def _update_document_with_remote_results(self, document: Document, anonymized_text: str, 
                                           entities_found: List[Dict[str, Any]], processing_time: float) -> None:
        """Update document with remote PII processing results."""
        document.update_content(anonymized_text)
        document.add_metadata('pii_processed', True)
        document.add_metadata('pii_entities_found', len(entities_found))
        document.add_metadata('pii_processing_time', processing_time)
        document.add_metadata('pii_method', 'remote_server')
        document.add_metadata('pii_server_url', self.server_url)
    
    def _log_remote_debug_info(self, document: Document, entities_found: List[Dict[str, Any]], processing_time: float) -> None:
        """Log debug information about remote PII processing."""
        if self.debug:
            self.logger.info(f"Found {len(entities_found)} PII entities in {document.filename}")
            for entity in entities_found:
                self.logger.debug(f"  {entity['entity_type']}: {entity['text']} (score: {entity['score']:.2f})")
        
        self.logger.info(f"{document.filename} - PII processing completed in {processing_time:.2f}s")

    async def process(self, document: Document) -> Document:
        """Process document for PII removal using remote server."""
        if self.debug:
            self.logger.info(f"Processing {document.filename} for PII removal (remote)")
        
        if document.is_empty():
            self.logger.warning(f"{document.filename} - Empty content in PII processing")
            return document
        
        try:
            start_time = time.time()
            
            # Make request to server
            result = await self._make_request(document.content)
            
            # Validate response
            self._validate_server_response(result)
            
            # Extract results
            anonymized_text = result.get('anonymized_text', document.content)
            entities_found = result.get('entities_found', [])
            
            # Calculate processing time and update document
            processing_time = time.time() - start_time
            self._update_document_with_remote_results(document, anonymized_text, entities_found, processing_time)
            
            # Log debug information
            self._log_remote_debug_info(document, entities_found, processing_time)
            
            return document
            
        except Exception as e:
            self.logger.error(f"{document.filename} - PII processing failed: {str(e)}")
            return document


