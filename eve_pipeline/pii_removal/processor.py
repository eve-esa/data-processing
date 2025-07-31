"""PII removal processor for the pipeline."""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus


class PIIRemover(ProcessorBase):
    """Processor for removing personally identifiable information (PII) from text."""
    
    def __init__(
        self,
        entities: Optional[List[str]] = None,
        score_threshold: float = 0.35,
        use_presidio: bool = True,
        use_flair: bool = True,
        server_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize PII remover.
        
        Args:
            entities: List of entity types to detect.
            score_threshold: Minimum confidence score for entity detection.
            use_presidio: Whether to use Presidio for detection.
            use_flair: Whether to use Flair for NER.
            server_url: URL of PII removal server (if using remote processing).
            **kwargs: Additional configuration.
        """
        super().__init__(name="PIIRemover", **kwargs)
        self.entities = entities or ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
        self.score_threshold = score_threshold
        self.use_presidio = use_presidio
        self.use_flair = use_flair
        self.server_url = server_url
        
        # Initialize components
        self._initialize_analyzer()
        self._initialize_patterns()
    
    def cleanup(self) -> None:
        """Clean up resources to prevent memory leaks."""
        try:
            # Clean up Flair/PyTorch resources
            if hasattr(self, 'analyzer') and self.analyzer:
                # Clear any cached models or tensors
                if hasattr(self.analyzer, 'nlp_engine'):
                    nlp_engine = self.analyzer.nlp_engine
                    if hasattr(nlp_engine, 'nlp') and nlp_engine.nlp:
                        if isinstance(nlp_engine.nlp, dict):
                            for lang_code, model in nlp_engine.nlp.items():
                                if hasattr(model, 'vocab') and hasattr(model.vocab, 'strings'):
                                    try:
                                        model.vocab.strings._reset_index()
                                    except (AttributeError, TypeError):
                                        # Skip if reset_index is not available or fails
                                        pass
                        elif hasattr(nlp_engine.nlp, 'vocab') and hasattr(nlp_engine.nlp.vocab, 'strings'):
                            try:
                                nlp_engine.nlp.vocab.strings._reset_index()
                            except (AttributeError, TypeError):
                                pass
                
                # Clean up Presidio analyzer
                self.analyzer = None
            
            # Clean up CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.logger.info("PII remover cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during PII remover cleanup: {e}")
    
    def _initialize_analyzer(self) -> None:
        """Initialize the PII analyzer."""
        if self.use_presidio:
            try:
                self._initialize_presidio()
            except ImportError as e:
                self.logger.warning(f"Presidio not available: {e}")
                self.use_presidio = False
        
        if not self.use_presidio and not self.server_url:
            self.logger.warning("No PII detection method available")
    
    def _initialize_presidio(self) -> None:
        """Initialize Presidio analyzer."""
        try:
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            
            # Configure NLP engine
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
            
            # Create registry
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)
            
            # Add Flair recognizer if requested
            if self.use_flair:
                try:
                    flair_recognizer = self._create_flair_recognizer()
                    registry.remove_recognizer("SpacyRecognizer")  # Remove default spacy
                    registry.add_recognizer(flair_recognizer)
                except ImportError:
                    self.logger.warning("Flair not available, using default recognizers")
            
            # Create analyzer
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
            self.logger.info("Presidio analyzer initialized successfully")
            
        except ImportError as e:
            raise ImportError(f"Presidio dependencies not available: {e}")
    
    def _create_flair_recognizer(self):
        """Create Flair-based recognizer."""
        try:
            from flair.models import SequenceTagger
            from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation
            from presidio_analyzer.nlp_engine import NlpArtifacts
            from flair.data import Sentence
            
            class FlairRecognizer(EntityRecognizer):
                """Flair-based entity recognizer."""
                
                def __init__(self):
                    self.model = SequenceTagger.load("flair/ner-english-large")
                    super().__init__(
                        supported_entities=["PERSON"],
                        supported_language="en",
                        name="FlairRecognizer",
                    )
                
                def load(self) -> None:
                    """Model already loaded."""
                    pass
                
                def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None) -> List[RecognizerResult]:
                    """Analyze text for entities."""
                    results = []
                    sentences = Sentence(text)
                    self.model.predict(sentences)
                    
                    for entity in sentences.get_spans("ner"):
                        if entity.labels[0].value in ["PER", "PERSON"]:
                            explanation = AnalysisExplanation(
                                recognizer=self.__class__.__name__,
                                original_score=round(entity.score, 2),
                                textual_explanation=f"Identified as {entity.labels[0].value} by Flair's NER"
                            )
                            
                            result = RecognizerResult(
                                entity_type="PERSON",
                                start=entity.start_position,
                                end=entity.end_position,
                                score=round(entity.score, 2),
                                analysis_explanation=explanation,
                            )
                            results.append(result)
                    
                    return results
            
            return FlairRecognizer()
            
        except ImportError as e:
            raise ImportError(f"Flair not available: {e}")
    
    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for additional PII detection."""
        self.email_pattern = re.compile(r'([-a-zA-Z0-9.`?{}]+@\w+(?:\.\w+)+)')
        self.phone_pattern = re.compile(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})')
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content to remove PII.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with PII removed.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        start_time = time.time()
        
        try:
            # Use server if configured
            if self.server_url:
                return self._process_with_server(content, input_path, start_time)
            
            # Use local processing
            return self._process_locally(content, input_path, start_time)
            
        except Exception as e:
            self.logger.error(f"PII removal failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )
    
    def _process_with_server(
        self, 
        content: str, 
        input_path: Optional[Union[str, Path]], 
        start_time: float,
    ) -> ProcessorResult:
        """Process content using remote server."""
        try:
            import requests
            
            payload = {
                "text": content,
                "entities": self.entities,
                "score_threshold": self.score_threshold,
                "return_analysis": True,
            }
            
            response = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            
            result_data = response.json()
            
            if not result_data.get("success", False):
                return ProcessorResult(
                    status=ProcessorStatus.FAILED,
                    input_path=input_path,
                    content=content,
                    processing_time=time.time() - start_time,
                    error_message=result_data.get("error_message", "Server processing failed"),
                )
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=result_data["anonymized_text"],
                processing_time=time.time() - start_time,
                metadata={
                    "entities_found": result_data.get("entities_found", []),
                    "server_processing_time": result_data.get("processing_time", 0),
                    "method": "server",
                },
            )
            
        except Exception as e:
            self.logger.error(f"Server processing failed: {e}")
            # Fallback to local processing
            return self._process_locally(content, input_path, start_time)
    
    def _process_locally(
        self, 
        content: str, 
        input_path: Optional[Union[str, Path]], 
        start_time: float,
    ) -> ProcessorResult:
        """Process content locally using Presidio."""
        if not self.use_presidio or not hasattr(self, 'analyzer'):
            return ProcessorResult(
                status=ProcessorStatus.SKIPPED,
                input_path=input_path,
                content=content,
                processing_time=time.time() - start_time,
                error_message="No PII detection method available",
            )
        
        try:
            # Analyze content
            analyze_results = self.analyzer.analyze(
                text=content,
                entities=self.entities,
                language="en",
                score_threshold=self.score_threshold,
            )
            
            # Anonymize content
            anonymized_text = self._anonymize_text(content, analyze_results)
            
            # Collect entity information
            entities_found = []
            for result in analyze_results:
                entities_found.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": content[result.start:result.end],
                })
            
            self.logger.info(f"Removed {len(entities_found)} PII entities")
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=anonymized_text,
                processing_time=time.time() - start_time,
                metadata={
                    "entities_found": entities_found,
                    "entities_removed": len(entities_found),
                    "method": "local",
                    "original_length": len(content),
                    "anonymized_length": len(anonymized_text),
                },
            )
            
        except Exception as e:
            self.logger.error(f"Local PII processing failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )
    
    def _anonymize_text(self, text: str, results: List[Any]) -> str:
        """Anonymize text by replacing detected entities."""
        # Sort results by start position (reverse order for safe replacement)
        results_sorted = sorted(results, key=lambda r: r.start, reverse=True)
        
        # Track covered spans to avoid overlaps
        covered_spans = set()
        anonymized_text = text
        
        for result in results_sorted:
            if result.entity_type in self.entities:
                # Check for overlaps
                if not any(i in covered_spans for i in range(result.start, result.end)):
                    placeholder = f"[{result.entity_type.upper()}]"
                    anonymized_text = (
                        anonymized_text[:result.start] + 
                        placeholder + 
                        anonymized_text[result.end:]
                    )
                    covered_spans.update(range(result.start, result.end))
        
        # Apply regex patterns for additional cleanup
        anonymized_text = self._apply_regex_anonymization(anonymized_text, covered_spans)
        
        return anonymized_text
    
    def _apply_regex_anonymization(self, text: str, covered_spans: set) -> str:
        """Apply regex-based anonymization for additional PII detection."""
        # Email addresses
        for match in reversed(list(self.email_pattern.finditer(text))):
            start, end = match.span()
            if not any(i in covered_spans for i in range(start, end)):
                text = text[:start] + "[EMAIL_ADDRESS]" + text[end:]
        
        return text