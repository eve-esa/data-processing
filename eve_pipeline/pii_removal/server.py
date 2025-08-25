"""PII removal server using LitServe for high-performance inference."""

import argparse
import logging
import re
import time
from typing import Any, Optional

import litserve as ls
from pydantic import BaseModel


class PIIRequest(BaseModel):
    """Request model for PII removal."""
    text: str
    entities: Optional[list[str]] = ["PERSON", "EMAIL_ADDRESS"]
    score_threshold: Optional[float] = 0.35
    return_analysis: Optional[bool] = False


class PIIResponse(BaseModel):
    """Response model for PII removal."""
    anonymized_text: str
    entities_found: list[dict[str, Any]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class FlairRecognizer:
    """Optimized Flair recognizer for person name detection."""

    def __init__(self, model) -> None:
        """Initialize with pre-loaded Flair model."""
        self.model = model
        self.supported_entities = ["PERSON"]

    def analyze(self, text: str, entities: list[str]) -> list[dict[str, Any]]:
        """Analyze text using Flair model."""
        try:
            from flair.data import Sentence

            results = []
            sentences = Sentence(text)
            self.model.predict(sentences)

            if not entities:
                entities = self.supported_entities

            for entity in entities:
                if entity not in self.supported_entities:
                    continue

                for ent in sentences.get_spans("ner"):
                    if ent.labels[0].value in ["PER", "PERSON"]:
                        result = {
                            "entity_type": "PERSON",
                            "start": ent.start_position,
                            "end": ent.end_position,
                            "score": round(ent.score, 2),
                            "text": text[ent.start_position:ent.end_position],
                        }
                        results.append(result)

            return results

        except Exception as e:
            logger = logging.getLogger("FlairRecognizer")
            logger.error(f"Flair analysis failed: {e}")
            return []


class PIILitAPI(ls.LitAPI):
    """LitServe API for PII removal."""

    def __init__(self, max_batch_size: int = 8, batch_timeout: float = 0.1):
        """Initialize PII API with batching configuration.
        
        Args:
            max_batch_size: Maximum batch size for processing requests.
            batch_timeout: Timeout for batch collection in seconds.
        """
        super().__init__()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout

    def setup(self, _device) -> None:
        """Initialize the PII removal pipeline during server startup."""
        self.logger = logging.getLogger("PIILitAPI")
        self.logger.info("Setting up PII Removal API...")
        start_init = time.time()

        # Load Flair model
        self.logger.info("Loading Flair model...")
        model_start = time.time()
        try:
            from flair.models import SequenceTagger
            self.flair_model = SequenceTagger.load("flair/ner-english-large")
            model_end = time.time()
            self.logger.info(f"[SETUP] Flair model loading: {model_end - model_start:.4f}s")
        except ImportError:
            self.logger.warning("Flair not available, skipping Flair model")
            self.flair_model = None

        # Initialize Presidio
        self.logger.info("Initializing Presidio...")
        presidio_start = time.time()
        try:
            self._initialize_presidio()
            presidio_end = time.time()
            self.logger.info(f"[SETUP] Presidio initialization: {presidio_end - presidio_start:.4f}s")
        except ImportError:
            self.logger.warning("Presidio not available")
            self.analyzer = None

        # Initialize regex patterns
        self.email_pattern = re.compile(r'([-a-zA-Z0-9.`?{}]+@\w+(?:\.\w+)+)')
        self.phone_pattern = re.compile(r'(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})')

        end_init = time.time()
        self.logger.info(f"[SETUP] Total initialization: {end_init - start_init:.4f}s")
        self.logger.info("PII Removal API setup complete!")

    def _initialize_presidio(self) -> None:
        """Initialize Presidio components."""
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Suppress excessive Presidio/SpaCy logging
        presidio_loggers = [
            'presidio_analyzer',
            'presidio_analyzer.nlp_engine', 
            'presidio_analyzer.recognizer_registry',
            'presidio_analyzer.analyzer_engine',
            'spacy',
            'transformers',
        ]
        for logger_name in presidio_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Configure NLP engine
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

        # Create registry
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)

        # Add Flair recognizer if available
        if self.flair_model:
            registry.remove_recognizer("SpacyRecognizer")  # Remove default spacy
            flair_recognizer = self._create_presidio_flair_recognizer()
            registry.add_recognizer(flair_recognizer)

        # Create analyzer
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

    def _create_presidio_flair_recognizer(self):
        """Create Presidio-compatible Flair recognizer."""
        from flair.data import Sentence
        from presidio_analyzer import (
            AnalysisExplanation,
            EntityRecognizer,
            RecognizerResult,
        )
        from presidio_analyzer.nlp_engine import NlpArtifacts

        class PresidioFlairRecognizer(EntityRecognizer):
            """Presidio-compatible Flair recognizer."""

            def __init__(self, model):
                self.model = model
                super().__init__(
                    supported_entities=["PERSON"],
                    supported_language="en",
                    name="FlairAnalytics",
                )

            def load(self) -> None:
                """Model already loaded."""
                pass

            def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts = None) -> list[RecognizerResult]:
                """Analyze text using Flair model."""
                results = []
                sentences = Sentence(text)
                self.model.predict(sentences)

                if not entities:
                    entities = self.supported_entities

                for entity in entities:
                    if entity not in self.supported_entities:
                        continue

                    for ent in sentences.get_spans("ner"):
                        if ent.labels[0].value in ["PER", "PERSON"]:
                            explanation = AnalysisExplanation(
                                recognizer=self.__class__.__name__,
                                original_score=round(ent.score, 2),
                                textual_explanation=f"Identified as {ent.labels[0].value} by Flair's NER",
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

        return PresidioFlairRecognizer(self.flair_model)

    def decode_request(self, request) -> PIIRequest:
        """Decode the incoming request."""
        return PIIRequest(**request)

    def predict(self, request) -> PIIResponse:
        """Process PII removal request(s)."""
        # Handle both single requests and batched requests
        if isinstance(request, list):
            # Batched requests - process each individually and return list of responses
            responses = []
            for single_request in request:
                if isinstance(single_request, dict):
                    single_request = PIIRequest(**single_request)
                responses.append(self._process_single_request(single_request))
            return responses
        else:
            # Single request
            if isinstance(request, dict):
                request = PIIRequest(**request)
            return self._process_single_request(request)

    def _process_single_request(self, request: PIIRequest) -> PIIResponse:
        """Process a single PII removal request."""
        start_time = time.time()

        try:
            if not request.text:
                return PIIResponse(
                    anonymized_text="",
                    entities_found=[],
                    processing_time=0.0,
                    success=False,
                    error_message="Empty text provided",
                )

            # Analyze with Presidio if available
            entities_found = []
            if self.analyzer:
                analyze_start = time.time()
                analyze_results = self.analyzer.analyze(
                    text=request.text,
                    entities=request.entities,
                    language="en",
                    score_threshold=request.score_threshold,
                )
                analyze_end = time.time()

                if request.return_analysis:
                    for result in analyze_results:
                        entities_found.append({
                            "entity_type": result.entity_type,
                            "start": result.start,
                            "end": result.end,
                            "score": result.score,
                            "text": request.text[result.start:result.end],
                        })

                # Anonymize text
                anonymized_text = self._anonymize_text(request.text, analyze_results)

                self.logger.debug(f"[API] Analysis: {analyze_end - analyze_start:.4f}s")
            else:
                # Fallback to regex-only processing
                anonymized_text = self._regex_anonymize(request.text)

            end_time = time.time()
            processing_time = end_time - start_time

            return PIIResponse(
                anonymized_text=anonymized_text,
                entities_found=entities_found,
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time

            self.logger.error(f"PII processing failed: {e!s}")

            return PIIResponse(
                anonymized_text=request.text,
                entities_found=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e),
            )

    def encode_response(self, response: PIIResponse) -> dict[str, Any]:
        """Encode the response."""
        return response.model_dump()

    def _anonymize_text(self, text: str, results: list[Any]) -> str:
        """Anonymize text by replacing detected entities."""
        # Sort results by start position (reverse order)
        results_sorted = sorted(results, key=lambda r: r.start, reverse=True)

        # Track covered spans to avoid overlaps
        covered_spans = set()

        for res in results_sorted:
            if res.entity_type in ['PERSON', 'EMAIL_ADDRESS', 'PHONE_NUMBER']:
                placeholder = f"[{res.entity_type.upper()}]"
                text = text[:res.start] + placeholder + text[res.end:]
                covered_spans.update(range(res.start, res.end))

        # Apply regex patterns for additional cleanup
        text = self._apply_regex_cleanup(text, covered_spans)

        return text

    def _regex_anonymize(self, text: str) -> str:
        """Fallback regex-based anonymization."""
        # Email addresses
        text = self.email_pattern.sub("[EMAIL_ADDRESS]", text)

        # Phone numbers
        text = self.phone_pattern.sub("[PHONE_NUMBER]", text)

        return text

    def _apply_regex_cleanup(self, text: str, covered_spans: set) -> str:
        """Apply regex patterns for additional PII detection."""
        # Email addresses
        for match in reversed(list(self.email_pattern.finditer(text))):
            start, end = match.span()
            if not any(i in covered_spans for i in range(start, end)):
                text = text[:start] + "[EMAIL_ADDRESS]" + text[end:]

        return text


def create_server(workers: int = 1, devices: str = "auto", max_batch_size: int = 8, batch_timeout: float = 0.1):
    """Create and configure the LitServe server."""
    api = PIILitAPI(max_batch_size=max_batch_size, batch_timeout=batch_timeout)

    server = ls.LitServer(
        api,
        accelerator=devices,
        workers_per_device=workers,
        timeout=300,
    )

    return server


def main():
    """Main function to run the PII removal server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description="PII Removal Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers per device")
    parser.add_argument("--devices", type=str, default="auto", help="Device configuration (auto, cpu, cuda)")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Maximum batch size for processing")
    parser.add_argument("--batch-timeout", type=float, default=0.1, help="Batch timeout in seconds")

    args = parser.parse_args()

    logger = logging.getLogger("PIIServer")
    logger.info(f"Starting PII Removal Server on port {args.port}")
    logger.info(f"Workers per device: {args.workers}")
    logger.info(f"Device configuration: {args.devices}")

    server = create_server(
        workers=args.workers,
        devices=args.devices,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
    )

    server.run(port=args.port)


if __name__ == "__main__":
    main()
