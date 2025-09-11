"""
PII Removal Server for EVE Pipeline

A LitServe-based server that provides PII removal as a service.
This allows for distributed PII processing and GPU resource sharing.

Usage:
    python eve/steps/pii/server.py --port 8000 --workers 2

For production deployment, consider using Docker or cloud services.
"""

import re
import time
import argparse
from typing import Optional, List, Dict, Any

import torch
from pydantic import BaseModel

try:
    import litserve as ls
    from flair.models import SequenceTagger
    from flair.data import Sentence
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer import RecognizerRegistry, EntityRecognizer, AnalysisExplanation
    from presidio_analyzer.nlp_engine import NlpArtifacts
except ImportError as e:
    print(f"Missing required dependencies for PII server: {e}")
    print("Please install: pip install litserve presidio-analyzer flair")
    exit(1)


class PIIRequest(BaseModel):
    text: str
    entities: Optional[List[str]] = ["PERSON", "EMAIL_ADDRESS"]
    score_threshold: Optional[float] = 0.35
    return_analysis: Optional[bool] = False


class PIIResponse(BaseModel):
    anonymized_text: str
    entities_found: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class FlairRecognizer(EntityRecognizer):
    """Optimized Flair recognizer for PII detection."""
    
    def __init__(self, model: SequenceTagger, supported_entities: List[str] = None):
        self.model = model
        supported_entities = supported_entities or ["PERSON"]
        
        super().__init__(
            supported_entities=supported_entities,
            supported_language="en",
            name="Flair Analytics",
        )

    def load(self) -> None:
        """Model already loaded during initialization"""
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts = None) -> List[RecognizerResult]:
        """Analyze text using Flair model"""
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


class PIILitAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the PII removal pipeline during server startup"""
        print("Setting up PII Removal API...")
        start_init = time.time()
        
        print("Loading Flair model...")
        model_start = time.time()
        self.flair_model = SequenceTagger.load("flair/ner-english-large")
        model_end = time.time()
        print(f"[SETUP] Flair model loading: {model_end - model_start:.4f}s")
        
        nlp_start = time.time()
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        nlp_end = time.time()
        print(f"[SETUP] NLP engine creation: {nlp_end - nlp_start:.4f}s")
        
        registry_start = time.time()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        registry.remove_recognizer("SpacyRecognizer")
        flair_recognizer = FlairRecognizer(self.flair_model)
        registry.add_recognizer(flair_recognizer)
        registry_end = time.time()
        print(f"[SETUP] Registry setup: {registry_end - registry_start:.4f}s")
        
        analyzer_start = time.time()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
        analyzer_end = time.time()
        print(f"[SETUP] Analyzer creation: {analyzer_end - analyzer_start:.4f}s")
        
        self.email_pattern = re.compile(r'([-a-zA-Z0-9.`?{}]+@\w+(?:\.\w+)+)')
        
        end_init = time.time()
        print(f"[SETUP] Total initialization: {end_init - start_init:.4f}s")
        print("PII Removal API setup complete!")

    def decode_request(self, request, context):
        """Decode the incoming request"""
        return PIIRequest(**request)

    def predict(self, request: PIIRequest, context) -> PIIResponse:
        """Process PII removal request"""
        start_time = time.time()
        
        try:
            if not request.text:
                return PIIResponse(
                    anonymized_text="",
                    entities_found=[],
                    processing_time=0.0,
                    success=False,
                    error_message="Empty text provided"
                )
            
            analyze_start = time.time()
            analyze_results = self.analyzer.analyze(
                text=request.text,
                entities=request.entities,
                language="en",
                score_threshold=request.score_threshold,
                return_decision_process=False
            )
            torch.cuda.empty_cache()
            analyze_end = time.time()
            
            anonymize_start = time.time()
            anonymized_text = self._anonymize_text(request.text, analyze_results)
            anonymize_end = time.time()
            
            entities_found = []
            if request.return_analysis:
                for result in analyze_results:
                    entities_found.append({
                        "entity_type": result.entity_type,
                        "start": result.start,
                        "end": result.end,
                        "score": result.score,
                        "text": request.text[result.start:result.end]
                    })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"[API] Analysis: {analyze_end - analyze_start:.4f}s, "
                  f"Anonymization: {anonymize_end - anonymize_start:.4f}s, "
                  f"Total: {processing_time:.4f}s")
            
            return PIIResponse(
                anonymized_text=anonymized_text,
                entities_found=entities_found,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"[ERROR] PII processing failed: {str(e)}")
            
            return PIIResponse(
                anonymized_text=request.text,
                entities_found=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def encode_response(self, response: PIIResponse, context):
        """Encode the response"""
        return response.dict()

    def _anonymize_text(self, text: str, results: List[RecognizerResult]) -> str:
        """Anonymize text by replacing PII entities with placeholders"""
        results_sorted = sorted(results, key=lambda r: r.start, reverse=True)
        covered_spans = set()
        
        for res in results_sorted:
            if res.entity_type in ['PERSON', 'EMAIL_ADDRESS']:
                placeholder = f"[{res.entity_type.upper()}]"
                text = text[:res.start] + placeholder + text[res.end:]
                covered_spans.update(range(res.start, res.end))

        # Additional email detection with regex
        for match in reversed(list(self.email_pattern.finditer(text))):
            start, end = match.span()
            if not any(i in covered_spans for i in range(start, end)):
                text = text[:start] + "[EMAIL_ADDRESS]" + text[end:]

        return text


def create_server(workers: int = 1, devices: str = "auto"):
    """Create and configure the LitServe server"""
    api = PIILitAPI()
    
    server = ls.LitServer(
        api, 
        accelerator=devices,
        workers_per_device=workers,
        timeout=120,
        max_batch_size=1,
    )
    
    return server


def main():
    parser = argparse.ArgumentParser(description="PII Removal Server for EVE Pipeline")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers per device")
    parser.add_argument("--devices", type=str, default="auto", help="Device configuration (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    print(f"Starting EVE PII Removal Server on port {args.port}")
    print(f"Workers per device: {args.workers}")
    print(f"Device configuration: {args.devices}")
    print("\nServer will be available at:")
    print(f"  - http://localhost:{args.port}")
    print(f"  - Prediction endpoint: http://localhost:{args.port}/predict")
    print("\nTo use with EVE pipeline, configure:")
    print("  method: 'remote_server'")
    print(f"  server_url: 'http://localhost:{args.port}'")
    
    server = create_server(
        workers=args.workers,
        devices=args.devices
    )
    
    server.run(port=args.port)


if __name__ == "__main__":
    main()
