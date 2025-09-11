"""Tests for PII removal components and PII step."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

from eve.steps.pii.pii_step import PIIStep
from eve.model.document import Document
from eve.steps.pii.pii_processors import (
    PIIProcessor,
    LocalPresidioProcessor,
    RemoteServerProcessor,
)


class TestPIIStep:
    """Test suite for the main PIIStep class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for PII step."""
        return {
            "method": "local_presidio",
            "entities": ["PERSON", "EMAIL_ADDRESS"],
            "score_threshold": 0.35,
            "debug": False
        }

    @pytest.fixture
    def presidio_config(self):
        """Configuration for Presidio method."""
        return {
            "method": "local_presidio",
            "entities": ["PERSON", "EMAIL_ADDRESS"],
            "score_threshold": 0.35,
            "model_name": "flair/ner-english-large",
            "debug": True
        }

    @pytest.fixture
    def remote_config(self):
        """Configuration for remote server method."""
        return {
            "method": "remote_server",
            "server_url": "http://localhost:8000",
            "entities": ["PERSON", "EMAIL_ADDRESS"],
            "score_threshold": 0.35,
            "timeout": 30,
            "debug": False
        }

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data with PII for testing."""
        return [
            Document(
                content="Hello John Smith, please contact us at john.smith@example.com",
                file_path=Path("file1.txt"),
                file_format="txt"
            ),
            Document(
                content="Meeting with Alice Johnson (alice@company.com) and Bob Wilson.",
                file_path=Path("file2.md"),
                file_format="md"
            ),
            Document(
                content="No PII content here, just regular text.",
                file_path=Path("file3.txt"),
                file_format="txt"
            )
        ]

    def test_pii_step_initialization_presidio(self, basic_config):
        """Test initialization of PII step with Presidio method."""
        step = PIIStep(basic_config)
        
        assert step.method == "local_presidio"
        assert step.entities == ["PERSON", "EMAIL_ADDRESS"]
        assert step.score_threshold == 0.35
        assert step.debug == False
        assert isinstance(step.processor, LocalPresidioProcessor)

    def test_pii_step_initialization_presidio_advanced(self, presidio_config):
        """Test initialization of PII step with advanced Presidio configuration."""
        step = PIIStep(presidio_config)
        
        assert step.method == "local_presidio"
        assert step.model_name == "flair/ner-english-large"
        assert step.debug == True
        assert isinstance(step.processor, LocalPresidioProcessor)

    def test_pii_step_initialization_remote(self, remote_config):
        """Test initialization of PII step with remote server method."""
        step = PIIStep(remote_config)
        
        assert step.method == "remote_server"
        assert step.server_url == "http://localhost:8000"
        assert step.timeout == 30
        assert isinstance(step.processor, RemoteServerProcessor)

    def test_pii_step_invalid_method(self):
        """Test initialization with invalid method raises error."""
        config = {"method": "invalid_method"}
        
        with pytest.raises(ValueError) as exc_info:
            PIIStep(config)
        
        assert "Unsupported PII method" in str(exc_info.value)


    @pytest.mark.asyncio
    async def test_execute_with_valid_input(self, basic_config, sample_input_data):
        """Test execute method with valid input data."""
        step = PIIStep(basic_config)
        
        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(side_effect=lambda doc: self._create_processed_document(doc))
        step.processor = mock_processor
        
        result = await step.execute(sample_input_data)
        
        assert len(result) == 3
        assert all(isinstance(item, Document) for item in result)
        assert all(item.get_metadata('pii_processed') == True for item in result)

    @pytest.mark.asyncio
    async def test_execute_with_empty_input(self, basic_config):
        """Test execute method with empty input."""
        step = PIIStep(basic_config)
        
        result = await step.execute([])
        
        assert result == []

    @pytest.mark.asyncio
    async def test_execute_with_tuple_input(self, basic_config):
        """Test execute method with tuple input (backwards compatibility)."""
        step = PIIStep(basic_config)
        
        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(side_effect=lambda doc: self._create_processed_document(doc))
        step.processor = mock_processor
        
        tuple_input = [
            (Path("file1.txt"), "John Smith works at john@company.com"),
            (Path("file2.txt"), "Contact Alice at alice@example.com")
        ]
        
        result = await step.execute(tuple_input)
        
        assert len(result) == 2
        assert all(isinstance(item, Document) for item in result)

    @pytest.mark.asyncio
    async def test_execute_with_path_input(self, basic_config):
        """Test execute method with Path input."""
        step = PIIStep(basic_config)
        
        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(side_effect=lambda doc: self._create_processed_document(doc))
        step.processor = mock_processor
        
        path_input = [Path("file1.txt"), Path("file2.txt")]
        
        result = await step.execute(path_input)
        
        assert len(result) == 2
        assert all(isinstance(item, Document) for item in result)

    @pytest.mark.asyncio
    async def test_execute_with_processor_failure(self, basic_config, sample_input_data):
        """Test execute method when processor fails."""
        step = PIIStep(basic_config)
        
        # Mock processor to fail on first document, succeed on others
        mock_processor = AsyncMock()
        side_effects = [
            Exception("Processing failed"),
            self._create_processed_document(sample_input_data[1]),
            self._create_processed_document(sample_input_data[2])
        ]
        mock_processor.process = AsyncMock(side_effect=side_effects)
        step.processor = mock_processor
        
        result = await step.execute(sample_input_data)
        
        # Should return all documents, even if processing failed
        assert len(result) == 3
        # First document should be original (processing failed)
        assert result[0].get_metadata('pii_processed') != True
        # Others should be processed
        assert result[1].get_metadata('pii_processed') == True
        assert result[2].get_metadata('pii_processed') == True

    def _create_processed_document(self, original_doc):
        """Helper method to create a processed document with PII metadata."""
        processed = Document(
            content="[PERSON] works at [EMAIL_ADDRESS]",  # Anonymized content
            file_path=original_doc.file_path,
            file_format=original_doc.file_format
        )
        processed.add_metadata('pii_processed', True)
        processed.add_metadata('pii_entities_found', 2)
        processed.add_metadata('pii_processing_time', 0.5)
        processed.add_metadata('pii_method', 'test_method')
        return processed


class TestPIIProcessors:
    """Test suite for PII processors."""

    @pytest.fixture
    def sample_document(self):
        """Sample document with PII content."""
        return Document(
            content="Hello John Doe, please contact support@company.com for assistance.",
            file_path=Path("test.txt"),
            file_format="txt"
        )

    @pytest.fixture
    def empty_document(self):
        """Empty document for testing."""
        return Document(
            content="",
            file_path=Path("empty.txt"),
            file_format="txt"
        )


    @pytest.mark.asyncio
    async def test_presidio_processor_initialization(self):
        """Test Presidio processor initialization."""
        processor = LocalPresidioProcessor(
            entities=["PERSON", "EMAIL_ADDRESS"],
            score_threshold=0.4,
            model_name="test_model",
            debug=True
        )
        
        assert processor.entities == ["PERSON", "EMAIL_ADDRESS"]
        assert processor.score_threshold == 0.4
        assert processor.model_name == "test_model"
        assert processor.debug == True
        assert processor._analyzer is None

    @pytest.mark.asyncio
    async def test_presidio_processor_with_empty_document(self, empty_document):
        """Test Presidio processor with empty document."""
        processor = LocalPresidioProcessor()
        
        result = await processor.process(empty_document)
        
        assert result.content == ""
        assert result.get_metadata('pii_processed') != True

    @pytest.mark.asyncio
    async def test_remote_processor_initialization(self):
        """Test remote server processor initialization."""
        processor = RemoteServerProcessor(
            server_url="http://test:8000",
            entities=["PERSON", "EMAIL_ADDRESS"],
            timeout=60
        )
        
        assert processor.server_url == "http://test:8000"
        assert processor.predict_url == "http://test:8000/predict"
        assert processor.timeout == 60

    @pytest.mark.asyncio
    async def test_remote_processor_with_empty_document(self, empty_document):
        """Test remote server processor with empty document."""
        processor = RemoteServerProcessor()
        
        result = await processor.process(empty_document)
        
        assert result.content == ""
        assert result.get_metadata('pii_processed') != True

    @pytest.mark.asyncio
    async def test_remote_processor_mock_request(self, sample_document):
        """Test remote server processor with mocked HTTP request."""
        processor = RemoteServerProcessor(server_url="http://test:8000", debug=True)
        
        # Mock the HTTP request
        mock_response = {
            "success": True,
            "anonymized_text": "Hello [PERSON], please contact [EMAIL_ADDRESS] for assistance.",
            "entities_found": [
                {"entity_type": "PERSON", "start": 6, "end": 14, "score": 0.9, "text": "John Doe"},
                {"entity_type": "EMAIL_ADDRESS", "start": 31, "end": 50, "score": 0.95, "text": "support@company.com"}
            ]
        }
        
        with patch.object(processor, '_make_request', return_value=mock_response):
            result = await processor.process(sample_document)
            
            assert result.get_metadata('pii_processed') == True
            assert result.get_metadata('pii_entities_found') == 2
            assert result.get_metadata('pii_method') == 'remote_server'
            assert result.get_metadata('pii_server_url') == "http://test:8000"
            assert "[PERSON]" in result.content
            assert "[EMAIL_ADDRESS]" in result.content

    @pytest.mark.asyncio
    async def test_remote_processor_request_failure(self, sample_document):
        """Test remote server processor with request failure."""
        processor = RemoteServerProcessor(debug=True)
        
        # Mock failed request
        with patch.object(processor, '_make_request', side_effect=Exception("Connection failed")):
            result = await processor.process(sample_document)
            
            # Should return original document when processing fails
            assert result.content == sample_document.content
            assert result.get_metadata('pii_processed') != True

    def test_pii_processor_anonymize_text(self):
        """Test the base anonymize_text method."""
        processor = LocalPresidioProcessor()  # Use concrete implementation
        
        text = "Hello John Doe, contact support@company.com"
        entities = [
            {"start": 6, "end": 14, "entity_type": "PERSON", "text": "John Doe"},
            {"start": 24, "end": 43, "entity_type": "EMAIL_ADDRESS", "text": "support@company.com"}
        ]
        
        result = processor._anonymize_text(text, entities)
        
        assert result == "Hello [PERSON], contact [EMAIL_ADDRESS]"

    def test_pii_processor_anonymize_text_overlapping(self):
        """Test anonymize_text with overlapping entities (should handle gracefully)."""
        processor = LocalPresidioProcessor()
        
        text = "John Doe john.doe@company.com"
        entities = [
            {"start": 0, "end": 8, "entity_type": "PERSON", "text": "John Doe"},
            {"start": 9, "end": 29, "entity_type": "EMAIL_ADDRESS", "text": "john.doe@company.com"}
        ]
        
        result = processor._anonymize_text(text, entities)
        
        # Should replace both entities
        assert "[PERSON]" in result
        assert "[EMAIL_ADDRESS]" in result

    def test_pii_processor_anonymize_text_empty_entities(self):
        """Test anonymize_text with no entities."""
        processor = LocalPresidioProcessor()
        
        text = "This text has no PII."
        entities = []
        
        result = processor._anonymize_text(text, entities)
        
        assert result == text  # Should remain unchanged


class TestPIIStepIntegration:
    """Integration tests for PII step with different configurations."""

    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end PII processing with mocked dependencies."""
        config = {
            "method": "local_presidio",
            "entities": ["PERSON", "EMAIL_ADDRESS"],
            "score_threshold": 0.5,
            "debug": True
        }
        
        documents = [
            Document(
                content="Contact John Smith at john.smith@example.com",
                file_path=Path("doc1.txt"),
                file_format="txt"
            ),
            Document(
                content="Meeting with Alice Johnson and Bob Wilson",
                file_path=Path("doc2.txt"),
                file_format="txt"
            )
        ]
        
        step = PIIStep(config)
        
        # Mock the underlying Presidio components
        mock_processor = AsyncMock()
        mock_processor.process = AsyncMock(side_effect=lambda doc: self._create_mock_processed_document(doc))
        step.processor = mock_processor
        
        result = await step.execute(documents)
        
        assert len(result) == 2
        # At least one document should have PII processed
        processed_docs = [doc for doc in result if doc.get_metadata('pii_processed')]
        assert len(processed_docs) >= 1
    
    def _create_mock_processed_document(self, original_doc):
        """Helper to create a mock processed document."""
        processed = Document(
            content="Contact [PERSON] at [EMAIL_ADDRESS]",
            file_path=original_doc.file_path,
            file_format=original_doc.file_format
        )
        processed.add_metadata('pii_processed', True)
        processed.add_metadata('pii_entities_found', 2)
        processed.add_metadata('pii_processing_time', 0.5)
        processed.add_metadata('pii_method', 'local_presidio')
        return processed

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test various configuration scenarios."""
        # Test default values
        step = PIIStep({})
        assert step.method == "local_presidio"  # Default method
        assert step.entities == ["PERSON", "EMAIL_ADDRESS"]  # Default entities
        assert step.score_threshold == 0.35  # Default threshold
        
        # Test custom values
        custom_config = {
            "method": "remote_server",
            "entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            "score_threshold": 0.8,
            "debug": True
        }
        step = PIIStep(custom_config)
        assert step.method == "remote_server"
        assert len(step.entities) == 3
        assert step.score_threshold == 0.8
        assert step.debug == True


class TestPIIProcessorExceptionHandling:
    """Test exception handling in PII processors."""


    @pytest.mark.asyncio
    async def test_presidio_import_error(self):
        """Test Presidio processor behavior when dependencies are missing."""
        processor = LocalPresidioProcessor()
        
        # Mock the import inside the _initialize_analyzer method
        with patch('builtins.__import__', side_effect=ImportError("Presidio not found")):
            with pytest.raises(ImportError):
                await processor._initialize_analyzer()

    @pytest.mark.asyncio
    async def test_processor_runtime_error(self):
        """Test processor behavior when runtime errors occur."""
        processor = LocalPresidioProcessor()
        document = Document(
            content="Some text",
            file_path=Path("test.txt"),
            file_format="txt"
        )
        
        # Mock analyzer to raise an error during processing
        with patch.object(processor, '_initialize_analyzer', side_effect=RuntimeError("Model failed")):
            result = await processor.process(document)
            
            # Should return original document when processing fails
            assert result.content == document.content
            assert result.get_metadata('pii_processed') != True
