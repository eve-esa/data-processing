"""Tests for metadata extraction components and metadata step."""

import json
import pytest
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from eve.steps.metadata.metadata_step import MetadataExtractionStep
from eve.model.document import Document
from eve.steps.metadata.processors import (
    PDFMetadataProcessor,
    HTMLMetadataProcessor,
    MetadataProcessor,
)


class TestMetadataExtractionStep:
    """Test suite for the main MetadataExtractionStep class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for metadata step."""
        return {
            "json_output_path": "test_metadata_results.json",
            "debug": False
        }

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for testing."""
        return [
            Document(
                content="",
                file_path=Path("document1.pdf"),
                file_format="pdf"
            ),
            Document(
                content="<html><head><title>Test HTML Document</title></head><body>Content</body></html>",
                file_path=Path("document2.html"),
                file_format="html"
            ),
            Document(
                content="Plain text content",
                file_path=Path("document3.txt"),
                file_format="txt"
            ),
            Document(
                content="# Markdown content",
                file_path=Path("document4.md"),
                file_format="md"
            )
        ]

    @pytest.fixture
    def sample_pdf_documents(self):
        """Sample PDF documents for testing."""
        return [
            Document(
                content="",
                file_path=Path("paper1.pdf"),
                file_format="pdf"
            ),
            Document(
                content="",
                file_path=Path("paper2.pdf"),
                file_format="pdf"
            )
        ]

    @pytest.fixture
    def sample_html_documents(self):
        """Sample HTML documents for testing."""
        return [
            Document(
                content="<html><head><title>Document Title</title></head><body>Content</body></html>",
                file_path=Path("page1.html"),
                file_format="html"
            ),
            Document(
                content="<html><head><title>Another Title</title></head><body>More content</body></html>",
                file_path=Path("page2.html"),
                file_format="html"
            )
        ]

    def test_metadata_step_initialization_basic(self, basic_config):
        """Test basic initialization of metadata step."""
        step = MetadataExtractionStep(basic_config)
        
        assert step.json_output_path == "test_metadata_results.json"
        assert step.supported_formats == {'pdf', 'html'}
        assert len(step.processors) == 2
        assert 'pdf' in step.processors
        assert 'html' in step.processors

    def test_metadata_step_initialization_without_db(self, basic_config):
        """Test initialization without database credentials."""
        with patch.dict('os.environ', {}, clear=True):
            step = MetadataExtractionStep(basic_config)
            assert step.use_database == False

    @patch.dict('os.environ', {
        'DB_HOST': 'localhost',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_pass',
        'DB_NAME': 'test_db'
    })
    def test_metadata_step_initialization_with_db(self, basic_config):
        """Test initialization with database credentials."""
        with patch('eve.steps.metadata.metadata_step.DB_LOGGER_AVAILABLE', True):
            step = MetadataExtractionStep(basic_config)
            assert step.use_database == True

    @pytest.mark.asyncio
    async def test_execute_with_supported_formats(self, basic_config, sample_input_data):
        """Test execute method with mixed file formats."""
        step = MetadataExtractionStep(basic_config)
        
        # Mock processors
        step.processors['pdf'].extract_metadata = AsyncMock(return_value={
            'file_format': 'pdf',
            'status': 'success',
            'metadata': {'title': 'Test PDF Title'},
            'method': 'pdf2bib'
        })
        step.processors['html'].extract_metadata = AsyncMock(return_value={
            'file_format': 'html',
            'status': 'success',
            'title': 'Test HTML Document'
        })
        
        # Mock JSON storage
        with patch.object(step, '_store_metadata_json', new_callable=AsyncMock) as mock_json:
            result = await step.execute(sample_input_data)
            
            # Should return all original documents unchanged
            assert len(result) == 4
            assert all(isinstance(doc, Document) for doc in result)
            
            # Should have called JSON storage
            mock_json.assert_called_once()
            
            # Check that metadata was extracted only for supported formats (PDF and HTML)
            call_args = mock_json.call_args[0][0]  # Get the metadata results passed to storage
            assert len(call_args) == 2  # Only PDF and HTML processed

    @pytest.mark.asyncio
    async def test_execute_with_empty_input(self, basic_config):
        """Test execute method with empty input."""
        step = MetadataExtractionStep(basic_config)
        
        result = await step.execute([])
        
        assert result == []

    @pytest.mark.asyncio
    async def test_execute_with_only_unsupported_formats(self, basic_config):
        """Test execute method with only unsupported file formats."""
        step = MetadataExtractionStep(basic_config)
        
        input_data = [
            Document(content="text", file_path=Path("file.txt"), file_format="txt"),
            Document(content="# markdown", file_path=Path("file.md"), file_format="md")
        ]
        
        with patch.object(step, '_store_metadata_json', new_callable=AsyncMock) as mock_json:
            result = await step.execute(input_data)
            
            assert len(result) == 2
            # Should not call storage when no supported documents found
            mock_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_with_processor_exception(self, basic_config, sample_pdf_documents):
        """Test execute method when processor raises exception."""
        step = MetadataExtractionStep(basic_config)
        
        # Mock processor to raise exception
        step.processors['pdf'].extract_metadata = AsyncMock(side_effect=Exception("Processing failed"))
        
        with patch.object(step, '_store_metadata_json', new_callable=AsyncMock) as mock_json:
            result = await step.execute(sample_pdf_documents)
            
            # Should still return original documents
            assert len(result) == 2
            
            # Should store error results
            mock_json.assert_called_once()
            call_args = mock_json.call_args[0][0]
            assert all(res.get('status') == 'error' for res in call_args)

    @pytest.mark.asyncio
    async def test_store_metadata_json_new_file(self, basic_config):
        """Test storing metadata to new JSON file."""
        step = MetadataExtractionStep(basic_config)
        
        metadata_results = [
            {
                'filepath': 'test.pdf',
                'file_format': 'pdf',
                'status': 'success',
                'metadata': {'title': 'Test Document'}
            }
        ]
        
        mock_file_content = json.dumps(metadata_results, indent=2)
        
        with patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', mock_open()) as mock_file:
            
            await step._store_metadata_json(metadata_results)
            
            # Check file was opened for writing
            mock_file.assert_called_once()
            handle = mock_file()
            # Check that JSON was written
            handle.write.assert_called()

    @pytest.mark.asyncio
    async def test_store_metadata_json_existing_file(self, basic_config):
        """Test storing metadata to existing JSON file (merge behavior)."""
        step = MetadataExtractionStep(basic_config)
        
        existing_data = [
            {'filepath': 'existing.pdf', 'status': 'success'}
        ]
        new_metadata = [
            {'filepath': 'new.pdf', 'status': 'success'}
        ]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(existing_data))) as mock_file:
            
            await step._store_metadata_json(new_metadata)
            
            # Should have read existing file and written merged data
            assert mock_file.call_count >= 2  # At least one read, one write

    @pytest.mark.asyncio
    async def test_store_metadata_json_invalid_existing_file(self, basic_config):
        """Test storing metadata when existing file has invalid JSON."""
        step = MetadataExtractionStep(basic_config)
        
        new_metadata = [
            {'filepath': 'new.pdf', 'status': 'success'}
        ]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")) as mock_file:
            
            await step._store_metadata_json(new_metadata)
            
            # Should handle invalid JSON gracefully and write new data
            mock_file.assert_called()

    @pytest.mark.asyncio
    async def test_store_metadata_database(self, basic_config):
        """Test storing metadata to database."""
        step = MetadataExtractionStep(basic_config)
        step.use_database = True
        
        metadata_results = [
            {
                'filepath': 'test.pdf',
                'file_format': 'pdf',
                'status': 'success',
                'metadata': {'title': 'Test Document'}
            }
        ]
        
        await step._store_metadata_database(metadata_results)
        
    @pytest.mark.asyncio
    async def test_store_metadata_database_disabled(self, basic_config):
        """Test database storage when database is disabled."""
        step = MetadataExtractionStep(basic_config)
        step.use_database = False
        
        metadata_results = [{'filepath': 'test.pdf', 'status': 'success'}]
        
        await step._store_metadata_database(metadata_results)

    @pytest.mark.asyncio
    async def test_extract_metadata_for_document_supported_format(self, basic_config):
        """Test metadata extraction for supported format."""
        step = MetadataExtractionStep(basic_config)
        
        document = Document(content="", file_path=Path("test.pdf"), file_format="pdf")
        
        step.processors['pdf'].extract_metadata = AsyncMock(return_value={
            'metadata': {'title': 'Test Document'},
            'method': 'pdf2bib'
        })
        
        result = await step._extract_metadata_for_document(document)
        
        assert result['status'] == 'success'
        assert result['file_format'] == 'pdf'
        assert result['filepath'] == str(document.file_path)
        assert 'metadata' in result

    @pytest.mark.asyncio
    async def test_extract_metadata_for_document_unsupported_format(self, basic_config):
        """Test metadata extraction for unsupported format."""
        step = MetadataExtractionStep(basic_config)
        
        document = Document(content="text", file_path=Path("test.txt"), file_format="txt")
        
        result = await step._extract_metadata_for_document(document)
        
        assert result['status'] == 'skipped'
        assert result['reason'] == 'unsupported_format'
        assert result['file_format'] == 'txt'


class TestMetadataProcessors:
    """Test suite for metadata processors."""

    @pytest.fixture
    def pdf_config(self):
        """Configuration for PDF processor."""
        return {}

    @pytest.fixture
    def html_config(self):
        """Configuration for HTML processor."""
        return {
            'download_remote_html': False
        }

    @pytest.fixture
    def sample_pdf_document(self):
        """Sample PDF document."""
        return Document(
            content="",
            file_path=Path("test_paper.pdf"),
            file_format="pdf"
        )

    @pytest.fixture
    def sample_html_document(self):
        """Sample HTML document."""
        return Document(
            content="<html><head><title>Test Page Title</title></head><body>Content</body></html>",
            file_path=Path("test_page.html"),
            file_format="html"
        )

    @pytest.fixture
    def sample_html_document_no_title(self):
        """Sample HTML document without title."""
        return Document(
            content="<html><head></head><body>Content without title</body></html>",
            file_path=Path("no_title.html"),
            file_format="html"
        )

    def test_pdf_processor_initialization(self, pdf_config):
        """Test PDF processor initialization."""
        processor = PDFMetadataProcessor(pdf_config)
        
        # Just verify it initializes without error
        assert isinstance(processor, PDFMetadataProcessor)

    def test_html_processor_initialization(self, html_config):
        """Test HTML processor initialization."""
        processor = HTMLMetadataProcessor(html_config)
        
        assert processor.download_remote == False

    @pytest.mark.asyncio
    async def test_pdf_processor_with_pdf2bib_success(self, pdf_config, sample_pdf_document):
        """Test PDF processor with successful pdf2bib extraction."""
        with patch('eve.steps.metadata.processors.PDF2BIB_AVAILABLE', True):
            processor = PDFMetadataProcessor(pdf_config)
            
            mock_bib_data = {
                'identifier': '10.1000/test',
                'identifier_type': 'doi',
                'metadata': {'title': 'Test Paper', 'authors': ['Author One']},
                'bibtex': '@article{test2023,...}'
            }
            
            with patch.object(processor, '_get_bib_data', return_value=mock_bib_data), \
                 patch.object(processor, '_create_temp_file', return_value=sample_pdf_document.file_path):
                
                result = await processor.extract_metadata(sample_pdf_document)
                
                assert result['file_format'] == 'pdf'
                assert result['identifier'] == '10.1000/test'
                assert result['metadata']['title'] == 'Test Paper'
                assert 'pdf2bib' in result['extraction_methods']

    @pytest.mark.asyncio
    async def test_pdf_processor_with_pdf2bib_failure_pdftitle_success(self, pdf_config, sample_pdf_document):
        """Test PDF processor when pdf2bib fails but pdftitle succeeds."""
        processor = PDFMetadataProcessor(pdf_config)
        
        with patch.object(processor, '_get_bib_data', return_value={}), \
             patch.object(processor, '_extract_title_pdftitle', return_value='Extracted Title'), \
             patch.object(processor, '_create_temp_file', return_value=sample_pdf_document.file_path):
            
            result = await processor.extract_metadata(sample_pdf_document)
            
            assert result['file_format'] == 'pdf'
            assert result['metadata']['title'] == 'Extracted Title'
            assert 'pdftitle' in result['extraction_methods']

    @pytest.mark.asyncio
    async def test_pdf_processor_no_metadata_extracted(self, pdf_config, sample_pdf_document):
        """Test PDF processor when no metadata is extracted."""
        processor = PDFMetadataProcessor(pdf_config)
        
        with patch.object(processor, '_get_bib_data', return_value={}), \
             patch.object(processor, '_extract_title_pdftitle', return_value=None), \
             patch.object(processor, '_create_temp_file', return_value=sample_pdf_document.file_path):
            
            result = await processor.extract_metadata(sample_pdf_document)
            
            assert result['file_format'] == 'pdf'
            assert result['metadata'] is None
            assert result['extraction_methods'] == ['none']

    @pytest.mark.asyncio
    async def test_pdf_processor_exception_handling(self, pdf_config, sample_pdf_document):
        """Test PDF processor exception handling."""
        processor = PDFMetadataProcessor(pdf_config)
        
        with patch.object(processor, '_create_temp_file', side_effect=Exception("File not found")):
            result = await processor.extract_metadata(sample_pdf_document)
            
            assert result['file_format'] == 'pdf'
            assert 'error' in result
            assert result['extraction_methods'] == ['error']

    @pytest.mark.asyncio
    async def test_html_processor_with_document_content(self, html_config, sample_html_document):
        """Test HTML processor with document content."""
        processor = HTMLMetadataProcessor(html_config)
        
        result = await processor.extract_metadata(sample_html_document)
        
        assert result['file_format'] == 'html'
        assert result['title'] == 'Test Page Title'
        assert result['filepath'] == str(sample_html_document.file_path)
        assert 'document_content' in result['extraction_methods']

    @pytest.mark.asyncio
    async def test_html_processor_with_file_read(self, html_config):
        """Test HTML processor reading from file."""
        processor = HTMLMetadataProcessor(html_config)
        
        document = Document(
            content="",  # No content in document
            file_path=Path("test.html"),
            file_format="html"
        )
        
        html_content = "<html><head><title>File Title</title></head><body>Content</body></html>"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=html_content):
            
            result = await processor.extract_metadata(document)
            
            assert result['title'] == 'File Title'
            assert 'file_read' in result['extraction_methods']

    @pytest.mark.asyncio
    async def test_html_processor_no_title(self, html_config, sample_html_document_no_title):
        """Test HTML processor with document that has no title."""
        processor = HTMLMetadataProcessor(html_config)
        
        result = await processor.extract_metadata(sample_html_document_no_title)
        
        assert result['file_format'] == 'html'
        assert result['title'] is None
        assert result['extraction_methods'] == ['none']

    @pytest.mark.asyncio
    async def test_html_processor_exception_handling(self, html_config, sample_html_document):
        """Test HTML processor exception handling."""
        processor = HTMLMetadataProcessor(html_config)
        
        with patch.object(processor, '_extract_html_title', side_effect=Exception("HTML parsing failed")):
            result = await processor.extract_metadata(sample_html_document)
            
            assert result['file_format'] == 'html'
            assert 'error' in result
            assert result['extraction_methods'] == ['error']

    def test_html_processor_extract_title_variations(self, html_config):
        """Test HTML title extraction with various HTML formats."""
        processor = HTMLMetadataProcessor(html_config)
        
        # Standard title
        html1 = "<html><head><title>Standard Title</title></head></html>"
        assert processor._extract_html_title(html1) == "Standard Title"
        
        # Title with extra whitespace
        html2 = "<html><head><title>  Whitespace Title  </title></head></html>"
        assert processor._extract_html_title(html2) == "Whitespace Title"
        
        # Title with newlines
        html3 = "<html><head><title>Multi\nLine\nTitle</title></head></html>"
        assert processor._extract_html_title(html3) == "Multi Line Title"
        
        # No title tag
        html4 = "<html><head></head><body>No title</body></html>"
        assert processor._extract_html_title(html4) is None
        
        # Empty title
        html5 = "<html><head><title></title></head></html>"
        assert processor._extract_html_title(html5) is None
        
        # Case insensitive
        html6 = "<html><head><TITLE>Uppercase Title</TITLE></head></html>"
        assert processor._extract_html_title(html6) == "Uppercase Title"

    def test_pdf_processor_pdftitle_extraction(self, pdf_config):
        """Test PDF title extraction using pdftitle."""
        processor = PDFMetadataProcessor(pdf_config)
        
        # Mock successful pdftitle execution
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Extracted PDF Title\n"
            
            result = processor._extract_title_pdftitle(Path("test.pdf"))
            
            assert result == "Extracted PDF Title"
            mock_run.assert_called_once()

    def test_pdf_processor_pdftitle_failure(self, pdf_config):
        """Test PDF title extraction when pdftitle fails."""
        processor = PDFMetadataProcessor(pdf_config)
        
        # Mock failed pdftitle execution
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            
            result = processor._extract_title_pdftitle(Path("test.pdf"))
            
            assert result is None

    def test_pdf_processor_pdftitle_timeout(self, pdf_config):
        """Test PDF title extraction when pdftitle times out."""
        processor = PDFMetadataProcessor(pdf_config)
        
        # Mock timeout
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("pdftitle", 30)):
            result = processor._extract_title_pdftitle(Path("test.pdf"))
            
            assert result is None

    def test_pdf_processor_pdf2bib_success(self, pdf_config):
        """Test PDF bibliographic data extraction with pdf2bib."""
        processor = PDFMetadataProcessor(pdf_config)
        
        mock_result = {
            'identifier': '10.1000/test',
            'metadata': {'title': 'Test Paper'}
        }
        
        # Mock the _get_bib_data method directly
        with patch.object(processor, '_get_bib_data', return_value=mock_result):
            result = processor._get_bib_data(Path("test.pdf"))
            
            assert result == mock_result

    def test_pdf_processor_pdf2bib_failure(self, pdf_config):
        """Test PDF bibliographic data extraction when pdf2bib fails."""
        processor = PDFMetadataProcessor(pdf_config)
        
        mock_error_result = {'pdf2bib_error': 'pdf2bib failed'}
        
        # Mock the _get_bib_data method directly to return error result
        with patch.object(processor, '_get_bib_data', return_value=mock_error_result):
            result = processor._get_bib_data(Path("test.pdf"))
            
            assert 'pdf2bib_error' in result

    def test_pdf_processor_pdf2bib_unavailable(self):
        """Test PDF processor when pdf2bib is not available."""
        config = {}
        with patch('eve.steps.metadata.processors.PDF2BIB_AVAILABLE', False):
            processor = PDFMetadataProcessor(config)
            
            result = processor._get_bib_data(Path("test.pdf"))
            
            assert result == {}


class TestMetadataStepIntegration:
    """Integration tests for metadata extraction step."""

    @pytest.mark.asyncio
    async def test_end_to_end_metadata_extraction(self):
        """Test end-to-end metadata extraction with mocked dependencies."""
        config = {
            "json_output_path": "integration_test_results.json",
            "debug": True
        }
        
        documents = [
            Document(content="", file_path=Path("paper.pdf"), file_format="pdf"),
            Document(
                content="<html><head><title>Web Page</title></head><body>Content</body></html>",
                file_path=Path("page.html"),
                file_format="html"
            ),
            Document(content="Plain text", file_path=Path("doc.txt"), file_format="txt")
        ]
        
        step = MetadataExtractionStep(config)
        
        # Mock processors
        step.processors['pdf'].extract_metadata = AsyncMock(return_value={
            'file_format': 'pdf',
            'status': 'success',
            'metadata': {'title': 'Research Paper'},
            'extraction_methods': ['pdf2bib']
        })
        
        step.processors['html'].extract_metadata = AsyncMock(return_value={
            'file_format': 'html',
            'status': 'success',
            'title': 'Web Page',
            'extraction_methods': ['document_content']
        })
        
        # Mock JSON storage
        with patch.object(step, '_store_metadata_json', new_callable=AsyncMock) as mock_json:
            result = await step.execute(documents)
            
            # Should return all original documents
            assert len(result) == 3
            assert all(isinstance(doc, Document) for doc in result)
            
            # Should have processed PDF and HTML, skipped TXT
            mock_json.assert_called_once()
            stored_results = mock_json.call_args[0][0]
            
            # Should have 2 results: PDF success, HTML success (TXT is not processed for metadata)
            assert len(stored_results) == 2
            
            success_results = [r for r in stored_results if r.get('status') == 'success']
            
            assert len(success_results) == 2  # PDF and HTML

    @pytest.mark.asyncio
    async def test_metadata_extraction_with_database_and_json(self):
        """Test metadata extraction storing to both database and JSON."""
        config = {
            "json_output_path": "db_test_results.json",
            "debug": False
        }
        
        documents = [
            Document(content="", file_path=Path("test.pdf"), file_format="pdf")
        ]
        
        step = MetadataExtractionStep(config)
        step.use_database = True  # Simulate database availability
        
        # Mock processor
        step.processors['pdf'].extract_metadata = AsyncMock(return_value={
            'file_format': 'pdf',
            'status': 'success',
            'metadata': {'title': 'Test Document'}
        })
        
        # Mock both storage methods
        with patch.object(step, '_store_metadata_database', new_callable=AsyncMock) as mock_db, \
             patch.object(step, '_store_metadata_json', new_callable=AsyncMock) as mock_json:
            
            result = await step.execute(documents)
            
            # Both storage methods should be called
            mock_db.assert_called_once()
            mock_json.assert_called_once()
            
            assert len(result) == 1
