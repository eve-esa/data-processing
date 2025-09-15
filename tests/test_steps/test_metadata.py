"""Tests for metadata extraction components and metadata step."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import subprocess
import json
import tempfile

from eve.steps.metadata.metadata_step import MetadataStep
from eve.steps.metadata.extractors.pdf_extractor import PdfMetadataExtractor
from eve.steps.metadata.extractors.html_extractor import HtmlMetadataExtractor
from eve.steps.metadata.extractors.base_extractor import BaseMetadataExtractor
from eve.model.document import Document


class TestMetadataStep:
    """Test suite for the main MetadataStep class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for metadata step."""
        return {
            "enabled_formats": ["pdf", "html"],
            "fallback_to_filename": True,
            "debug": False
        }

    @pytest.fixture
    def pdf_only_config(self):
        """Configuration for PDF only."""
        return {
            "enabled_formats": ["pdf"],
            "fallback_to_filename": False,
            "debug": True
        }

    @pytest.fixture
    def sample_pdf_document(self):
        """Sample PDF document for testing."""
        return Document(
            file_path=Path("test_document.pdf"),
            content="PDF content here",
            file_format="pdf",
            metadata={}
        )

    @pytest.fixture
    def sample_html_document(self):
        """Sample HTML document for testing."""
        return Document(
            file_path=Path("test_page.html"),
            content="<html><head><title>Test Page</title></head><body>Content</body></html>",
            file_format="html",
            metadata={}
        )

    @pytest.fixture
    def sample_unsupported_document(self):
        """Sample unsupported document for testing."""
        return Document(
            file_path=Path("test_doc.txt"),
            content="Plain text content",
            file_format="txt",
            metadata={}
        )

    def test_metadata_step_initialization_basic(self, basic_config):
        """Test basic initialization of metadata step."""
        step = MetadataStep(basic_config)
        
        assert step.enabled_formats == {"pdf", "html"}
        assert step.fallback_to_filename == True
        assert step.debug == False
        assert "pdf" in step.extractors
        assert "html" in step.extractors

    def test_metadata_step_initialization_pdf_only(self, pdf_only_config):
        """Test initialization with PDF only."""
        step = MetadataStep(pdf_only_config)
        
        assert step.enabled_formats == {"pdf"}
        assert step.fallback_to_filename == False
        assert step.debug == True

    @pytest.mark.asyncio
    async def test_extract_metadata_for_pdf_document(self, basic_config, sample_pdf_document):
        """Test metadata extraction for PDF document."""
        step = MetadataStep(basic_config)
        
        # Mock the PDF extractor
        mock_metadata = {
            "title": "Test PDF Title",
            "authors": ["John Doe"],
            "doi": "10.1234/example"
        }
        
        with patch.object(step.extractors["pdf"], "extract_metadata", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_metadata
            
            result = await step._extract_metadata_for_document(sample_pdf_document)
            
            assert result.get_metadata("extracted_title") == "Test PDF Title"
            assert result.get_metadata("extracted_authors") == ["John Doe"]
            assert result.get_metadata("extracted_doi") == "10.1234/example"
            assert result.get_metadata("extracted_metadata") == mock_metadata

    @pytest.mark.asyncio
    async def test_extract_metadata_for_html_document(self, basic_config, sample_html_document):
        """Test metadata extraction for HTML document."""
        step = MetadataStep(basic_config)
        
        mock_metadata = {
            "title": "Test Page",
            "meta_tags": {"description": "Test page description"}
        }
        
        with patch.object(step.extractors["html"], "extract_metadata", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_metadata
            
            result = await step._extract_metadata_for_document(sample_html_document)
            
            assert result.get_metadata("extracted_title") == "Test Page"
            assert result.get_metadata("extracted_meta_tags") == {"description": "Test page description"}

    @pytest.mark.asyncio
    async def test_extract_metadata_unsupported_format(self, basic_config, sample_unsupported_document):
        """Test handling of unsupported document format."""
        step = MetadataStep(basic_config)
        
        result = await step._extract_metadata_for_document(sample_unsupported_document)
        
        # Should return document unchanged
        assert result == sample_unsupported_document
        assert not result.get_metadata("extracted_metadata")

    @pytest.mark.asyncio
    async def test_extract_metadata_extraction_failure_with_fallback(self, basic_config, sample_pdf_document):
        """Test handling of extraction failure with filename fallback."""
        step = MetadataStep(basic_config)
        
        with patch.object(step.extractors["pdf"], "extract_metadata", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")
            
            result = await step._extract_metadata_for_document(sample_pdf_document)
            
            # Should have filename fallback
            assert result.get_metadata("extracted_title") == "test document"
            assert result.get_metadata("extraction_error") == "Extraction failed"

    @pytest.mark.asyncio
    async def test_extract_metadata_extraction_failure_no_fallback(self, pdf_only_config, sample_pdf_document):
        """Test handling of extraction failure without filename fallback."""
        step = MetadataStep(pdf_only_config)
        
        with patch.object(step.extractors["pdf"], "extract_metadata", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")
            
            result = await step._extract_metadata_for_document(sample_pdf_document)
            
            # Should not have filename fallback
            assert not result.get_metadata("extracted_title")
            assert result.get_metadata("extraction_error") == "Extraction failed"

    @pytest.mark.asyncio
    async def test_execute_with_mixed_documents(self, basic_config, sample_pdf_document, sample_html_document, sample_unsupported_document):
        """Test execute method with mixed document types."""
        step = MetadataStep(basic_config)
        
        documents = [sample_pdf_document, sample_html_document, sample_unsupported_document]
        
        # Mock extractors
        with patch.object(step.extractors["pdf"], "extract_metadata", new_callable=AsyncMock) as mock_pdf, \
             patch.object(step.extractors["html"], "extract_metadata", new_callable=AsyncMock) as mock_html:
            
            mock_pdf.return_value = {"title": "PDF Title"}
            mock_html.return_value = {"title": "HTML Title"}
            
            result = await step.execute(documents)
            
            assert len(result) == 3
            assert result[0].get_metadata("extracted_title") == "PDF Title"
            assert result[1].get_metadata("extracted_title") == "HTML Title"
            assert not result[2].get_metadata("extracted_metadata")  # Unsupported format

    @pytest.mark.asyncio
    async def test_execute_with_empty_list(self, basic_config):
        """Test execute method with empty document list."""
        step = MetadataStep(basic_config)
        
        result = await step.execute([])
        
        assert result == []

    def test_get_supported_formats(self, basic_config):
        """Test get_supported_formats method."""
        step = MetadataStep(basic_config)
        
        formats = step.get_supported_formats()
        assert set(formats) == {"pdf", "html"}

    def test_get_extractor_info(self, basic_config):
        """Test get_extractor_info method."""
        step = MetadataStep(basic_config)
        
        info = step.get_extractor_info()
        
        assert "supported_formats" in info
        assert "available_extractors" in info
        assert "fallback_enabled" in info
        assert "debug_enabled" in info
        assert set(info["supported_formats"]) == {"pdf", "html"}


class TestBaseMetadataExtractor:
    """Test suite for the base metadata extractor."""

    class MockExtractor(BaseMetadataExtractor):
        async def extract_metadata(self, document):
            return {"test": "metadata"}

    def test_clean_title_valid(self):
        """Test title cleaning with valid input."""
        extractor = self.MockExtractor()
        
        # Test normal title
        assert extractor._clean_title("Valid Title") == "Valid Title"
        
        # Test title with extra whitespace
        assert extractor._clean_title("  Title with spaces  ") == "Title with spaces"
        
        # Test title with newlines
        assert extractor._clean_title("Title\nwith\nnewlines") == "Title with newlines"
        
        # Test title with multiple spaces
        assert extractor._clean_title("Title  with   multiple    spaces") == "Title with multiple spaces"

    def test_clean_title_invalid(self):
        """Test title cleaning with invalid input."""
        extractor = self.MockExtractor()
        
        # Test empty/None input
        assert extractor._clean_title("") is None
        assert extractor._clean_title(None) is None
        assert extractor._clean_title("   ") is None
        
        # Test numeric only
        assert extractor._clean_title("123") is None
        
        # Test too short
        assert extractor._clean_title("ab") is None

    def test_extract_title_from_filename(self):
        """Test filename to title conversion."""
        extractor = self.MockExtractor()
        
        # Test normal filename
        assert extractor._extract_title_from_filename(Path("test_document.pdf")) == "Test Document"
        
        # Test filename with dashes
        assert extractor._extract_title_from_filename(Path("my-research-paper.html")) == "My Research Paper"
        
        # Test filename with dots
        assert extractor._extract_title_from_filename(Path("file.name.with.dots.txt")) == "File Name With Dots"


class TestPdfMetadataExtractor:
    """Test suite for PDF metadata extractor."""

    @pytest.fixture
    def pdf_extractor(self):
        """PDF extractor instance for testing."""
        return PdfMetadataExtractor(debug=False)

    @pytest.fixture
    def sample_pdf_document(self):
        """Sample PDF document."""
        return Document(
            file_path=Path("test.pdf"),
            content="PDF content",
            file_format="pdf"
        )

    def test_pdf_extractor_initialization(self, pdf_extractor):
        """Test PDF extractor initialization."""
        assert isinstance(pdf_extractor, PdfMetadataExtractor)
        assert pdf_extractor.debug == False

    @pytest.mark.asyncio
    async def test_extract_metadata_with_pdf2bib_success(self, pdf_extractor, sample_pdf_document):
        """Test successful metadata extraction with pdf2bib."""
        mock_bibtex_data = {
            "identifier": "10.1234/example",
            "identifier_type": "doi",
            "metadata": {
                "title": "Test Paper Title",
                "author": ["John Doe", "Jane Smith"],
                "year": "2023",
                "journal": "Test Journal"
            },
            "bibtex": "@article{...}"
        }
        
        with patch.object(pdf_extractor, "_extract_bibtex_metadata", new_callable=AsyncMock) as mock_bib:
            mock_bib.return_value = mock_bibtex_data
            
            result = await pdf_extractor.extract_metadata(sample_pdf_document)
            
            assert result["title"] == "Test Paper Title"
            assert result["authors"] == ["John Doe", "Jane Smith"]
            assert result["year"] == "2023"
            assert result["journal"] == "Test Journal"
            assert result["identifier"] == "10.1234/example"
            assert "pdf2bib" in result["extraction_methods"]

    @pytest.mark.asyncio
    async def test_extract_metadata_with_pdf_title_fallback(self, pdf_extractor, sample_pdf_document):
        """Test title extraction fallback to Python PDF reader."""
        
        with patch.object(pdf_extractor, "_extract_bibtex_metadata", new_callable=AsyncMock) as mock_bib, \
             patch.object(pdf_extractor, "_extract_title_from_pdf", new_callable=AsyncMock) as mock_title:
            
            mock_bib.return_value = None  # No bibtex data
            mock_title.return_value = "Extracted Title"
            
            result = await pdf_extractor.extract_metadata(sample_pdf_document)
            
            assert result["title"] == "Extracted Title"
            assert result["title_source"] == "extracted"

    @pytest.mark.asyncio
    async def test_extract_metadata_filename_fallback(self, pdf_extractor, sample_pdf_document):
        """Test filename fallback when no extraction methods work."""
        
        with patch.object(pdf_extractor, "_extract_bibtex_metadata", new_callable=AsyncMock) as mock_bib, \
             patch.object(pdf_extractor, "_extract_title_from_pdf", new_callable=AsyncMock) as mock_title:
            
            mock_bib.return_value = None
            mock_title.return_value = None
            
            result = await pdf_extractor.extract_metadata(sample_pdf_document)
            
            assert result["title"] == "Test"  # From filename
            assert result["title_source"] == "filename"

    @pytest.mark.asyncio
    async def test_extract_metadata_wrong_format(self, pdf_extractor):
        """Test handling of non-PDF document."""
        html_doc = Document(
            file_path=Path("test.html"),
            content="HTML content",
            file_format="html"
        )
        
        result = await pdf_extractor.extract_metadata(html_doc)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_title_from_pdf_success(self, pdf_extractor):
        """Test successful title extraction from PDF using pdfplumber."""
        
        mock_pdf = Mock()
        mock_pdf.metadata = {'Title': 'PDF Metadata Title'}
        mock_pdf.pages = []
        mock_pdf.__enter__ = Mock(return_value=mock_pdf)
        mock_pdf.__exit__ = Mock(return_value=None)
        
        with patch("pdfplumber.open", return_value=mock_pdf):
            result = await pdf_extractor._extract_title_from_pdf("test.pdf")
            
            assert result == "PDF Metadata Title"

    @pytest.mark.asyncio
    async def test_extract_title_from_pdf_failure(self, pdf_extractor):
        """Test title extraction when pdfplumber fails."""
        
        with patch("pdfplumber.open", side_effect=Exception("pdfplumber failed")):
            result = await pdf_extractor._extract_title_from_pdf("test.pdf")
            
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_bibtex_metadata_success(self, pdf_extractor):
        """Test successful bibtex extraction."""
        
        mock_bib_data = {
            "identifier": "10.1234/example",
            "metadata": {"title": "Test Title"}
        }
        
        with patch("pdf2bib.pdf2bib", return_value=mock_bib_data):
            result = await pdf_extractor._extract_bibtex_metadata("test.pdf")
            
            assert result == mock_bib_data

    @pytest.mark.asyncio
    async def test_extract_bibtex_metadata_no_data(self, pdf_extractor):
        """Test bibtex extraction when no data found."""
        
        with patch("pdf2bib.pdf2bib", return_value=None):
            result = await pdf_extractor._extract_bibtex_metadata("test.pdf")
            
            assert result is None


class TestHtmlMetadataExtractor:
    """Test suite for HTML metadata extractor."""

    @pytest.fixture
    def html_extractor(self):
        """HTML extractor instance for testing."""
        return HtmlMetadataExtractor(debug=False)

    @pytest.fixture
    def sample_html_document(self):
        """Sample HTML document."""
        html_content = """
        <html>
            <head>
                <title>Sample HTML Page</title>
                <meta name="description" content="This is a test page">
                <meta name="author" content="John Doe">
                <meta property="og:title" content="Social Media Title">
            </head>
            <body>
                <h1>Content</h1>
            </body>
        </html>
        """
        return Document(
            file_path=Path("test.html"),
            content=html_content,
            file_format="html",
            metadata={"url": "https://example.com/test.html"}
        )

    def test_html_extractor_initialization(self, html_extractor):
        """Test HTML extractor initialization."""
        assert isinstance(html_extractor, HtmlMetadataExtractor)
        assert html_extractor.debug == False

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, html_extractor, sample_html_document):
        """Test successful HTML metadata extraction."""
        
        result = await html_extractor.extract_metadata(sample_html_document)
        
        assert result["title"] == "Sample HTML Page"
        assert result["title_source"] == "html_tag"
        assert "meta_tags" in result
        assert result["meta_tags"]["description"] == "This is a test page"
        assert result["meta_tags"]["author"] == "John Doe"
        assert result["url"] == "https://example.com/test.html"
        assert result["domain"] == "example.com"

    def test_extract_title_from_html_success(self, html_extractor):
        """Test successful title extraction from HTML."""
        html_content = "<html><head><title>Test Title</title></head></html>"
        
        result = html_extractor._extract_title_from_html(html_content)
        
        assert result == "Test Title"

    def test_extract_title_from_html_with_nested_tags(self, html_extractor):
        """Test title extraction with nested HTML tags."""
        html_content = "<html><head><title>Title with <span>nested</span> tags</title></head></html>"
        
        result = html_extractor._extract_title_from_html(html_content)
        
        assert "nested" in result
        assert "<span>" not in result

    def test_extract_title_from_html_no_title(self, html_extractor):
        """Test title extraction when no title tag exists."""
        html_content = "<html><head></head><body>No title here</body></html>"
        
        result = html_extractor._extract_title_from_html(html_content)
        
        assert result is None

    def test_extract_meta_tags(self, html_extractor):
        """Test meta tag extraction."""
        html_content = """
        <html>
            <head>
                <meta name="description" content="Test description">
                <meta name="keywords" content="test, html, meta">
                <meta property="og:title" content="OG Title">
            </head>
        </html>
        """
        
        result = html_extractor._extract_meta_tags(html_content)
        
        assert result["description"] == "Test description"
        assert result["keywords"] == "test, html, meta"
        assert result["og_title"] == "OG Title"

    def test_extract_structured_data(self, html_extractor):
        """Test structured data detection."""
        html_content = """
        <html>
            <head>
                <script type="application/ld+json">{"@type": "Article"}</script>
                <script type="application/ld+json">{"@type": "Person"}</script>
            </head>
        </html>
        """
        
        result = html_extractor._extract_structured_data(html_content)
        
        assert result["json_ld_count"] == 2

    @pytest.mark.asyncio
    async def test_extract_metadata_wrong_format(self, html_extractor):
        """Test handling of non-HTML document."""
        pdf_doc = Document(
            file_path=Path("test.pdf"),
            content="PDF content",
            file_format="pdf"
        )
        
        result = await html_extractor.extract_metadata(pdf_doc)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_metadata_filename_fallback(self, html_extractor):
        """Test filename fallback for title."""
        html_doc = Document(
            file_path=Path("test_page.html"),
            content="<html><body>No title</body></html>",
            file_format="html"
        )
        
        result = await html_extractor.extract_metadata(html_doc)
        
        assert result["title"] == "Test Page"
        assert result["title_source"] == "filename"


class TestMetadataStepWithJsonExport:
    """Test suite for the metadata step with JSON export functionality."""

    @pytest.fixture
    def metadata_config_with_export(self):
        """Configuration for metadata step with JSON export."""
        return {
            "enabled_formats": ["pdf", "html"],
            "fallback_to_filename": True,
            "debug": False,
            "export_metadata": True,
            "metadata_destination": "test_output",
            "metadata_filename": "test_metadata.json"
        }

    @pytest.fixture
    def sample_documents_for_export(self):
        """Sample documents for testing JSON export."""
        doc1 = Document(
            file_path=Path("test1.pdf"),
            content="Content of document 1",
            file_format="pdf"
        )
        
        doc2 = Document(
            file_path=Path("test2.html"),
            content="Content of document 2",
            file_format="html"
        )

        return [doc1, doc2]

    @pytest.mark.asyncio
    async def test_metadata_step_with_json_export_enabled(self, metadata_config_with_export, sample_documents_for_export):
        """Test metadata step with JSON export enabled."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_config_with_export["metadata_destination"] = temp_dir
            
            # Mock the extractors to return test metadata
            with patch("eve.steps.metadata.extractors.pdf_extractor.PdfMetadataExtractor"), \
                 patch("eve.steps.metadata.extractors.html_extractor.HtmlMetadataExtractor"):
                
                metadata_step = MetadataStep(metadata_config_with_export)
                
                # Mock extractor methods
                mock_pdf_metadata = {"title": "Test PDF Title", "authors": ["John Doe"]}
                mock_html_metadata = {"title": "Test HTML Title", "url": "https://example.com"}
                
                metadata_step.extractors["pdf"].extract_metadata = AsyncMock(return_value=mock_pdf_metadata)
                metadata_step.extractors["html"].extract_metadata = AsyncMock(return_value=mock_html_metadata)
                
                result = await metadata_step.execute(sample_documents_for_export)
                
                # Check that documents were processed
                assert len(result) == 2
                
                # Check that metadata file was created
                metadata_file = Path(temp_dir) / "test_metadata.json"
                assert metadata_file.exists()
                
                # Verify metadata content
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                assert metadata["export_info"]["total_documents"] == 2
                assert len(metadata["documents"]) == 2

    @pytest.mark.asyncio
    async def test_metadata_step_with_json_export_disabled(self, sample_documents_for_export):
        """Test metadata step with JSON export disabled."""
        
        config = {
            "enabled_formats": ["pdf", "html"],
            "export_metadata": False
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["metadata_destination"] = temp_dir
            
            with patch("eve.steps.metadata.extractors.pdf_extractor.PdfMetadataExtractor"), \
                 patch("eve.steps.metadata.extractors.html_extractor.HtmlMetadataExtractor"):
                
                metadata_step = MetadataStep(config)
                metadata_step.extractors["pdf"].extract_metadata = AsyncMock(return_value={"title": "Test"})
                metadata_step.extractors["html"].extract_metadata = AsyncMock(return_value={"title": "Test"})
                
                await metadata_step.execute(sample_documents_for_export)
                
                # Check that metadata file was NOT created
                metadata_file = Path(temp_dir) / "metadata.json"
                assert not metadata_file.exists()

    @pytest.mark.asyncio
    async def test_metadata_step_with_custom_export_settings(self, sample_documents_for_export):
        """Test metadata step with custom export settings."""
        
        config = {
            "enabled_formats": ["pdf"],
            "export_metadata": True,
            "metadata_filename": "custom_metadata.json"
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config["metadata_destination"] = temp_dir
            
            with patch("eve.steps.metadata.extractors.pdf_extractor.PdfMetadataExtractor"):
                metadata_step = MetadataStep(config)
                metadata_step.extractors["pdf"].extract_metadata = AsyncMock(return_value={"title": "Test PDF"})
                
                # Only process PDF document
                pdf_doc = sample_documents_for_export[0]
                await metadata_step.execute([pdf_doc])
                
                # Check custom filename
                metadata_file = Path(temp_dir) / "custom_metadata.json"
                assert metadata_file.exists()
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                assert metadata["export_info"]["total_documents"] == 1
                assert metadata["documents"][0]["file_format"] == "pdf"

    def test_get_extractor_info_includes_export_settings(self):
        """Test that get_extractor_info includes export configuration."""
        config = {
            "export_metadata": True,
            "metadata_destination": "./custom_output",
            "metadata_filename": "custom.json"
        }
        
        with patch("eve.steps.metadata.extractors.pdf_extractor.PdfMetadataExtractor"), \
             patch("eve.steps.metadata.extractors.html_extractor.HtmlMetadataExtractor"):
            
            metadata_step = MetadataStep(config)
            info = metadata_step.get_extractor_info()
            
            assert info["export_metadata"] == True
            assert info["metadata_destination"] == "custom_output"  # Path strips ./ prefix
            assert info["metadata_filename"] == "custom.json"
