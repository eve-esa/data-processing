import tempfile
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from eve.steps.extraction.extract_step import ExtractionStep


@pytest.fixture
def temp_html_file():
    with tempfile.NamedTemporaryFile(mode = "w", delete = False, suffix = ".html") as f:
        f.write("<html><body><p>This is a HTML document</p></body></html>")
        f.close()
        yield f.name


@pytest.fixture
def temp_xml_file():
    with tempfile.NamedTemporaryFile(mode = "w", delete = False, suffix = ".xml") as f:
        f.write("<root><child>This is a XML document</child></root>")
        f.close()
        yield f.name


@pytest.fixture
def temp_pdf_file():
    with tempfile.NamedTemporaryFile(mode = "wb", delete = False, suffix = ".pdf") as f:
        f.write(b"%PDF-1.4\n%Fake PDF content")
        f.close()
        yield f.name

@pytest.fixture
def temp_dir():
    """temporary directory for test output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.mark.asyncio
async def test_html_extraction(temp_html_file):
    step = ExtractionStep(config={"format": "html"})
    # patch HtmlExtractor to avoid calling trafilatura
    with patch("eve.steps.extraction.extract_step.HtmlExtractor") as MockHtmlExtractor:
        instance = MockHtmlExtractor.return_value
        instance.extract_text = AsyncMock(return_value="Hello World")
        result = await step.execute([Path(temp_html_file)])
        assert len(result) == 1
        assert result[0][1] == "Hello World"


@pytest.mark.asyncio
async def test_xml_extraction(temp_xml_file):
    step = ExtractionStep(config = {"format": "xml"})
    result = await step.execute([Path(temp_xml_file)])
    assert len(result) == 1
    assert "This is a XML document" in result[0][1]


@pytest.mark.asyncio
async def test_pdf_extraction(temp_pdf_file):
    step = ExtractionStep(config={"format": "pdf", "url": "http://fake-endpoint"})

    with patch("eve.steps.extraction.extract_step.PdfExtractor") as MockPdfExtractor:
        instance = MockPdfExtractor.return_value
        instance.extract_text = AsyncMock(return_value="PDF Extracted Text")
        
        result = await step.execute([Path(temp_pdf_file)])
        
        assert len(result) == 1
        assert result[0][1] == "PDF Extracted Text"


@pytest.mark.asyncio
async def test_pdf_extraction_failure(temp_pdf_file):
    step = ExtractionStep(config = {"format": "pdf", "url": "http://fake-endpoint"})
    with patch("eve.steps.extraction.extract_step.PdfExtractor") as MockPdfExtractor:
        instance = MockPdfExtractor.return_value
        instance.extract_text = AsyncMock(side_effect=Exception("Extraction failed"))
        
        result = await step.execute([Path(temp_pdf_file)])
        assert result == []  # Failed extractions are not included in result


@pytest.mark.asyncio
async def test_invalid_format(temp_html_file):
    step = ExtractionStep(config = {"format": "invalid"})
    # The ValueError is caught and logged, so no exception is raised
    result = await step.execute([Path(temp_html_file)])
    assert result == []  # Failed extractions are not included in result
