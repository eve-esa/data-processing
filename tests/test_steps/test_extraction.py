import tempfile
import pytest
from unittest.mock import patch, MagicMock
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

def test_html_extraction(temp_html_file):
    step = ExtractionStep(config={"format": "html"})
    # patch HtmlExtractor to avoid calling trafilatura
    with patch("eve.steps.extraction.extract_step.HtmlExtractor") as MockHtmlExtractor:
        instance = MockHtmlExtractor.return_value
        instance.extract_text.return_value = ["Hello World"]
        result = step.execute([temp_html_file])
        assert result == ["Hello World"]


def test_xml_extraction(temp_xml_file):
    step = ExtractionStep(config = {"format": "xml"})
    result = step.execute([temp_xml_file])
    assert any("This is a XML document" in r for r in result)


def test_pdf_extraction(temp_pdf_file):
    step = ExtractionStep(config={"format": "pdf", "url": "http://fake-endpoint"})

    with patch("eve.steps.extraction.pdfs.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "PDF Extracted Text"
        mock_post.return_value = mock_response

        result = step.execute([Path(temp_pdf_file)]) # explicitly wrap in Path

        assert result == ["PDF Extracted Text"]


def test_pdf_extraction_failure(temp_pdf_file):
    step = ExtractionStep(config = {"format": "pdf", "url": "http://fake-endpoint"})
    with patch("eve.steps.extraction.pdfs.requests.post") as mock_post:
        mock_post.return_value.status_code = 500
        result = step.execute([temp_pdf_file])
        assert result == [None]


def test_invalid_format(temp_html_file):
    step = ExtractionStep(config = {"format": "invalid"})
    with pytest.raises(ValueError, match = "unsupported format: invalid"):
        step.execute([temp_html_file])
