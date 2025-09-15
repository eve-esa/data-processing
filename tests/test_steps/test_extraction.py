#TO-DO find a nicer way to test multiple formats

import pytest
from unittest.mock import patch, AsyncMock
from pathlib import Path

from eve.steps.extraction.extract_step import ExtractionStep
from eve.model.document import Document


@pytest.fixture  
def extraction_step():
    return ExtractionStep()


@pytest.mark.asyncio
async def test_html_extraction():
    """test HTML extraction."""
    step = ExtractionStep(config={"format": "html"})
    
    test_doc = Document(
        file_path = Path("test.html"), 
        content = "<html><body><p>Hello World</p></body></html>",
        file_format = "html"
    )

    # Mock HtmlExtractor to avoid calling trafilatura
    with patch("eve.steps.extraction.extract_step.HtmlExtractor") as MockHtmlExtractor:
        instance = MockHtmlExtractor.return_value
        
        mock_doc = Document(
            file_path = test_doc.file_path,
            content = "Hello World",
            file_format = "html"
        )
        instance.extract_text = AsyncMock(return_value = mock_doc)

        result = await step.execute([test_doc])
        
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].content == "Hello World"

@pytest.mark.asyncio
async def test_pdf_extraction():
    """test HTML extraction."""
    step = ExtractionStep(config={"format": "pdf"})
    
    test_doc = Document(
        file_path = Path("test.pdf"), 
        content = "Hello World",
        file_format = "pdf"
    )

    # Mock HtmlExtractor to avoid calling trafilatura
    with patch("eve.steps.extraction.extract_step.PdfExtractor") as MockHtmlExtractor:
        instance = MockHtmlExtractor.return_value
        
        mock_doc = Document(
            file_path = test_doc.file_path,
            content = "Hello World",
            file_format = "pdf"
        )
        instance.extract_text = AsyncMock(return_value = mock_doc)

        result = await step.execute([test_doc])
        
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].content == "Hello World"

@pytest.mark.asyncio
async def test_xml_extraction():
    """test HTML extraction."""
    step = ExtractionStep(config={"format": "xml"})
    
    test_doc = Document(
        file_path = Path("test.xml"), 
        content = "Hello World",
        file_format = "xml"
    )

    # Mock HtmlExtractor to avoid calling trafilatura
    with patch("eve.steps.extraction.extract_step.XmlExtractor") as MockHtmlExtractor:
        instance = MockHtmlExtractor.return_value
        
        mock_doc = Document(
            file_path = test_doc.file_path,
            content = "Hello World",
            file_format = "xml"
        )
        instance.extract_text = AsyncMock(return_value = mock_doc)

        result = await step.execute([test_doc])
        
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].content == "Hello World"