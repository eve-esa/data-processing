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
        content = "<html><body><p>This is a HTML document</p></body></html>",
        file_format = 'html'
    )
    
    # Mock HtmlExtractor to avoid calling trafilatura
    with patch("eve.steps.extraction.extract_step.HtmlExtractor") as MockHtmlExtractor:
        instance = MockHtmlExtractor.return_value
        instance.extract_text = AsyncMock(return_value="Hello World")
        result = await step.execute([test_doc])
        
        assert len(result) == 1
        assert isinstance(result[0], Document)
        assert result[0].content == "Hello World"


# @pytest.mark.asyncio
# async def test_xml_extraction():
#     """Test XML extraction."""
#     step = ExtractionStep(config={"format": "xml"})
    
#     # Create a test document  
#     test_doc = Document(
#         file_path = Path("test.xml"),
#         content = "<root><child>This is a XML document</child></root>",
#         file_format = 'xml'
#     )
    
#     result = await step.execute([test_doc])
    
#     assert len(result) == 1
#     assert isinstance(result[0], Document)
#     assert "This is a XML document" in result[0].content


# @pytest.mark.asyncio
# async def test_pdf_extraction():
#     """Test PDF extraction."""
#     step = ExtractionStep(config={"format": "pdf", "url": "http://fake-endpoint"})
    
#     # Create a test document
#     test_doc = Document(
#         file_path = Path("test.pdf"),
#         content = b"%PDF-1.4\n%Fake PDF content",
#         file_format = 'pdf'
#     )

#     with patch("eve.steps.extraction.extract_step.PdfExtractor") as MockPdfExtractor:
#         instance = MockPdfExtractor.return_value
#         instance.extract_text = AsyncMock(return_value="PDF Extracted Text")
        
#         result = await step.execute([test_doc])
#         print(result)
        
#         assert len(result) == 1
#         assert isinstance(result[0], Document)
#         assert result[0] == "PDF Extracted Text"


# @pytest.mark.asyncio
# async def test_pdf_extraction_failure():
#     """Test PDF extraction failure handling."""
#     step = ExtractionStep(config={"format": "pdf", "url": "http://fake-endpoint"})
    
#     # Create a test document
#     test_doc = Document(
#         Path("test.pdf"),
#         b"%PDF-1.4\n%Fake PDF content",
#         file_format = 'pdf'
#     )
    
#     with patch("eve.steps.extraction.extract_step.PdfExtractor") as MockPdfExtractor:
#         instance = MockPdfExtractor.return_value
#         instance.extract_text = AsyncMock(side_effect=Exception("Extraction failed"))
        
#         result = await step.execute([test_doc])
#         assert result == []  # Failed extractions are not included in result


# @pytest.mark.asyncio
# async def test_invalid_format(temp_files):
#     """Test invalid format handling."""
#     step = ExtractionStep(config={"format": "invalid"})
#     # The ValueError is caught and logged, so no exception is raised
#     result = await step.execute(temp_files)
#     assert result == []  # Failed extractions are not included in result


# @pytest.mark.asyncio
# async def test_execute_with_empty_input():
#     """Test execute method with empty input."""
#     step = ExtractionStep(config={"format": "txt"})
#     result = await step.execute([])
#     assert result == []


# @pytest.mark.asyncio
# async def test_execute_with_valid_input(temp_files, extraction_step):
#     """Test execute method with valid input data."""
#     result = await extraction_step.execute(temp_files)
    
#     assert len(result) >= 0  # Should handle the input without errors
#     assert all(isinstance(item, Document) for item in result)