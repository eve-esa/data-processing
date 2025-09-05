"""Tests for data cleaning components and cleaning step."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from eve.steps.cleaning.cleaning_step import CleaningStep
from eve.model.document import Document
from eve.steps.cleaning.processors import (
    OCRProcessor,
    DuplicateRemovalProcessor,
    NougatProcessor,
    RuleBasedProcessor,
    LaTeXProcessor,
)


class TestCleaningStep:
    """Test suite for the main CleaningStep class."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for cleaning step."""
        return {
            "ocr_threshold": 0.99,
            "min_words": 2,
            "enable_latex_correction": False,
            "debug": False
        }

    @pytest.fixture
    def latex_config(self):
        """Configuration with LaTeX correction enabled."""
        return {
            "ocr_threshold": 0.95,
            "min_words": 3,
            "enable_latex_correction": True,
            "openrouter_api_key": "test_key",
            "openrouter_model": "test_model",
            "debug": True
        }

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for testing."""
        return [
            Document.from_path_and_content(Path("file1.txt"), "This is  test20content with OCR issues."),
            Document.from_path_and_content(Path("file2.md"), "Some text with duplicate content.\nSome text with duplicate content.\n"),
            Document.from_path_and_content(Path("file3.tex"), "LaTeX content with $x^2$ formulas.")
        ]

    def test_cleaning_step_initialization_basic(self, basic_config):
        """Test basic initialization of cleaning step."""
        step = CleaningStep(basic_config)
        
        assert step.debug == False
        assert len(step.processors) == 4  # Without LaTeX correction
        assert step.get_component_info()["latex_correction_enabled"] == False

    def test_cleaning_step_initialization_with_latex(self, latex_config):
        """Test initialization with LaTeX correction enabled."""
        with patch('eve.steps.cleaning.processors.PDFLaTeX'):
            step = CleaningStep(latex_config)
            
            assert step.debug == True
            assert len(step.processors) == 5  # With LaTeX correction
            assert step.get_component_info()["latex_correction_enabled"] == True

    @pytest.mark.asyncio
    async def test_execute_with_valid_input(self, basic_config, sample_input_data):
        """Test execute method with valid input data."""
        step = CleaningStep(basic_config)
        
        # Mock all processor process methods
        for processor in step.processors:
            processor.process = AsyncMock(side_effect=lambda doc: Document.from_path_and_content(doc.file_path, "cleaned content"))
        
        result = await step.execute(sample_input_data)
        
        assert len(result) == 3
        assert all(isinstance(item, Document) for item in result)
        assert all(item.content == "cleaned content" for item in result)

    @pytest.mark.asyncio
    async def test_execute_with_empty_input(self, basic_config):
        """Test execute method with empty input."""
        step = CleaningStep(basic_config)
        
        result = await step.execute([])
        
        assert result == []

    @pytest.mark.asyncio
    async def test_execute_with_empty_content(self, basic_config):
        """Test execute method with files containing empty content."""
        step = CleaningStep(basic_config)
        input_data = [Document.from_path_and_content(Path("empty.txt"), "")]
        
        result = await step.execute(input_data)
        
        assert len(result) == 1
        assert result[0].content == ""  # Empty content should remain empty

    @pytest.mark.asyncio
    async def test_execute_with_component_failure(self, basic_config, sample_input_data):
        """Test execute method when a component fails."""
        step = CleaningStep(basic_config)
        
        # Mock first processor to fail, others to succeed
        step.processors[0].process = AsyncMock(side_effect=Exception("Processor failed"))
        for processor in step.processors[1:]:
            processor.process = AsyncMock(side_effect=lambda doc: Document.from_path_and_content(doc.file_path, "cleaned content"))
        
        result = await step.execute(sample_input_data)
        
        # Should still return results, continuing with other processors
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_execute_with_component_returning_none(self, basic_config, sample_input_data):
        """Test execute method when a component returns None."""
        step = CleaningStep(basic_config)
        
        # Mock first processor to return None, others to succeed
        step.processors[0].process = AsyncMock(return_value=None)
        for processor in step.processors[1:]:
            processor.process = AsyncMock(side_effect=lambda doc: Document.from_path_and_content(doc.file_path, "cleaned content"))
        
        result = await step.execute(sample_input_data)
        
        # Should fall back to original content when processor returns None
        assert len(result) == 3

    def test_get_applicable_formats(self, basic_config):
        """Test get_applicable_formats method."""
        step = CleaningStep(basic_config)
        formats = step._get_applicable_formats()
        
        expected_formats = ["md", "txt", "tex", "html", "xml"]
        assert formats == expected_formats

    def test_get_component_info(self, basic_config):
        """Test get_component_info method."""
        step = CleaningStep(basic_config)
        info = step.get_component_info()
        
        assert "total_processors" in info
        assert "processors" in info
        assert "applicable_formats" in info
        assert "debug_enabled" in info
        assert "latex_correction_enabled" in info
        
        assert info["total_processors"] == 4
        assert info["debug_enabled"] == False
        assert info["latex_correction_enabled"] == False


class TestProcessors:
    """Test suite for the consolidated processing components."""

    @pytest.mark.asyncio
    async def test_ocr_processor(self):
        """Test OCR processor functionality."""
        processor = OCRProcessor(debug=False)
        document = Document.from_path_and_content(Path("test.txt"), "Test20content")
        
        result = await processor.process(document)
        
        assert result.content == "Test20 content"  # Pattern adds space before letters after digits
        assert result.get_metadata('ocr_processed') == True

    @pytest.mark.asyncio
    async def test_duplicate_removal_processor(self):
        """Test duplicate removal processor functionality."""
        processor = DuplicateRemovalProcessor(threshold=0.99, min_words=2, debug=False)
        document = Document.from_path_and_content(
            Path("test.txt"), 
            "First line of text.\nFirst line of text.\nSecond line of text."
        )
        
        result = await processor.process(document)
        
        lines = result.content.split('\n')
        assert len(lines) == 2
        assert result.get_metadata('duplicates_removed') > 0

    @pytest.mark.asyncio
    async def test_nougat_processor(self):
        """Test Nougat processor functionality."""
        processor = NougatProcessor(debug=False)
        document = Document.from_path_and_content(
            Path("test.txt"), 
            '"Content with quotes\\nand escaped newlines"'
        )
        
        result = await processor.process(document)
        
        assert '"' not in result.content
        assert '\\n' not in result.content
        assert result.get_metadata('nougat_processed') == True

    @pytest.mark.asyncio 
    async def test_rule_based_processor(self):
        """Test rule-based processor functionality."""
        processor = RuleBasedProcessor(debug=False)
        document = Document.from_path_and_content(
            Path("test.txt"), 
            "Good line.\n!\nAnother good line.\n\n\n\nMore content."
        )
        
        result = await processor.process(document)
        
        assert "!" not in result.content  # Single symbol line removed
        assert "\n\n\n" not in result.content  # Excessive newlines normalized
        assert result.get_metadata('rule_based_processed') == True

    @pytest.mark.asyncio
    async def test_latex_processor_no_formulas(self):
        """Test LaTeX processor with no formulas."""
        with patch('eve.steps.cleaning.processors.PDFLaTeX'):
            processor = LaTeXProcessor(debug=False, api_key="test_key")
            document = Document.from_path_and_content(
                Path("test.txt"), 
                "Regular text without any LaTeX formulas."
            )
            
            result = await processor.process(document)
            
            assert result.content == document.content
            assert result.get_metadata('latex_processed') == True
