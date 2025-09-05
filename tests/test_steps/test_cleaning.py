"""Tests for data cleaning components and cleaning step."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from eve.steps.cleaning.cleaning_step import CleaningStep
from eve.steps.cleaning.ocr_corrections import OCRCorrections
from eve.steps.cleaning.ocr_duplicate_remover import OCRDuplicateRemover
from eve.steps.cleaning.nougat_correction import NougatCorrection
from eve.steps.cleaning.rule_based_corrections import RuleBasedCorrections
from eve.steps.cleaning.nougat_artifact_removal import NougatArtifactRemovalComponent
from eve.steps.cleaning.latex_correction import LatexCorrectionComponent


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
            (Path("file1.txt"), "This is  test20content with OCR issues."),
            (Path("file2.md"), "Some text with duplicate content.\nSome text with duplicate content.\n"),
            (Path("file3.tex"), "LaTeX content with $x^2$ formulas.")
        ]

    def test_cleaning_step_initialization_basic(self, basic_config):
        """Test basic initialization of cleaning step."""
        step = CleaningStep(basic_config)
        
        assert step.debug == False
        assert len(step.components) == 5  # Without LaTeX correction
        assert step.get_component_info()["latex_correction_enabled"] == False

    def test_cleaning_step_initialization_with_latex(self, latex_config):
        """Test initialization with LaTeX correction enabled."""
        with patch('eve.steps.cleaning.latex_correction.PDFLaTeX'):
            step = CleaningStep(latex_config)
            
            assert step.debug == True
            assert len(step.components) == 6  # With LaTeX correction
            assert step.get_component_info()["latex_correction_enabled"] == True

    @pytest.mark.asyncio
    async def test_execute_with_valid_input(self, basic_config, sample_input_data):
        """Test execute method with valid input data."""
        step = CleaningStep(basic_config)
        
        # Mock all component process methods
        for component in step.components:
            component.process = AsyncMock(return_value="cleaned content")
        
        result = await step.execute(sample_input_data)
        
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(item[1] == "cleaned content" for item in result)

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
        input_data = [(Path("empty.txt"), "")]
        
        result = await step.execute(input_data)
        
        assert len(result) == 1
        assert result[0][1] == ""  # Empty content should remain empty

    @pytest.mark.asyncio
    async def test_execute_with_component_failure(self, basic_config, sample_input_data):
        """Test execute method when a component fails."""
        step = CleaningStep(basic_config)
        
        # Mock first component to fail, others to succeed
        step.components[0].process = AsyncMock(side_effect=Exception("Component failed"))
        for component in step.components[1:]:
            component.process = AsyncMock(return_value="cleaned content")
        
        result = await step.execute(sample_input_data)
        
        # Should still return results, continuing with other components
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_execute_with_component_returning_none(self, basic_config, sample_input_data):
        """Test execute method when a component returns None."""
        step = CleaningStep(basic_config)
        
        # Mock first component to return None, others to succeed
        step.components[0].process = AsyncMock(return_value=None)
        for component in step.components[1:]:
            component.process = AsyncMock(return_value="cleaned content")
        
        result = await step.execute(sample_input_data)
        
        # Should fall back to original content when component returns None
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
        
        assert "total_components" in info
        assert "components" in info
        assert "applicable_formats" in info
        assert "debug_enabled" in info
        assert "latex_correction_enabled" in info
        
        assert info["total_components"] == 5
        assert info["debug_enabled"] == False
        assert info["latex_correction_enabled"] == False


class TestOCRCorrections:
    """Test suite for OCRCorrections component."""

    @pytest.fixture
    def ocr_component(self):
        return OCRCorrections(debug=False)

    @pytest.mark.asyncio
    async def test_basic_ocr_correction(self, ocr_component):
        """Test basic OCR correction functionality."""
        input_text = "Test content with 20Papers and 100Documents."
        result = await ocr_component.process(input_text, "test.txt")
        
        expected = "Test content with 20 Papers and 100 Documents."
        assert result == expected

    @pytest.mark.asyncio
    async def test_preserves_abbreviations(self, ocr_component):
        """Test that abbreviations like 20M, 100k are preserved."""
        input_text = "Dataset has 20M samples and 100k features."
        result = await ocr_component.process(input_text, "test.txt")
        
        # Should not change abbreviations
        assert result == input_text

    @pytest.mark.asyncio
    async def test_empty_content(self, ocr_component):
        """Test handling of empty content."""
        result = await ocr_component.process("", "test.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_ocr_correction_with_debug(self):
        """Test OCR correction with debug enabled."""
        component = OCRCorrections(debug=True)
        input_text = "Test20content"
        
        result = await component.process(input_text, "test.txt")
        assert result == "Test 20 content"


class TestOCRDuplicateRemover:
    """Test suite for OCRDuplicateRemover component."""

    @pytest.fixture
    def ocr_duplicate_remover(self):
        return OCRDuplicateRemover(threshold=0.99, min_words=2, debug=False)

    @pytest.mark.asyncio
    async def test_removes_exact_duplicates(self, ocr_duplicate_remover):
        """Test removal of exact duplicate lines."""
        input_text = "First line of text.\nFirst line of text.\nSecond line of text."
        result = await ocr_duplicate_remover.process(input_text, "test.txt")
        
        # Should remove the duplicate
        lines = result.split('\n')
        assert len(lines) == 2
        assert "First line of text." in lines
        assert "Second line of text." in lines

    @pytest.mark.asyncio
    async def test_removes_similar_duplicates(self, ocr_duplicate_remover):
        """Test removal of similar duplicate lines."""
        component = OCRDuplicateRemover(threshold=0.8, min_words=2, debug=False)
        input_text = "The quick brown fox.\nThe quick brown fox jumps.\nCompletely different line."
        result = await component.process(input_text, "test.txt")
        
        lines = result.split('\n')
        assert len(lines) == 2  # One duplicate should be removed

    @pytest.mark.asyncio
    async def test_preserves_short_lines(self, ocr_duplicate_remover):
        """Test that short lines below min_words threshold are preserved."""
        input_text = "A\nA\nLonger line of text."
        result = await ocr_duplicate_remover.process(input_text, "test.txt")
        
        # Short lines should be preserved even if duplicate
        lines = result.split('\n')
        assert lines.count("A") == 2

    @pytest.mark.asyncio
    async def test_empty_content(self, ocr_duplicate_remover):
        """Test handling of empty content."""
        result = await ocr_duplicate_remover.process("", "test.txt")
        assert result is None


class TestNougatCorrection:
    """Test suite for NougatCorrection component."""

    @pytest.fixture
    def nougat_component(self):
        return NougatCorrection(debug=False)

    @pytest.mark.asyncio
    async def test_basic_nougat_correction(self, nougat_component):
        """Test basic Nougat correction functionality."""
        with patch('eve.steps.cleaning.nougat_correction.postprocess_single') as mock_postprocess:
            mock_postprocess.return_value = "processed content"
            
            input_text = "Some raw nougat content"
            result = await nougat_component.process(input_text, "test.txt")
            
            mock_postprocess.assert_called_once_with(input_text, markdown_fix=True)
            assert result == "processed content"

    @pytest.mark.asyncio
    async def test_latex_table_cleaning(self, nougat_component):
        """Test LaTeX table cleaning functionality."""
        with patch('eve.steps.cleaning.nougat_correction.postprocess_single') as mock_postprocess:
            mock_postprocess.return_value = "\\\\begin{table}\\\\\\end{table}"
            
            input_text = "table content"
            result = await nougat_component.process(input_text, "test.txt")
            
            # Should clean up doubled backslashes
            assert "\\begin{table}\\end{table}" in result

    @pytest.mark.asyncio
    async def test_empty_content(self, nougat_component):
        """Test handling of empty content."""
        result = await nougat_component.process("", "test.txt")
        assert result is None


class TestRuleBasedCorrections:
    """Test suite for RuleBasedCorrections component."""

    @pytest.fixture
    def rule_component(self):
        return RuleBasedCorrections(debug=False)

    @pytest.mark.asyncio
    async def test_removes_single_symbol_lines(self, rule_component):
        """Test removal of single symbol/punctuation lines."""
        input_text = "Good line.\n!\nAnother good line.\n?"
        result = await rule_component.process(input_text, "test.txt")
        
        lines = result.split('\n')
        assert len(lines) == 2
        assert "Good line." in lines
        assert "Another good line." in lines

    @pytest.mark.asyncio
    async def test_normalizes_excessive_newlines(self, rule_component):
        """Test normalization of excessive newlines."""
        input_text = "Line 1.\n\n\n\nLine 2.\n\n\n\n\nLine 3."
        result = await rule_component.process(input_text, "test.txt")
        
        # Should convert 3+ newlines to exactly 2
        assert "\n\n\n" not in result
        assert "\n\n" in result

    @pytest.mark.asyncio
    async def test_trims_whitespace(self, rule_component):
        """Test trimming of leading and trailing whitespace."""
        input_text = "   \n  Content with whitespace.  \n   "
        result = await rule_component.process(input_text, "test.txt")
        
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    @pytest.mark.asyncio
    async def test_empty_content(self, rule_component):
        """Test handling of empty content."""
        result = await rule_component.process("", "test.txt")
        assert result is None


class TestNougatArtifactRemoval:
    """Test suite for NougatArtifactRemovalComponent."""

    @pytest.fixture
    def artifact_component(self):
        return NougatArtifactRemovalComponent(debug=False)

    @pytest.mark.asyncio
    async def test_removes_surrounding_quotes(self, artifact_component):
        """Test removal of surrounding quotes."""
        input_text = '"Content with quotes"'
        result = await artifact_component.process(input_text, "test.txt")
        
        assert result == "Content with quotes"

    @pytest.mark.asyncio
    async def test_converts_escaped_newlines(self, artifact_component):
        """Test conversion of escaped newlines."""
        input_text = "Line 1\\nLine 2\\nLine 3"
        result = await artifact_component.process(input_text, "test.txt")
        
        assert result == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_removes_warning_messages(self, artifact_component):
        """Test removal of warning messages."""
        input_text = "Good content+++==WARNING: Truncated because of repetitions==bad content+++more good content"
        result = await artifact_component.process(input_text, "test.txt")
        
        assert "WARNING" not in result
        assert "Good content" in result
        assert "more good content" in result

    @pytest.mark.asyncio
    async def test_removes_error_messages(self, artifact_component):
        """Test removal of error messages."""
        input_text = "Good content+++==ERROR: No output for this page==bad content+++more good content"
        result = await artifact_component.process(input_text, "test.txt")
        
        assert "ERROR" not in result
        assert "Good content" in result
        assert "more good content" in result

    @pytest.mark.asyncio
    async def test_removes_missing_page_markers(self, artifact_component):
        """Test removal of missing page markers."""
        input_text = "Content before[MISSING_PAGE_POST]content after"
        result = await artifact_component.process(input_text, "test.txt")
        
        assert "[MISSING_PAGE_POST]" not in result
        assert result == "Content beforecontent after"

    @pytest.mark.asyncio
    async def test_empty_content(self, artifact_component):
        """Test handling of empty content."""
        result = await artifact_component.process("", "test.txt")
        assert result is None


class TestLatexCorrection:
    """Test suite for LatexCorrectionComponent."""

    @pytest.fixture
    def latex_component(self):
        with patch('eve.steps.cleaning.latex_correction.PDFLaTeX'):
            return LatexCorrectionComponent(debug=False, api_key="test_key")

    @pytest.mark.asyncio
    async def test_no_latex_formulas(self, latex_component):
        """Test content with no LaTeX formulas."""
        input_text = "Regular text without any LaTeX formulas."
        result = await latex_component.process(input_text, "test.txt")
        
        assert result == input_text

    @pytest.mark.asyncio
    async def test_extracts_inline_formulas(self, latex_component):
        """Test extraction of inline LaTeX formulas."""
        input_text = "Text with inline formula $x^2$ and more text."
        formulas = latex_component._extract_latex_formulas(input_text)
        
        assert len(formulas) == 1
        assert formulas[0] == ("inline", "x^2")

    @pytest.mark.asyncio
    async def test_extracts_display_formulas(self, latex_component):
        """Test extraction of display LaTeX formulas."""
        input_text = "Text with display formula $$E = mc^2$$ and more text."
        formulas = latex_component._extract_latex_formulas(input_text)
        
        assert len(formulas) == 1
        assert formulas[0] == ("display", "E = mc^2")

    @pytest.mark.asyncio
    async def test_extracts_bracket_formulas(self, latex_component):
        """Test extraction of bracket LaTeX formulas."""
        input_text = "Text with bracket formula \\(a + b\\) and more text."
        formulas = latex_component._extract_latex_formulas(input_text)
        
        assert len(formulas) == 1
        assert formulas[0] == ("bracket", "a + b")

    @pytest.mark.asyncio
    async def test_extracts_square_bracket_formulas(self, latex_component):
        """Test extraction of square bracket LaTeX formulas."""
        input_text = "Text with square bracket formula \\[\\int x dx\\] and more text."
        formulas = latex_component._extract_latex_formulas(input_text)
        
        assert len(formulas) == 1
        assert formulas[0] == ("square_bracket", "\\int x dx")

    @pytest.mark.asyncio
    async def test_valid_formula_syntax_check(self, latex_component):
        """Test syntax checking for valid formulas."""
        with patch.object(latex_component, '_check_formula_syntax', return_value=(True, "Valid")):
            input_text = "Text with valid formula $x^2$."
            result = await latex_component.process(input_text, "test.txt")
            
            assert result == input_text  # Should remain unchanged for valid formulas

    @pytest.mark.asyncio
    async def test_invalid_formula_without_api_key(self):
        """Test handling of invalid formulas without API key."""
        with patch('eve.steps.cleaning.latex_correction.PDFLaTeX'):
            component = LatexCorrectionComponent(debug=False, api_key=None)
            
            with patch.object(component, '_check_formula_syntax', return_value=(False, "Invalid syntax")):
                input_text = "Text with invalid formula $\\invalid$."
                result = await component.process(input_text, "test.txt")
                
                # Should return original content when no API key for corrections
                assert result == input_text

    @pytest.mark.asyncio
    async def test_formula_replacement(self, latex_component):
        """Test formula replacement functionality."""
        original = "x^2"
        corrected = "x^{2}"
        content = "Formula $x^2$ in text."
        
        result = latex_component._replace_formula_in_content(content, original, corrected, "inline")
        expected = "Formula $x^{2}$ in text."
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_empty_content(self, latex_component):
        """Test handling of empty content."""
        result = await latex_component.process("", "test.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_ai_correction_with_valid_response(self, latex_component):
        """Test AI correction with valid API response."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "x^{2}"}}]
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await latex_component._correct_formula_with_ai(
                "x^2", "Invalid syntax", "inline", "test.txt", "context"
            )
            
            assert result == "x^{2}"

    @pytest.mark.asyncio
    async def test_ai_correction_with_failed_response(self, latex_component):
        """Test AI correction with failed API response."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status = 500
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await latex_component._correct_formula_with_ai(
                "x^2", "Invalid syntax", "inline", "test.txt", "context"
            )
            
            assert result is None
