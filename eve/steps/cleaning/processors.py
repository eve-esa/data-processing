"""
Consolidated text processing components for the cleaning pipeline.

This module combines all the individual cleaning components into a unified structure
for better organization and maintainability.
"""

import os
import re
import asyncio
import tempfile
import subprocess
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

from pylatex import Document as LaTeXDocument, NoEscape
from pylatex.package import Package

from eve.logging import get_logger
from eve.model.document import Document
from eve.common.regex_patterns import (
    fix_ocr_digit_letter_spacing,
    normalize_excessive_newlines,
    remove_single_symbol_lines,
    remove_nougat_artifacts,
    clean_doubled_backslashes,
    get_latex_formula_patterns,
)
from eve.common.prompts import get_latex_correction_prompt
from eve.common.http_utils import make_openrouter_request
from eve.steps.cleaning.nougat_helpers import postprocess_single


class TextProcessor(ABC):
    """Abstract base class for text processing components."""

    def __init__(self, debug: bool = False):
        """Initialize the text processor.

        Args:
            debug: Enable debug output.
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def process(self, document: Document) -> Document:
        """Process a document and return the cleaned result.

        Args:
            document: The document to process.

        Returns:
            Processed document.
        """
        pass


class OCRProcessor(TextProcessor):
    """Processor for fixing OCR-induced text issues."""

    async def process(self, document: Document) -> Document:
        """Fix OCR issues in the document content."""
        if self.debug:
            self.logger.info(
                f"Before OCR processing ({document.filename}): {document.content[:200]}..."
            )

        if document.is_empty():
            self.logger.warning(
                f"{document.filename} - Empty content in OCR processing"
            )
            return document

        try:
            # Fix digit-letter spacing issues
            cleaned_content = fix_ocr_digit_letter_spacing(document.content)

            document.update_content(cleaned_content)
            document.add_metadata("ocr_processed", True)

            self.logger.info(f"{document.filename} - OCR processing completed")

            if self.debug:
                self.logger.info(
                    f"After OCR processing ({document.filename}): {document.content[:200]}..."
                )

            return document

        except Exception as e:
            self.logger.error(f"{document.filename} - OCR processing failed: {str(e)}")
            return document


class DuplicateRemovalProcessor(TextProcessor):
    """Processor for removing OCR-induced duplicate text segments."""

    def __init__(
        self, threshold: float = 0.99, min_words: int = 2, debug: bool = False
    ):
        """
        Initialize the duplicate removal processor.

        Args:
            threshold: Similarity threshold for duplicates.
            min_words: Minimum words required for a unit to be processed.
            debug: Enable debug output.
        """
        super().__init__(debug=debug)
        self.threshold = threshold
        self.min_words = min_words

    def _is_similar(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are similar based on word overlap."""
        words1 = sent1.lower().split()
        words2 = sent2.lower().split()

        if len(words1) < self.min_words:
            return False

        set1, set2 = set(words1), set(words2)
        overlap = len(set1 & set2)
        return (
            overlap / len(set1) >= self.threshold
            or overlap / len(set2) >= self.threshold
        )

    def _remove_near_adjacent_duplicates(
        self, content: str, filename: str
    ) -> Tuple[str, List[str]]:
        """Remove near-adjacent duplicate sentences."""
        sentences = content.split("\n")
        cleaned = []
        removed = []
        i = 0

        while i < len(sentences):
            current = sentences[i]
            if len(current.split()) < self.min_words:
                cleaned.append(current)
                i += 1
                continue

            j = i + 1
            while j < len(sentences) and not sentences[j].strip():
                j += 1

            if j < len(sentences) and self._is_similar(current, sentences[j]):
                self.logger.info(
                    f"{filename} - Removing near-duplicate: {repr(sentences[j])}"
                )
                removed.append(sentences[j])
                i = j
            else:
                cleaned.append(current)
                i += 1

        return "\n".join(cleaned), removed

    async def process(self, document: Document) -> Document:
        """Remove duplicate content from the document."""
        if self.debug:
            self.logger.info(
                f"Before duplicate removal ({document.filename}): {document.content[:200]}..."
            )

        if document.is_empty():
            self.logger.warning(
                f"{document.filename} - Empty content in duplicate removal"
            )
            return document

        try:
            cleaned_content, removed = self._remove_near_adjacent_duplicates(
                document.content, document.filename
            )

            percent_removed = 0.0
            if document.content:
                percent_removed = (
                    (len(document.content) - len(cleaned_content))
                    / len(document.content)
                    * 100
                )

            document.update_content(cleaned_content)
            document.add_metadata("duplicates_removed", len(removed))
            document.add_metadata("duplicate_removal_percent", percent_removed)

            self.logger.info(
                f"{document.filename} - Duplicate removal: {len(removed)} segments, {percent_removed:.2f}% text removed"
            )

            if self.debug:
                self.logger.info(
                    f"After duplicate removal ({document.filename}): {document.content[:200]}..."
                )

            return document

        except Exception as e:
            self.logger.error(
                f"{document.filename} - Duplicate removal failed: {str(e)}"
            )
            return document


class NougatProcessor(TextProcessor):
    """Processor for fixing Nougat-related issues and artifacts."""

    async def process(self, document: Document) -> Document:
        """Process Nougat-specific issues in the document."""
        if self.debug:
            self.logger.info(
                f"Before Nougat processing ({document.filename}): {document.content[:200]}..."
            )

        if document.is_empty():
            self.logger.warning(
                f"{document.filename} - Empty content in Nougat processing"
            )
            return document

        try:
            # Apply Nougat postprocessing
            cleaned = postprocess_single(document.content, markdown_fix=True)

            # Clean LaTeX table formatting
            cleaned = clean_doubled_backslashes(cleaned)

            # Remove Nougat artifacts
            cleaned = remove_nougat_artifacts(cleaned)

            # Convert escaped newlines
            cleaned = cleaned.replace("\\n", "\n")

            # Remove surrounding quotes
            cleaned = cleaned.strip('"')

            document.update_content(cleaned)
            document.add_metadata("nougat_processed", True)

            self.logger.info(f"{document.filename} - Nougat processing completed")

            if self.debug:
                self.logger.info(
                    f"After Nougat processing ({document.filename}): {document.content[:200]}..."
                )

            return document

        except Exception as e:
            self.logger.error(
                f"{document.filename} - Nougat processing failed: {str(e)}"
            )
            return document


class RuleBasedProcessor(TextProcessor):
    """Processor for applying rule-based text corrections."""

    async def process(self, document: Document) -> Document:
        """Apply rule-based corrections to the document."""
        if self.debug:
            self.logger.info(
                f"Before rule-based processing ({document.filename}): {document.content[:200]}..."
            )

        if document.is_empty():
            self.logger.warning(
                f"{document.filename} - Empty content in rule-based processing"
            )
            return document

        try:
            # Remove single symbol lines
            cleaned = remove_single_symbol_lines(document.content)

            # Normalize excessive newlines
            cleaned = normalize_excessive_newlines(cleaned)

            # Trim whitespace
            cleaned = cleaned.strip()

            document.update_content(cleaned)
            document.add_metadata("rule_based_processed", True)

            self.logger.info(f"{document.filename} - Rule-based processing completed")

            if self.debug:
                self.logger.info(
                    f"After rule-based processing ({document.filename}): {document.content[:200]}..."
                )

            return document

        except Exception as e:
            self.logger.error(
                f"{document.filename} - Rule-based processing failed: {str(e)}"
            )
            return document


class LaTeXProcessor(TextProcessor):
    """Processor for detecting and correcting LaTeX syntax errors."""

    def __init__(
        self,
        debug: bool = False,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3-haiku",
    ):
        """Initialize the LaTeX processor.

        Args:
            debug: Enable debug output.
            api_key: OpenRouter API key. If None, will use OPENROUTER_API_KEY environment variable.
            model: OpenRouter model to use for corrections.
        """
        super().__init__(debug=debug)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.formula_patterns = get_latex_formula_patterns()

        if not self.api_key and debug:
            self.logger.warning(
                "No OPENROUTER_API_KEY found. LaTeX correction will only detect errors."
            )

    def _extract_latex_formulas(self, content: str) -> List[Tuple[str, str]]:
        """Extract LaTeX formulas from content with their types."""
        formulas = []

        for formula_type, pattern in self.formula_patterns.items():
            for match in pattern.finditer(content):
                if formula_type == "environment":
                    env_name = match.group(1)
                    env_content = match.group(2).strip()
                    formulas.append(
                        (
                            formula_type,
                            f"\\begin{{{env_name}}}{env_content}\\end{{{env_name}}}",
                        )
                    )
                else:
                    formulas.append((formula_type, match.group(1).strip()))

        return formulas

    async def _check_formula_syntax(
        self, formula: str, formula_type: str
    ) -> Tuple[bool, str]:
        """Check if a LaTeX formula has valid syntax using pylatex and subprocess."""
        try:
            def check_latex():
                """Run pdflatex compilation in a separate thread."""
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Create a pylatex Document
                        doc = LaTeXDocument(documentclass="article")

                        # Add required packages
                        doc.packages.append(Package("amsmath"))
                        doc.packages.append(Package("amssymb"))

                        # Add formula-specific content wrapped in NoEscape
                        if formula_type == "inline":
                            doc.append(NoEscape(f"${formula}$"))
                        elif formula_type == "display":
                            doc.append(NoEscape(f"$${formula}$$"))
                        elif formula_type == "bracket":
                            doc.append(NoEscape(f"\\({formula}\\)"))
                        elif formula_type == "square_bracket":
                            doc.append(NoEscape(f"\\[{formula}\\]"))
                        else:
                            # Environment type - add extra packages
                            doc.packages.append(Package("multirow"))
                            doc.packages.append(Package("bm"))
                            doc.append(NoEscape(formula))

                        # Generate the .tex file
                        tex_file = os.path.join(tmp_dir, "test")
                        doc.generate_tex(tex_file)

                        # Run pdflatex using subprocess
                        result = subprocess.run(
                            ["pdflatex", "-interaction=nonstopmode", f"{tex_file}.tex"],
                            cwd=tmp_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=30,
                        )

                        if result.returncode == 0:
                            return True, "Formula syntax is valid"
                        else:
                            # Parse output for error messages
                            output = result.stdout.decode("utf-8", errors="replace")
                            error_lines = output.split("\n")
                            error_msg = "Unknown error"
                            for i, line in enumerate(error_lines):
                                if "! " in line:
                                    error_msg = line.strip()
                                    if (
                                        i + 1 < len(error_lines)
                                        and error_lines[i + 1].strip()
                                    ):
                                        error_msg += " " + error_lines[i + 1].strip()
                                    break
                            return False, error_msg
                except subprocess.TimeoutExpired:
                    return False, "PDFLaTeX compilation timed out"
                except FileNotFoundError:
                    return False, "pdflatex command not found. Please ensure LaTeX is installed."
                except Exception as e:
                    return False, f"PDFLaTeX compilation failed: {str(e)}"

            # Run the blocking operation in a thread pool
            return await asyncio.get_event_loop().run_in_executor(None, check_latex)

        except Exception as e:
            return False, f"Syntax check failed: {str(e)}"

    def _replace_formula_in_content(
        self, content: str, original: str, corrected: str, formula_type: str
    ) -> str:
        """Replace original formula with corrected version in content."""
        try:
            if formula_type == "inline":
                pattern = re.escape(f"${original}$")
                replacement = f"${corrected}$"
            elif formula_type == "display":
                pattern = re.escape(f"$${original}$$")
                replacement = f"$${corrected}$$"
            elif formula_type == "bracket":
                pattern = re.escape(f"\\({original}\\)")
                replacement = f"\\({corrected}\\)"
            elif formula_type == "square_bracket":
                pattern = re.escape(f"\\[{original}\\]")
                replacement = f"\\[{corrected}\\]"
            else:
                pattern = re.escape(original)
                replacement = corrected

            return re.sub(pattern, replacement, content, count=1)
        except Exception:
            return content.replace(original, corrected, 1)

    async def process(self, document: Document) -> Document:
        """Process document to detect and correct LaTeX syntax errors."""
        if self.debug:
            self.logger.info(
                f"Before LaTeX processing ({document.filename}): {document.content[:200]}..."
            )

        if document.is_empty():
            self.logger.warning(
                f"{document.filename} - Empty content in LaTeX processing"
            )
            return document

        try:
            formulas = self._extract_latex_formulas(document.content)

            if not formulas:
                self.logger.info(f"{document.filename} - No LaTeX formulas found")
                document.add_metadata("latex_processed", True)
                return document

            errors_found = 0
            corrections_made = 0
            modified_content = document.content

            for formula_type, formula in formulas:
                is_valid, error_message = await self._check_formula_syntax(
                    formula, formula_type
                )

                if not is_valid:
                    errors_found += 1
                    self.logger.warning(
                        f"{document.filename} - Invalid LaTeX formula: {formula[:10]}... Error: {error_message}"
                    )

                    if self.api_key:
                        prompt = get_latex_correction_prompt(
                            formula_type, error_message, formula, document.content
                        )
                        corrected_formula = await make_openrouter_request(
                            self.api_key, self.model, prompt
                        )

                        if corrected_formula and corrected_formula != formula:
                            is_corrected_valid, _ = await self._check_formula_syntax(
                                corrected_formula, formula_type
                            )

                            if is_corrected_valid:
                                modified_content = self._replace_formula_in_content(
                                    modified_content,
                                    formula,
                                    corrected_formula,
                                    formula_type,
                                )
                                corrections_made += 1
                                self.logger.info(
                                    f"{document.filename} - Corrected LaTeX formula: {formula[:30]}... -> {corrected_formula[:30]}..."
                                )
                            else:
                                self.logger.warning(
                                    f"{document.filename} - LLM correction still invalid: {corrected_formula[:50]}..."
                                )

            document.update_content(modified_content)
            document.add_metadata("latex_errors_found", errors_found)
            document.add_metadata("latex_corrections_made", corrections_made)
            document.add_metadata("latex_processed", True)

            if errors_found > 0:
                self.logger.info(
                    f"{document.filename} - LaTeX processing complete: {errors_found} errors found, {corrections_made} corrected"
                )
            else:
                self.logger.info(f"{document.filename} - All LaTeX formulas are valid")

            if self.debug:
                self.logger.info(
                    f"After LaTeX processing ({document.filename}): {errors_found} errors, {corrections_made} fixed"
                )

            return document

        except Exception as e:
            self.logger.error(
                f"{document.filename} - LaTeX processing failed: {str(e)}"
            )
            return document
