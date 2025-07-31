"""Data cleaning processors based on the existing 5-check cleaning process."""

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus


class OCRCorrector(ProcessorBase):
    """Processor to fix OCR-induced errors by separating merged numbers and words."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize OCR corrector."""
        super().__init__(name="OCRCorrector", **kwargs)
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content to fix OCR errors.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with processed content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            # Separate merged numbers and words only if the word starts with a number
            # Don't separate good combinations like 40M, 100k, etc.
            cleaned = re.sub(r'(\d+)([A-Za-z]{2,})', r'\1 \2', content)
            
            changes_made = cleaned != content
            if changes_made:
                self.logger.info(f"OCR corrections applied to content")
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=cleaned,
                metadata={
                    "changes_made": changes_made,
                    "original_length": len(content),
                    "cleaned_length": len(cleaned),
                },
            )
            
        except Exception as e:
            self.logger.error(f"OCR correction failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )


class OCRDeduplicator(ProcessorBase):
    """Processor to remove OCR-induced duplicate text segments using threshold-based overlap."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.99,
        min_words: int = 2,
        **kwargs,
    ) -> None:
        """Initialize OCR deduplicator.
        
        Args:
            similarity_threshold: Jaccard similarity threshold for duplicates.
            min_words: Minimum words required for processing a sentence.
            **kwargs: Additional configuration.
        """
        super().__init__(name="OCRDeduplicator", **kwargs)
        self.similarity_threshold = similarity_threshold
        self.min_words = min_words
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content to remove OCR duplicates.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with deduplicated content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            cleaned_content, removed_segments = self._remove_near_adjacent_duplicates(content)
            
            percent_removed = 0.0
            if content:
                percent_removed = (len(content) - len(cleaned_content)) / len(content) * 100
            
            self.logger.info(
                f"Removed {len(removed_segments)} duplicate segments, "
                f"{percent_removed:.2f}% of text removed"
            )
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=cleaned_content,
                metadata={
                    "removed_segments": len(removed_segments),
                    "percent_removed": percent_removed,
                    "original_length": len(content),
                    "cleaned_length": len(cleaned_content),
                },
            )
            
        except Exception as e:
            self.logger.error(f"OCR deduplication failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )
    
    def _is_noise_line(self, line: str) -> bool:
        """Check if line is just noise (whitespace or punctuation)."""
        return (
            line.strip() == '' or
            re.fullmatch(r'[\W_]+', line.strip())
        )
    
    def _is_similar(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are similar based on word overlap."""
        words1 = sent1.lower().split()
        words2 = sent2.lower().split()
        
        if len(words1) < self.min_words:
            return False
        
        set1, set2 = set(words1), set(words2)
        overlap = len(set1 & set2)
        
        return (overlap / len(set1) >= self.similarity_threshold or 
                overlap / len(set2) >= self.similarity_threshold)
    
    def _remove_near_adjacent_duplicates(self, content: str) -> Tuple[str, List[str]]:
        """Remove near-adjacent duplicate sentences."""
        sentences = content.split('\n')
        cleaned = []
        removed = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            if len(current.split()) < self.min_words:
                cleaned.append(current)
                i += 1
                continue
            
            # Look ahead skipping noise lines
            j = i + 1
            while j < len(sentences) and self._is_noise_line(sentences[j]):
                j += 1
            
            if j < len(sentences) and self._is_similar(current, sentences[j]):
                self.logger.debug(f"Removing near-duplicate: {repr(sentences[j])}")
                removed.append(sentences[j])
                # Skip all noise and the similar sentence
                i = j
            else:
                cleaned.append(current)
                i += 1
        
        return '\n'.join(cleaned), removed


class NougatCorrector(ProcessorBase):
    """Processor to apply Nougat-specific corrections and postprocessing."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize Nougat corrector."""
        super().__init__(name="NougatCorrector", **kwargs)
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content with Nougat corrections.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with corrected content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            # Apply Nougat's postprocessing if available
            cleaned = self._apply_nougat_postprocessing(content)
            
            # Apply custom LaTeX table logic
            cleaned = self._clean_latex_table(cleaned)
            
            changes_made = cleaned != content
            if changes_made:
                self.logger.info("Nougat corrections applied to content")
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=cleaned,
                metadata={
                    "changes_made": changes_made,
                    "original_length": len(content),
                    "cleaned_length": len(cleaned),
                },
            )
            
        except Exception as e:
            self.logger.error(f"Nougat correction failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )
    
    def _apply_nougat_postprocessing(self, content: str) -> str:
        """Apply Nougat's built-in postprocessing if available."""
        try:
            from nougat.postprocessing import markdown_compatible
            return markdown_compatible(content)
        except ImportError:
            self.logger.warning("Nougat postprocessing not available, skipping")
            return content
        except Exception as e:
            self.logger.warning(f"Nougat postprocessing failed: {e}")
            return content
    
    def _clean_latex_table(self, raw_table: str) -> str:
        """Clean LaTeX table formatting by fixing escaped backslashes."""
        # Replace multiple backslashes with single backslashes
        # This fixes issues like \\\\hline -> \\hline
        table = re.sub(r'\\{2,}', lambda m: '\\' * (len(m.group()) // 2), raw_table)
        return table


class RuleBasedCorrector(ProcessorBase):
    """Processor to apply rule-based corrections for common formatting issues."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize rule-based corrector."""
        super().__init__(name="RuleBasedCorrector", **kwargs)
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content with rule-based corrections.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with corrected content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            cleaned_lines = []
            removed_lines = 0
            
            for line in content.split('\n'):
                stripped = line.strip()
                
                # Skip lines with no alphanumeric characters (only symbols/punctuation) and single char
                if not re.search(r'\w', stripped) and len(stripped) == 1:
                    removed_lines += 1
                    continue
                
                cleaned_lines.append(line)
            
            cleaned = '\n'.join(cleaned_lines)
            
            # Replace 3+ consecutive newlines with exactly 2
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            
            # Remove leading/trailing whitespace
            cleaned = cleaned.strip()
            
            changes_made = cleaned != content
            if changes_made:
                self.logger.info(f"Rule-based corrections applied, removed {removed_lines} symbol-only lines")
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=cleaned,
                metadata={
                    "changes_made": changes_made,
                    "removed_lines": removed_lines,
                    "original_length": len(content),
                    "cleaned_length": len(cleaned),
                },
            )
            
        except Exception as e:
            self.logger.error(f"Rule-based correction failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )


class ArtifactRemover(ProcessorBase):
    """Processor to remove Nougat-specific artifacts and warnings."""
    
    def __init__(self, **kwargs) -> None:
        """Initialize artifact remover."""
        super().__init__(name="ArtifactRemover", **kwargs)
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content to remove artifacts.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with artifacts removed.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            # Remove surrounding quotes
            cleaned = content.strip('"')
            
            # Replace escaped newlines with actual newlines
            cleaned = cleaned.replace('\\n', '\n')
            
            # Remove Nougat warning messages
            warnings_removed = 0
            
            # Remove WARNING messages
            warning_pattern = r'\+\+\+\s*==WARNING: Truncated because of repetitions==.*?\+\+\+'
            if re.search(warning_pattern, cleaned, flags=re.DOTALL):
                warnings_removed += len(re.findall(warning_pattern, cleaned, flags=re.DOTALL))
                cleaned = re.sub(warning_pattern, '', cleaned, flags=re.DOTALL)
            
            # Remove ERROR messages
            error_pattern = r'\+\+\+\s*==ERROR: No output for this page==.*?\+\+\+'
            if re.search(error_pattern, cleaned, flags=re.DOTALL):
                warnings_removed += len(re.findall(error_pattern, cleaned, flags=re.DOTALL))
                cleaned = re.sub(error_pattern, '', cleaned, flags=re.DOTALL)
            
            # Remove missing page indicators
            cleaned = cleaned.replace('[MISSING_PAGE_POST]', '')
            
            changes_made = cleaned != content
            if changes_made:
                self.logger.info(f"Removed {warnings_removed} Nougat artifacts")
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=cleaned,
                metadata={
                    "changes_made": changes_made,
                    "warnings_removed": warnings_removed,
                    "original_length": len(content),
                    "cleaned_length": len(cleaned),
                },
            )
            
        except Exception as e:
            self.logger.error(f"Artifact removal failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )


class LatexCorrector(ProcessorBase):
    """Processor to detect and correct LaTeX syntax errors using OpenAI GPT-4o-mini."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize LaTeX corrector.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
            **kwargs: Additional processor parameters.
        """
        super().__init__(name="LatexCorrector", **kwargs)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            self.logger.warning("No OPENAI_API_KEY found. LaTeX correction will only detect errors.")
        
        # Regex patterns for different LaTeX formula types
        self.inline_pattern = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
        self.display_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
        self.bracket_pattern = re.compile(r'\\[(](.*?)\\[)]', re.DOTALL)
        self.square_bracket_pattern = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
        self.latex_env_pattern = re.compile(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', re.DOTALL)
        self.table_env_pattern = re.compile(r'\\begin\{(table)\}(.*?)\\end\{table\}', re.DOTALL)
        
        # API configuration
        self.model = "gpt-4o-mini"
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        } if self.api_key else None
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content with LaTeX corrections.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with corrected content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            formulas = self._extract_formulas(content)
            
            if not formulas:
                self.logger.info("No LaTeX formulas found")
                return ProcessorResult(
                    status=ProcessorStatus.SUCCESS,
                    input_path=input_path,
                    content=content,
                    metadata={"formulas_found": 0, "corrections_made": 0, "errors_found": 0},
                )
            
            self.logger.info(f"Found {len(formulas)} LaTeX formulas")
            
            corrections_made = 0
            errors_found = 0
            modified_content = content
            
            for formula_type, formula in formulas:
                is_valid, error_message = self._check_formula_syntax(formula, formula_type)
                
                if not is_valid:
                    errors_found += 1
                    self.logger.warning(f"Invalid LaTeX formula: {formula[:50]}... Error: {error_message}")
                    
                    if self.api_key:
                        corrected_formula = self._correct_formula_with_ai(
                            formula, error_message, formula_type, content
                        )
                        
                        if corrected_formula and corrected_formula != formula:
                            is_corrected_valid, _ = self._check_formula_syntax(corrected_formula, formula_type)
                            
                            if is_corrected_valid:
                                modified_content = self._replace_formula_in_content(
                                    modified_content, formula, corrected_formula, formula_type
                                )
                                corrections_made += 1
                                self.logger.info(f"Corrected LaTeX formula: {formula[:30]}... -> {corrected_formula[:30]}...")
                            else:
                                self.logger.warning(f"AI correction still invalid: {corrected_formula[:50]}...")
            
            if errors_found > 0:
                self.logger.info(f"LaTeX processing complete: {errors_found} errors found, {corrections_made} corrected")
            else:
                self.logger.info("All LaTeX formulas are valid")
            
            changes_made = modified_content != content
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=modified_content,
                metadata={
                    "changes_made": changes_made,
                    "formulas_found": len(formulas),
                    "errors_found": errors_found,
                    "corrections_made": corrections_made,
                    "original_length": len(content),
                    "cleaned_length": len(modified_content),
                },
            )
            
        except Exception as e:
            self.logger.error(f"LaTeX correction failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )
    
    def _extract_formulas(self, text: str) -> List[Tuple[str, str]]:
        """Extract all LaTeX formulas from text."""
        formulas = []
        
        # Extract inline formulas ($...$)
        for match in self.inline_pattern.finditer(text):
            formulas.append(('inline', match.group(1)))
        
        # Extract display formulas ($$...$$)
        for match in self.display_pattern.finditer(text):
            formulas.append(('display', match.group(1)))
        
        # Extract \( ... \) formulas
        for match in self.bracket_pattern.finditer(text):
            formulas.append(('inline-explicit', match.group(1)))
        
        # Extract \[ ... \] formulas
        for match in self.square_bracket_pattern.finditer(text):
            formulas.append(('display-explicit', match.group(1)))
        
        # Extract LaTeX environments
        for match in self.latex_env_pattern.finditer(text):
            env_type = match.group(1)
            if env_type in ['equation', 'align', 'gather', 'multline', 'eqnarray', 'matrix', 
                           'equation*', 'align*', 'bmatrix', 'pmatrix']:
                formulas.append((f'env:{env_type}', match.group(2)))
        
        # Extract table environments
        for match in self.table_env_pattern.finditer(text):
            formulas.append(('table-env:', match.group(0)))
        
        return formulas
    
    def _check_formula_syntax(self, formula: str, formula_type: str) -> Tuple[bool, str]:
        """Check if a LaTeX formula has valid syntax using subprocess compilation."""
        if not formula.strip():
            return True, "Empty formula"
        
        try:
            test_content = self._create_test_document(formula, formula_type)
            
            # Create temporary directory for testing
            import tempfile
            import subprocess
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                tex_file = os.path.join(tmp_dir, "test.tex")
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)
                
                # Run pdflatex to check for errors
                process = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_file],
                    cwd=tmp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if process.returncode == 0:
                    return True, "Formula syntax is valid"
                else:
                    # Extract error message
                    error_lines = process.stdout.split('\n')
                    error_msg = "LaTeX compilation failed"
                    for i, line in enumerate(error_lines):
                        if '! ' in line:
                            error_msg = line.strip()
                            if i + 1 < len(error_lines) and error_lines[i + 1].strip():
                                error_msg += " " + error_lines[i + 1].strip()
                            break
                    
                    return False, error_msg
        
        except FileNotFoundError:
            self.logger.warning("pdflatex not found - skipping LaTeX syntax validation")
            return True, "pdflatex not available"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"
    
    def _create_test_document(self, formula: str, formula_type: str) -> str:
        """Create a minimal LaTeX document to test formula syntax."""
        packages = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{multirow}\usepackage{bm}\usepackage{upgreek}"
        
        if formula_type == 'inline':
            content = f"${formula}$"
        elif formula_type == 'inline-explicit':
            content = f"\\({formula}\\)"
        elif formula_type == 'display':
            content = f"$${formula}$$"
        elif formula_type == 'display-explicit':
            content = f"\\[{formula}\\]"
        elif formula_type.startswith('env:'):
            env = formula_type.split(':')[1]
            content = f"\\begin{{{env}}}{formula}\\end{{{env}}}"
        elif formula_type.startswith('table-env:'):
            content = formula
        else:
            content = formula
        
        return f"\\documentclass{{article}}{packages}\\begin{{document}}{content}\\end{{document}}"
    
    def _correct_formula_with_ai(
        self, 
        formula: str, 
        error_message: str, 
        formula_type: str, 
        content: str = ""
    ) -> Optional[str]:
        """Use OpenAI GPT-4o-mini to correct the LaTeX formula."""
        if not self.headers:
            return None
        
        surrounding_context = self._get_surrounding_context(formula, content) if content else ""
        
        prompt = f"""You are correcting a LaTeX formula extracted from a scientific document.

SURROUNDING CONTEXT:
{surrounding_context}

FORMULA DETAILS:
- Type: {formula_type}
- Error: {error_message}
- Original: {formula}

Please provide ONLY the corrected LaTeX formula without any explanation."""
        
        try:
            import requests
            import time
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert LaTeX mathematician specializing in fixing syntax errors in scientific formulas. "
                                  "Your corrections must preserve mathematical meaning while ensuring LaTeX compilation success. "
                                  "You have extensive knowledge of common LaTeX errors from OCR extraction, web scraping, encoding issues, and missing packages. "
                                  "Return ONLY the corrected formula - no explanations, no markdown formatting, no surrounding text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 4000
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                corrected = result["choices"][0]["message"]["content"].strip()
                return corrected if corrected != formula else None
            elif response.status_code == 429:  # Rate limit
                self.logger.warning("API rate limit hit, skipping correction")
                time.sleep(1)
                return None
            else:
                self.logger.warning(f"API request failed: {response.status_code}")
                return None
        
        except Exception as e:
            self.logger.warning(f"AI correction failed: {str(e)}")
            return None
    
    def _replace_formula_in_content(
        self, 
        content: str, 
        original: str, 
        corrected: str, 
        formula_type: str
    ) -> str:
        """Replace the original formula with the corrected version in content."""
        try:
            if formula_type == 'inline':
                old_pattern = f"${re.escape(original)}$"
                new_replacement = f"${corrected}$"
            elif formula_type == 'display':
                old_pattern = f"$$" + re.escape(original) + "$$"
                new_replacement = f"$${corrected}$$"
            elif formula_type == 'inline-explicit':
                old_pattern = f"\\(" + re.escape(original) + "\\)"
                new_replacement = f"\\({corrected}\\)"
            elif formula_type == 'display-explicit':
                old_pattern = f"\\[" + re.escape(original) + "\\]"
                new_replacement = f"\\[{corrected}\\]"
            elif formula_type.startswith('env:'):
                env = formula_type.split(':')[1]
                old_pattern = f"\\begin{{{env}}}" + re.escape(original) + f"\\end{{{env}}}"
                new_replacement = f"\\begin{{{env}}}{corrected}\\end{{{env}}}"
            elif formula_type.startswith('table-env:'):
                old_pattern = re.escape(original)
                new_replacement = corrected
            else:
                return content
            
            updated_content = re.sub(old_pattern, new_replacement, content, count=1)
            return updated_content
        
        except Exception:
            return content.replace(original, corrected, 1)
    
    def _get_surrounding_context(self, formula: str, content: str, context_chars: int = 300) -> str:
        """Extract surrounding text context around the formula."""
        if not content or not formula:
            return "No surrounding context available."
        
        formula_patterns = [
            f"${re.escape(formula)}$",
            f"$${re.escape(formula)}$$",
            f"\\({re.escape(formula)}\\)",
            f"\\[{re.escape(formula)}\\]",
            re.escape(formula)
        ]
        
        formula_position = -1
        matched_pattern = ""
        
        for pattern in formula_patterns:
            try:
                match = re.search(pattern, content)
                if match:
                    formula_position = match.start()
                    matched_pattern = match.group(0)
                    break
            except re.error:
                continue
        
        if formula_position == -1:
            formula_position = content.find(formula)
            matched_pattern = formula
        
        if formula_position == -1:
            return "Formula not found in document context."
        
        start_pos = max(0, formula_position - context_chars)
        end_pos = min(len(content), formula_position + len(matched_pattern) + context_chars)
        
        before_context = content[start_pos:formula_position].strip()
        after_context = content[formula_position + len(matched_pattern):end_pos].strip()
        
        before_context = re.sub(r'\s+', ' ', before_context)
        after_context = re.sub(r'\s+', ' ', after_context)
        
        if len(before_context) > context_chars:
            before_context = "..." + before_context[-(context_chars-3):]
        if len(after_context) > context_chars:
            after_context = after_context[:context_chars-3] + "..."
        
        context_parts = []
        if before_context:
            context_parts.append(f"Before: {before_context}")
        if after_context:
            context_parts.append(f"After: {after_context}")
        
        return "\n".join(context_parts) if context_parts else "No meaningful context found."