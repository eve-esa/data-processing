"""LaTeX correction component using OpenRouter API for fixing LaTeX syntax errors."""

from typing import Optional
import re
import os
import asyncio
import aiohttp
import tempfile

from pdflatex import PDFLaTeX

from eve.steps.cleaning.base_component import DataProcessingComponent


class LatexCorrectionComponent(DataProcessingComponent):
    """
    Component to detect and correct LaTeX syntax errors using OpenRouter API.
    Uses the pdflatex Python package to validate LaTeX syntax and OpenRouter LLMs for corrections.
    """

    def __init__(self, debug: bool = False, api_key: Optional[str] = None, model: str = "anthropic/claude-3-haiku"):
        """Initialize the LaTeX correction component.
        
        Args:
            debug: Enable debug output.
            api_key: OpenRouter API key. If None, will use OPENROUTER_API_KEY environment variable.
            model: OpenRouter model to use for corrections.
        """
        super().__init__(debug=debug)
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        if not self.api_key:
            if debug:
                self.logger.warning("No OPENROUTER_API_KEY found. LaTeX correction will only detect errors.")
        
        self.inline_pattern = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
        self.display_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
        self.bracket_pattern = re.compile(r'\\[(](.*?)\\[)]', re.DOTALL)
        self.square_bracket_pattern = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
        self.latex_env_pattern = re.compile(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', re.DOTALL)

    def _extract_latex_formulas(self, content: str) -> list[tuple[str, str]]:
        """Extract LaTeX formulas from content with their types."""
        formulas = []
        
        # Extract inline math: $...$
        for match in self.inline_pattern.finditer(content):
            formulas.append(("inline", match.group(1).strip()))
        
        # Extract display math: $$...$$
        for match in self.display_pattern.finditer(content):
            formulas.append(("display", match.group(1).strip()))
        
        # Extract bracket math: \(...\)
        for match in self.bracket_pattern.finditer(content):
            formulas.append(("bracket", match.group(1).strip()))
        
        # Extract square bracket math: \[...\]
        for match in self.square_bracket_pattern.finditer(content):
            formulas.append(("square_bracket", match.group(1).strip()))
        
        # Extract LaTeX environments
        for match in self.latex_env_pattern.finditer(content):
            env_name = match.group(1)
            env_content = match.group(2).strip()
            formulas.append(("environment", f"\\begin{{{env_name}}}{env_content}\\end{{{env_name}}}"))
        
        return formulas

    async def _check_formula_syntax(self, formula: str, formula_type: str) -> tuple[bool, str]:
        """Check if a LaTeX formula has valid syntax using pdflatex package."""
        try:
            if formula_type == "inline":
                test_content = f"\\documentclass{{article}}\\usepackage{{amsmath}}\\usepackage{{amssymb}}\\begin{{document}}${formula}$\\end{{document}}"
            elif formula_type == "display":
                test_content = f"\\documentclass{{article}}\\usepackage{{amsmath}}\\usepackage{{amssymb}}\\begin{{document}}$${formula}$$\\end{{document}}"
            elif formula_type == "bracket":
                test_content = f"\\documentclass{{article}}\\usepackage{{amsmath}}\\usepackage{{amssymb}}\\begin{{document}}\\({formula}\\)\\end{{document}}"
            elif formula_type == "square_bracket":
                test_content = f"\\documentclass{{article}}\\usepackage{{amsmath}}\\usepackage{{amssymb}}\\begin{{document}}\\[{formula}\\]\\end{{document}}"
            else:
                test_content = f"\\documentclass{{article}}\\usepackage{{amsmath}}\\usepackage{{amssymb}}\\usepackage{{multirow}}\\usepackage{{bm}}\\begin{{document}}{formula}\\end{{document}}"

            def check_latex():
                """Run pdflatex compilation in a separate thread."""
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tex_file = os.path.join(tmp_dir, "test.tex")
                        with open(tex_file, 'w', encoding='utf-8') as f:
                            f.write(test_content)
                        
                        # Use PDFLaTeX package
                        pdfl = PDFLaTeX.from_texfile(tex_file)
                        pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=False, keep_log_file=False)
                        
                        if completed_process.returncode == 0:
                            return True, "Formula syntax is valid"
                        else:
                            # Parse log for error messages
                            log_content = log.decode('utf-8', errors='replace') if log else "No log available"
                            error_lines = log_content.split('\n')
                            error_msg = "Unknown error"
                            for i, line in enumerate(error_lines):
                                if "! " in line:
                                    error_msg = line.strip()
                                    if i + 1 < len(error_lines) and error_lines[i + 1].strip():
                                        error_msg += " " + error_lines[i + 1].strip()
                                    break
                            return False, error_msg
                except Exception as e:
                    return False, f"PDFLaTeX compilation failed: {str(e)}"
            
            # Run the blocking operation in a thread pool
            return await asyncio.get_event_loop().run_in_executor(None, check_latex)

        except Exception as e:
            return False, f"Syntax check failed: {str(e)}"

    async def _correct_formula_with_ai(self, formula: str, error_message: str, formula_type: str, filename: str, context: str) -> Optional[str]:
        """Correct LaTeX formula using OpenRouter API."""
        if not self.api_key:
            return None
        
        try:
            context_snippet = context[:1000] + "..." if len(context) > 1000 else context
            
            prompt = f"""Please correct the following LaTeX formula that has a syntax error:

Formula type: {formula_type}
Error message: {error_message}
Formula: {formula}

Context (first 1000 chars): {context_snippet}

Please provide ONLY the corrected LaTeX formula without any explanations or surrounding text. Keep the mathematical meaning intact while fixing the syntax errors."""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        corrected_formula = result["choices"][0]["message"]["content"].strip()
                        
                        corrected_formula = re.sub(r'^```latex\n?', '', corrected_formula)
                        corrected_formula = re.sub(r'\n?```$', '', corrected_formula)
                        corrected_formula = corrected_formula.strip()
                        
                        return corrected_formula
                    else:
                        self.logger.error(f"{filename} - OpenRouter API request failed with status {response.status}")
                        return None

        except Exception as e:
            self.logger.error(f"{filename} - AI correction failed: {str(e)}")
            return None

    def _replace_formula_in_content(self, content: str, original: str, corrected: str, formula_type: str) -> str:
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

    async def process(self, content: str, filename: str) -> Optional[str]:
        """Process content to detect and correct LaTeX syntax errors.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Content with LaTeX errors corrected, or None if processing fails.
        """
        if self.debug:
            self.logger.info(f"Before LaTeX Correction ({filename}): {content[:500]}{'...' if len(content) > 500 else ''}")

        if not content:
            self.logger.error(f"{filename} - Empty content in LaTeX Correction")
            return None

        try:
            formulas = self._extract_latex_formulas(content)
            
            if not formulas:
                self.logger.info(f"{filename} - No LaTeX formulas found")
                return content

            errors_found = 0
            corrections_made = 0
            modified_content = content

            for formula_type, formula in formulas:
                is_valid, error_message = await self._check_formula_syntax(formula, formula_type)
                
                if not is_valid:
                    errors_found += 1
                    self.logger.warning(f"{filename} - Invalid LaTeX formula: {formula[:50]}... Error: {error_message}")
                    
                    if self.api_key:
                        corrected_formula = await self._correct_formula_with_ai(formula, error_message, formula_type, filename, content)
                        
                        if corrected_formula and corrected_formula != formula:
                            is_corrected_valid, _ = await self._check_formula_syntax(corrected_formula, formula_type)
                            
                            if is_corrected_valid:
                                modified_content = self._replace_formula_in_content(
                                    modified_content, formula, corrected_formula, formula_type
                                )
                                corrections_made += 1
                                self.logger.info(f"{filename} - Corrected LaTeX formula: {formula[:30]}... -> {corrected_formula[:30]}...")
                            else:
                                self.logger.warning(f"{filename} - AI correction still invalid: {corrected_formula[:50]}...")

            if errors_found > 0:
                self.logger.info(f"{filename} - LaTeX processing complete: {errors_found} errors found, {corrections_made} corrected")
            else:
                self.logger.info(f"{filename} - All LaTeX formulas are valid")

            if self.debug:
                self.logger.info(f"After LaTeX Correction ({filename}): {errors_found} errors, {corrections_made} fixed")

            return modified_content

        except Exception as e:
            self.logger.error(f"{filename} - LaTeX Correction failed: {str(e)}")
            return content
