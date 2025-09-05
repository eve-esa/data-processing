"""Nougat correction component for fixing Nougat-related issues."""

from typing import Optional
import re

from eve.steps.cleaning.base_component import DataProcessingComponent
from eve.steps.cleaning.nougat_helpers import postprocess_single


class NougatCorrection(DataProcessingComponent):
    """Component to correct Nougat-related text issues and LaTeX tables."""

    def __init__(self, debug: bool = False):
        """Initialize the Nougat correction component.
        
        Args:
            debug: Enable debug output.
        """
        super().__init__(debug=debug)
    
    @staticmethod
    def _clean_latex_table(raw_table: str) -> str:
        """Clean LaTeX table formatting by fixing escaped backslashes."""
        table = re.sub(r'\\{2,}', lambda m: '\\' * (len(m.group()) // 2), raw_table)
        return table
    
    async def process(self, content: str, filename: str) -> Optional[str]:
        """Process content to fix Nougat-related issues.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Cleaned content with Nougat issues fixed, or None if processing fails.
        """
        if self.debug:
            self.logger.info(f"Before Nougat Correction ({filename}): {content[:500]}{'...' if len(content) > 500 else ''}")

        if not content:
            self.logger.error(f"{filename} - Empty content in Nougat Correction")
            return None
            
        try:
            cleaned = postprocess_single(content, markdown_fix=True)

            cleaned = NougatCorrection._clean_latex_table(cleaned)
            
            self.logger.info(f"{filename} - Fixed Nougat Correction")
            
            if self.debug:
                self.logger.info(f"After Nougat Correction ({filename}): {cleaned[:500]}{'...' if len(cleaned) > 500 else ''}")
                
            return cleaned
            
        except Exception as e:
            self.logger.error(f"{filename} - Nougat Correction failed: {str(e)}")
            return content
