"""OCR corrections component for fixing OCR-induced text issues."""

from typing import Optional
import re

from eve.steps.cleaning.base_component import DataProcessingComponent


class OCRCorrections(DataProcessingComponent):
    """Component to correct OCR-induced text issues."""

    def __init__(self, debug: bool = False):
        """Initialize the OCR corrections component.
        
        Args:
            debug: Enable debug output.
        """
        super().__init__(debug=debug)
    
    async def process(self, content: str, filename: str) -> Optional[str]:
        """Process content to fix OCR issues.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Cleaned content with OCR issues fixed, or None if processing fails.
        """
        if self.debug:
            self.logger.info(f"Before OCRCorrections ({filename}): {content[:500]}{'...' if len(content) > 500 else ''}")

        if not content:
            self.logger.error(f"{filename} - Empty content in OCRCorrections")
            return None
            
        try:
            cleaned = re.sub(r'(\d+)([A-Za-z]{2,})', r'\1 \2', content)
            
            self.logger.info(f"{filename} - Fixed OCRCorrections")
            
            if self.debug:
                self.logger.info(f"After OCRCorrections ({filename}): {cleaned[:500]}{'...' if len(cleaned) > 500 else ''}")
                
            return cleaned
            
        except Exception as e:
            self.logger.error(f"{filename} - OCRCorrections failed: {str(e)}")
            return content
