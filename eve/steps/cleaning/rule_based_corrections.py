"""Rule-based corrections component for fixing common text issues."""

from typing import Optional
import re

from eve.steps.cleaning.base_component import DataProcessingComponent


class RuleBasedCorrections(DataProcessingComponent):
    """Component to apply rule-based text corrections."""

    def __init__(self, debug: bool = False):
        """Initialize the rule-based corrections component.
        
        Args:
            debug: Enable debug output.
        """
        super().__init__(debug=debug)
    
    async def process(self, content: str, filename: str) -> Optional[str]:
        """Process content to apply rule-based corrections.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Cleaned content with rule-based corrections applied, or None if processing fails.
        """
        if self.debug:
            self.logger.info(f"Before Rule Based Correction ({filename}): {content[:500]}{'...' if len(content) > 500 else ''}")

        if not content:
            self.logger.error(f"{filename} - Empty content in Rule Based Correction")
            return None
            
        try:
            cleaned_lines = []
            for line in content.split('\n'):
                stripped = line.strip()

                if not re.search(r'\w', stripped) and len(stripped) == 1:
                    continue
                
                cleaned_lines.append(line)

            cleaned = '\n'.join(cleaned_lines)
            
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

            cleaned = cleaned.strip()
            
            self.logger.info(f"{filename} - Fixed Rule Based Correction")
            
            if self.debug:
                self.logger.info(f"After Rule Based Correction ({filename}): {cleaned[:500]}{'...' if len(cleaned) > 500 else ''}")
                
            return cleaned
            
        except Exception as e:
            self.logger.error(f"{filename} - Rule Based Correction failed: {str(e)}")
            return content
