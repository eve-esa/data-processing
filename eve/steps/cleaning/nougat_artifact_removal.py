"""Nougat artifact removal component for removing Nougat-specific artifacts."""

from typing import Optional
import re

from eve.steps.cleaning.base_component import DataProcessingComponent


class NougatArtifactRemovalComponent(DataProcessingComponent):
    """Component to remove Nougat-specific artifacts from text."""

    def __init__(self, debug: bool = False):
        """Initialize the Nougat artifact removal component.
        
        Args:
            debug: Enable debug output.
        """
        super().__init__(debug=debug)

    async def process(self, content: str, filename: str) -> Optional[str]:
        """Process content to remove Nougat artifacts.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Cleaned content with Nougat artifacts removed, or None if processing fails.
        """
        if self.debug:
            self.logger.info(f"Before NougatArtifactRemovalComponent ({filename}): {content[:500]}{'...' if len(content) > 500 else ''}")

        if not content:
            self.logger.error(f"{filename} - Empty content in Nougat artifact removal")
            return None
            
        try:
            cleaned = content.strip('"')
            
            cleaned = cleaned.replace('\\n', '\n')

            cleaned = re.sub(r'\+\+\+\s*==WARNING: Truncated because of repetitions==.*?\+\+\+',
                            '', cleaned, flags=re.DOTALL)
            
            cleaned = re.sub(r'\+\+\+\s*==ERROR: No output for this page==.*?\+\+\+',
                            '', cleaned, flags=re.DOTALL)
            
            cleaned = cleaned.replace('[MISSING_PAGE_POST]', '')
            
            self.logger.info(f"{filename} - Removed Nougat artifacts")
            
            if self.debug:
                self.logger.info(f"After NougatArtifactRemovalComponent ({filename}): {cleaned[:500]}{'...' if len(cleaned) > 500 else ''}")
                
            return cleaned
            
        except Exception as e:
            self.logger.error(f"{filename} - Nougat artifact removal failed: {str(e)}")
            return content
