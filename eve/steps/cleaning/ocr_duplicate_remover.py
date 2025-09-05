"""OCR duplicate remover component for removing OCR-induced duplicate text."""

from typing import Optional
import re

from eve.steps.cleaning.base_component import DataProcessingComponent


class OCRDuplicateRemover(DataProcessingComponent):
    """
    Component to detect and remove OCR-induced duplicate text segments.
    Uses sub-string matching with a threshold to remove near duplicates.
    """
    
    def __init__(self, 
                 threshold: float = 0.99,
                 min_words: int = 2,
                 debug: bool = False):
        """
        Initialize the OCR duplicate remover.
        
        Args:
            threshold: Similarity threshold for duplicates.
            min_words: Minimum words required for a unit to be processed.
            debug: Enable debug output.
        """
        super().__init__(debug=debug)
        self.threshold = threshold
        self.min_words = min_words
    
    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """Check if a line is noise (empty or only symbols)."""
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
        return overlap / len(set1) >= self.threshold or overlap / len(set2) >= self.threshold
    
    def _remove_near_adjacent_duplicates(self, content: str, filename: str) -> tuple[str, list[str]]:
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

            j = i + 1
            while j < len(sentences) and self._is_noise_line(sentences[j]):
                j += 1

            if j < len(sentences) and self._is_similar(current, sentences[j]):
                self.logger.info(f"{filename} - Removing near-duplicate: {repr(sentences[j])}")
                removed.append(sentences[j])
                i = j
            else:
                cleaned.append(current)
                i += 1

        return '\n'.join(cleaned), removed
    
    async def process(self, content: str, filename: str) -> Optional[str]:
        """
        Process content to remove OCR-induced duplicate text segments.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Cleaned content with duplicates removed, or None if processing fails.
        """
        if self.debug:
            self.logger.info(f"Before OCRDuplicateRemover ({filename}): {content[:500]}{'...' if len(content) > 500 else ''}")

        if not content:
            self.logger.error(f"{filename} - Empty content in OCRDuplicateRemover")
            return None

        try:
            cleaned_content, removed = self._remove_near_adjacent_duplicates(content, filename)
            
            percent_removed = 0.0
            if content:
                percent_removed = (len(content) - len(cleaned_content)) / len(content) * 100
            self.logger.info(f"{filename} - OCRDuplicateRemover removed {len(removed)} segments, {percent_removed:.2f}% of text removed")

            if self.debug:
                self.logger.info(f"After OCRDuplicateRemover ({filename}): {cleaned_content[:500]}{'...' if len(cleaned_content) > 500 else ''}")
            
            return cleaned_content
            
        except Exception as e:
            self.logger.error(f"{filename} - OCRDuplicateRemover failed: {str(e)}")
            return content
