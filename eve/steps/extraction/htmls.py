from pathlib import Path
from typing import Optional
import asyncio
from trafilatura import extract

from eve.utils import read_file
from eve.logging import logger

class HtmlExtractor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.extraction = None
    
    async def extract_text(self) -> Optional[str]:
        """Extract text from a single HTML file."""
        try:
            content = await read_file(self.file_path, 'r')
            if not content:
                logger.error(f"Failed to read file: {self.file_path}")
                return None
            
            def parse_html():
                return extract(content, include_comments=False, include_tables=True)
            
            self.extraction = await asyncio.to_thread(parse_html)
            return self.extraction if self.extraction else None
        except Exception as e:
            logger.error(f"Error processing HTML file {self.file_path}: {e}")
            return None