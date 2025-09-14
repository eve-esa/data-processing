from typing import Optional
import asyncio
from trafilatura import extract

from eve.utils import read_file
from eve.logging import logger
from eve.model.document import Document

class HtmlExtractor:
    def __init__(self, document: Document):
        self.document = document
    
    async def extract_text(self) -> Optional[Document]:
        """Extract text from a single HTML file.
        
        Returns:
            Document object with extracted text if successful, None otherwise
        """
        try:
            content = await read_file(self.document.file_path, 'r')
            if not content:
                logger.error(f"Failed to read file: {self.document.file_path}")
                return None
            
            def parse_html():
                return extract(content, include_comments = False, include_tables = True)
            
            self.document.content = await asyncio.to_thread(parse_html)
            return self.document
        except Exception as e:
            logger.error(f"Error processing HTML file {self.document.file_path}: {e}")
            return None