from pathlib import Path
from typing import Optional
import aiofiles
import asyncio
from trafilatura import extract

from eve.logging import logger
class HtmlExtractor:
    def __init__(self, input_data: list):
        self.input_data = input_data
        self.extractions = []
    
    async def _extract_single_file(self, file_path: Path) -> Optional[str]:
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    data = await f.read()
                    text = await asyncio.to_thread(extract, data)
                    return text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                break
        return None

    async def extract_text(self) -> list:
        """extract text from all HTML files concurrently."""
        tasks = [self._extract_single_file(file_path) for file_path in self.input_data]
        results = await asyncio.gather(*tasks, return_exceptions = True)
        
        self.extractions = [result for result in results 
                          if result is not None and not isinstance(result, Exception)]
        return self.extractions