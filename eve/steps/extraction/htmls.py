from pathlib import Path
from typing import Optional
import aiofiles
import asyncio
from trafilatura import extract

from eve.utils import read_file
from eve.logging import logger
class HtmlExtractor:
    def __init__(self, input_data: list):
        self.input_data = input_data
        self.extractions = []
    
    async def _extract_single_file(self, file_path: Path) -> Optional[str]:
        text = await read_file(file_path, 'r')
        return text

    async def extract_text(self) -> list:
        """extract text from all HTML files concurrently."""
        tasks = [self._extract_single_file(file_path) for file_path in self.input_data]
        results = await asyncio.gather(*tasks, return_exceptions = True)
        
        self.extractions = [result for result in results 
                          if result is not None and not isinstance(result, Exception)]
        return self.extractions