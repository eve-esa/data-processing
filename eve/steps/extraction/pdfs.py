import aiohttp
import asyncio

from pathlib import Path
from typing import Optional

from eve.utils import read_file
from eve.logging import logger

class PdfExtractor:
    def __init__(self, input_data: list, endpoint: str):
        self.input_data = input_data
        self.endpoint = f"{endpoint}/predict"
        self.extractions = []

    async def _call_nougat(self, session: aiohttp.ClientSession, file_path: Path) -> Optional[str]:
        """internal method to call the Nougat API."""
        try:
            file_content = await read_file(file_path, 'rb')
                
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename = file_path.name, content_type = 'application/pdf')
            
            async with session.post(self.endpoint, data=data) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return None

    async def extract_text(self) -> list:
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_nougat(session, file_path) for file_path in self.input_data]
            self.extractions = await asyncio.gather(*tasks, return_exceptions = True) # do we need a task manager?
            
            # Filter out exceptions and None results
            self.extractions = [result for result in self.extractions 
                              if result is not None and not isinstance(result, Exception)]
        
        return self.extractions