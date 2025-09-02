import aiohttp
import asyncio
from pathlib import Path
from typing import Optional

from eve.utils import read_file
from eve.logging import logger

class PdfExtractor:
    def __init__(self, file_path: Path, endpoint: str):
        self.file_path = file_path
        self.endpoint = f"{endpoint}/predict"
        self.extraction = None

    async def _call_nougat(self, session: aiohttp.ClientSession) -> Optional[str]:
        """internal method to call the Nougat API."""
        try:
            file_content = await read_file(self.file_path, 'rb')
            if not file_content:
                logger.error(f"Failed to read file: {self.file_path}")
                return None
                
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename = self.file_path.name, content_type = 'application/pdf')
            
            async with session.post(self.endpoint, data = data) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Nougat API request for {self.file_path} failed with status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Failed to process {self.file_path}: {str(e)}")
            return None

    async def extract_text(self) -> Optional[str]:
        """Extract text from a single PDF file."""
        async with aiohttp.ClientSession() as session:
            self.extraction = await self._call_nougat(session)
        return self.extraction if self.extraction else None