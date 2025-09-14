import aiohttp
from typing import Optional

from eve.model.document import Document
from eve.utils import read_file
from eve.logging import logger

class PdfExtractor:
    def __init__(self, document: Document, endpoint: str):
        self.document = document
        self.endpoint = f"{endpoint}/predict"
        self.extraction = None

    async def _call_nougat(self, session: aiohttp.ClientSession) -> Optional[str]:
        """internal method to call the Nougat API."""
        try:
            file_content = await read_file(self.document.file_path, 'rb')
            if not file_content:
                logger.error(f"Failed to read file: {self.file_path}")
                return None
                
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename = self.document.filename, content_type = 'application/pdf')
            
            async with session.post(self.endpoint, data = data) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Nougat API request for {self.document.file_path} failed with status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Failed to process {self.document.file_path}: {str(e)}")
            return None

    async def extract_text(self) -> Optional[Document]:
        """Extract text from a single PDF file.
        
        Returns:
            Document object with extracted text if successful, None otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                content = await self._call_nougat(session)
                if not content:
                    logger.error(f"Failed to extract content from {self.document.file_path}")
                    return None
                self.document.content = content
                return self.document
        except Exception as e:
            logger.error(f"Error in PDF extraction for {self.document.file_path}: {str(e)}")
            return None