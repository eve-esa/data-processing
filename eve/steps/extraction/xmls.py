from pathlib import Path
from typing import Optional
import aiofiles
import asyncio
import re
import xml.etree.ElementTree as ET

from eve.logging import logger
class XmlExtractor:
    def __init__(self, input_data: list):
        self.input_data = input_data
        self.outputs = []

    async def _extract_single_file(self, file_path: Path) -> Optional[str]:
        """extract text from a single XML file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            def parse_and_extract():
                
                root = ET.fromstring(content)
                
                def extract_text_from_tree(element):
                    texts = []
                    if element.text:
                        texts.append(element.text)
                    for child in element:
                        texts.extend(extract_text_from_tree(child))
                    if element.tail:
                        texts.append(element.tail)
                    return texts
                
                extracted_texts = extract_text_from_tree(root)
                full_text = ''.join(extracted_texts)

                cleaned_text = re.sub(r'\n{3,}', '\n\n', full_text)
                return cleaned_text.strip()
            
            return await asyncio.to_thread(parse_and_extract)
            
        except Exception as e:
            logger.error(f"Error processing XML file {file_path}: {e}")
            return None

    async def extract_text(self) -> list:
        """extract text from all XML files concurrently."""
        tasks = [self._extract_single_file(file_path) for file_path in self.input_data]
        results = await asyncio.gather(*tasks, return_exceptions = True)
        
        self.outputs = [result for result in results 
                       if result is not None and not isinstance(result, Exception)]
        return self.outputs