from pathlib import Path
from typing import Optional
import asyncio
import re
import xml.etree.ElementTree as ET

from eve.utils import read_file
from eve.logging import logger

class XmlExtractor:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.output = None

    async def extract_text(self) -> Optional[str]:
        """Extract text from a single XML file."""
        try:
            content = await read_file(self.file_path, 'r')
            if not content:
                logger.error(f"Failed to read file: {self.file_path}")
                return None
            
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
            
            self.output = await asyncio.to_thread(parse_and_extract)
            return self.output if self.output else None
        except Exception as e:
            logger.error(f"Error processing XML file {self.file_path}: {e}")
            return None