from typing import Optional, List
import asyncio
from trafilatura import extract
import json

from eve.utils import read_file
from eve.logging import logger
from eve.model.document import Document


class JSONLExtractor:
    def __init__(self, document: Document):
        self.document = document

    async def extract_documents(self) -> Optional[List[Document]]:
        """Extract text from a single HTML file.

        Returns:
            Document object with extracted text if successful, None otherwise
        """
        try:
            docs: List[Document] = []
            with open(self.document.file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    json_doc = json.loads(line.strip())
                    if "content" not in json_doc:
                        logger.warning(
                            f"No content found in {self.document.file_path} line {i+1}"
                        )
                    else:
                        docs.append(
                            Document(
                                file_path=self.document.file_path,
                                content=json_doc["content"],
                                metadata=json_doc.get("metadata", {}),
                                file_format="md",
                            )
                        )
                return docs
        except Exception as e:
            logger.error(f"Error processing JSONL file {self.document.file_path}: {e}")
            return None
