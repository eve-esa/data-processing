from typing import Optional, List
import json

from eve.utils import read_file
from eve.logging import logger
from eve.model.document import Document


class JSONLExtractor:
    def __init__(self, document: Document):
        self.document = document

    async def extract_documents(self) -> Optional[List[Document]]:
        """Extract text from a single Jsonl file.

        Returns:
            Document object with extracted text if successful, None otherwise
        """
        try:
            input_docs = await read_file(self.document.file_path, "r")
            input_docs = input_docs.split("\n")

            docs: List[Document] = []
            for i, doc in enumerate(input_docs):
                json_doc = json.loads(doc)
                if "content" not in json_doc:
                    logger.warning(
                        f"No content found in {self.document.file_path} line {i+1}"
                    )
                else:
                    docs.append(
                        Document(
                            file_path = self.document.file_path,
                            content = json_doc["content"],
                            metadata = json_doc.get("metadata", {}),
                            file_format = "md",
                        )
                    )
            return docs
        except Exception as e:
            logger.error(f"Error processing JSONL file {self.document.file_path}: {e}")
            return None
