from pathlib import Path
import aiofiles
from typing import List

from eve.model.document import Document
from eve.base_step import PipelineStep

from eve.steps.chunking.chunker import MarkdownTwoStepChunker
from langchain_core.documents import Document as LangchainDocument


class ChunkerStep(PipelineStep):

    def __init__(self, config: dict):
        super().__init__(config, name="ChunkerStep")

        self.chunk_overlap = config.get("chunk_overlap", 0)
        self.max_chunk_size = config.get("max_chunk_size", 512)
        self.word_overlap = config.get("word_overlap", 0)
        self.add_headers = config.get("add_headers", False)
        self.merge_small_chunks = config.get("merge_small_chunks", True)
        self.headers_to_split_on = config.get("headers_to_split_on", [1, 2, 3, 4, 5, 6])

        self.chunker = MarkdownTwoStepChunker(
            self.max_chunk_size,
            self.chunk_overlap,
            self.add_headers,
            self.word_overlap,
            self.headers_to_split_on,
            self.merge_small_chunks,
        )

    async def execute(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for document in documents:
            chunks = self.chunker.chunk(document.content)
            langchain_docs = [
                convert_langchain_doc(document, chunk) for i, chunk in enumerate(chunks)
            ]
            all_chunks.extend(langchain_docs)
        return all_chunks


def convert_langchain_doc(doc: Document, chunk: LangchainDocument):
    return Document(
        content=chunk.page_content,
        file_path=doc.file_path,
        file_format=doc.file_format,
        metadata={"headers": chunk.metadata.get("headers", []), **doc.metadata},
    )
