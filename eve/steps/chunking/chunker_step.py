from typing import List, Dict, Any
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
        self.max_workers = config.get("max_workers", None)  # None = CPU count

        self.chunker = MarkdownTwoStepChunker(
            self.max_chunk_size,
            self.chunk_overlap,
            self.add_headers,
            self.word_overlap,
            self.headers_to_split_on,
            self.merge_small_chunks,
        )

    async def execute(self, documents: List[Document]) -> List[Document]:
        self.logger.info(f"Chunking {len(documents)} documents")
        self.logger.info(f"Using max_chunk_size={self.max_chunk_size}, chunk_overlap={self.chunk_overlap}")
        self.logger.info(f"Parallel processing with max_workers={self.max_workers or 'CPU count'}")

        loop = asyncio.get_event_loop()

        # Serialize documents to plain dicts for pickling
        serialized_docs = [_serialize_document(doc) for doc in documents]

        # Create a partial function with the chunker configuration
        chunk_func = partial(
            _chunk_document,
            max_chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_headers=self.add_headers,
            word_overlap=self.word_overlap,
            headers_to_split_on=self.headers_to_split_on,
            merge_small_chunks=self.merge_small_chunks,
        )

        # Process documents in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, chunk_func, doc)
                for doc in serialized_docs
            ]
            results = await asyncio.gather(*tasks)

        # Flatten and deserialize results
        all_chunks = []
        for doc_chunks in results:
            all_chunks.extend([_deserialize_document(chunk) for chunk in doc_chunks])

        self.logger.info(f"Chunking complete: {len(documents)} documents -> {len(all_chunks)} chunks")

        return all_chunks


def _serialize_document(doc: Document) -> Dict[str, Any]:
    """Convert Document to a plain dict for pickling."""
    return {
        "content": doc.content,
        "file_path": doc.file_path,
        "file_format": doc.file_format,
        "metadata": doc.metadata,
    }


def _deserialize_document(doc_dict: Dict[str, Any]) -> Document:
    """Convert plain dict back to Document."""
    return Document(
        content=doc_dict["content"],
        file_path=doc_dict["file_path"],
        file_format=doc_dict["file_format"],
        metadata=doc_dict["metadata"],
    )


# This function needs to be at module level for pickling
def _chunk_document(
    document_dict: Dict[str, Any],
    max_chunk_size: int,
    chunk_overlap: int,
    add_headers: bool,
    word_overlap: int,
    headers_to_split_on: List[int],
    merge_small_chunks: bool,
) -> List[Dict[str, Any]]:
    """Chunk a single document - designed to run in a separate process."""
    chunker = MarkdownTwoStepChunker(
        max_chunk_size,
        chunk_overlap,
        add_headers,
        word_overlap,
        headers_to_split_on,
        merge_small_chunks,
    )

    chunks = chunker.chunk(document_dict["content"])

    # Convert to serializable dicts
    result_chunks = []
    for chunk in chunks:
        headers = ["#" * key + value for key, value in chunk.metadata.items()]
        result_chunks.append(
            {
                "content": chunk.page_content,
                "file_path": document_dict["file_path"],
                "file_format": document_dict["file_format"],
                "metadata": {"headers": headers, **document_dict["metadata"]},
            }
        )

    return result_chunks


def convert_langchain_doc(doc: Document, chunk: LangchainDocument):
    headers = ["#" * key + value for key, value in chunk.metadata.items()]
    return Document(
        content=chunk.page_content,
        file_path=doc.file_path,
        file_format=doc.file_format,
        metadata={"headers": headers, **doc.metadata},
    )
