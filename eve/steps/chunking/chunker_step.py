"""Document chunking step using semantic two-step chunking strategy."""

from typing import List, Dict, Any
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from eve.model.document import Document
from eve.base_step import PipelineStep

from eve.steps.chunking.chunker import MarkdownTwoStepChunker
from langchain_core.documents import Document as LangchainDocument


class ChunkerStep(PipelineStep):
    """Chunk documents into smaller, semantically meaningful pieces.

    Uses a two-step chunking strategy:
    1. Split by Markdown headers to maintain document structure
    2. Further split large sections by sentences while preserving LaTeX and tables
    3. Optionally merge small chunks that share compatible heading levels

    The chunker processes documents in parallel using multiprocessing for performance.

    Config parameters:
        max_chunk_size (int): Maximum size of any chunk in words (default: 512)
        chunk_overlap (int): Number of characters to overlap between chunks (default: 0)
        word_overlap (int): Number of words to overlap between chunks (default: 0)
        add_headers (bool): Whether to prepend section headers to chunks (default: False)
        merge_small_chunks (bool): Whether to merge small chunks with compatible headers (default: True)
        headers_to_split_on (list[int]): Markdown header levels to split on (default: [1, 2, 3, 4, 5, 6])
        max_workers (int): Number of parallel workers, None uses CPU count (default: None)

    Examples:
        # Basic chunking with default settings
        config: {max_chunk_size: 512}

        # Chunking with headers and overlap for retrieval
        config: {
            max_chunk_size: 512,
            add_headers: true,
            word_overlap: 20,
            merge_small_chunks: true
        }

        # Large chunks for context preservation
        config: {
            max_chunk_size: 2048,
            headers_to_split_on: [1, 2],
            merge_small_chunks: true
        }
    """

    def __init__(self, config: dict):
        """Initialize the chunker step.

        Args:
            config: Configuration dictionary containing chunking parameters
        """
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
        """Execute chunking on documents in parallel.

        Processes each document independently using multiprocessing, then flattens
        all chunks into a single list.

        Args:
            documents: List of documents to chunk

        Returns:
            Flattened list of all chunks from all documents
        """
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
    """Convert Document to a plain dict for pickling.

    Required for multiprocessing as Document objects need to be serialized
    to pass between processes.

    Args:
        doc: Document to serialize

    Returns:
        Dictionary containing document fields
    """
    return {
        "content": doc.content,
        "file_path": doc.file_path,
        "file_format": doc.file_format,
        "metadata": doc.metadata,
    }


def _deserialize_document(doc_dict: Dict[str, Any]) -> Document:
    """Convert plain dict back to Document.

    Reconstructs a Document object from a serialized dictionary.

    Args:
        doc_dict: Dictionary containing document fields

    Returns:
        Reconstructed Document object
    """
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
    """Chunk a single document - designed to run in a separate process.

    Creates a chunker instance and processes the document, converting results
    to serializable dictionaries.

    Args:
        document_dict: Serialized document dictionary
        max_chunk_size: Maximum chunk size in words
        chunk_overlap: Character overlap between chunks
        add_headers: Whether to add section headers to chunks
        word_overlap: Word overlap between chunks
        headers_to_split_on: List of header levels to split on
        merge_small_chunks: Whether to merge small chunks

    Returns:
        List of serialized chunk dictionaries
    """
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


def convert_langchain_doc(doc: Document, chunk: LangchainDocument) -> Document:
    """Convert a LangChain Document chunk to an Eve Document.

    Extracts headers from chunk metadata and combines with original document metadata.

    Args:
        doc: Original Eve Document
        chunk: LangChain Document chunk with header metadata

    Returns:
        Eve Document with chunk content and combined metadata
    """
    headers = ["#" * key + value for key, value in chunk.metadata.items()]
    return Document(
        content=chunk.page_content,
        file_path=doc.file_path,
        file_format=doc.file_format,
        metadata={"headers": headers, **doc.metadata},
    )
