"""
Streaming file processor for memory-efficient handling of large files.
"""

import gc
import logging
import mmap
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

from eve_pipeline.storage.factory import StorageFactory


class StreamingFileProcessor:
    """
    Memory-efficient file processor that handles large files using streaming.
    """

    def __init__(self, chunk_size: int = 8192, overlap: int = 512, storage_config: Optional[dict] = None):
        """
        Initialize streaming processor.

        Args:
            chunk_size: Size of each chunk in characters/bytes
            overlap: Overlap between chunks to maintain context
            storage_config: Storage configuration for S3/local files
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.storage_config = storage_config or {}
        self.logger = logging.getLogger(__name__)

    def stream_file_content(self, file_path: Union[str, Path]) -> Generator[str, None, None]:
        """
        Stream file content in chunks with overlap.

        Args:
            file_path: Path to file (local or S3)

        Yields:
            Chunks of file content with overlap
        """
        file_path_str = str(file_path)
        storage = StorageFactory.get_storage_for_path(file_path_str, **self.storage_config)

        if file_path_str.startswith('s3://'):
            yield from self._stream_s3_content(storage, file_path_str)
        else:
            yield from self._stream_local_content(Path(file_path_str))

    def _stream_local_content(self, file_path: Path) -> Generator[str, None, None]:
        """Stream local file content efficiently using memory mapping when possible."""
        try:
            file_size = file_path.stat().st_size

            # Use memory mapping for large files
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                yield from self._stream_with_mmap(file_path)
            else:
                yield from self._stream_with_standard_io(file_path)

        except Exception as e:
            self.logger.error(f"Error streaming file {file_path}: {e}")
            raise

    def _stream_with_mmap(self, file_path: Path) -> Generator[str, None, None]:
        """Stream large files using memory mapping."""
        try:
            with (
                open(file_path, encoding='utf-8', errors='ignore') as f,
                mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
            ):
                position = 0
                previous_chunk_end = ""

                while position < len(mm):
                    # Calculate chunk boundaries
                    start_pos = max(0, position - self.overlap)
                    end_pos = min(len(mm), position + self.chunk_size)

                    # Read chunk
                    mm.seek(start_pos)
                    chunk_bytes = mm.read(end_pos - start_pos)
                    chunk = chunk_bytes.decode('utf-8', errors='ignore')

                    # Handle overlap
                    if position > 0:
                        chunk = previous_chunk_end + chunk

                    # Store overlap for next iteration
                    if len(chunk) > self.overlap:
                        previous_chunk_end = chunk[-self.overlap:]

                    yield chunk
                    position = end_pos

                    # Force garbage collection periodically
                    if position % (self.chunk_size * 10) == 0:
                        gc.collect()

        except Exception as e:
            self.logger.error(f"Memory mapping failed for {file_path}, falling back to standard I/O: {e}")
            yield from self._stream_with_standard_io(file_path)

    def _stream_with_standard_io(self, file_path: Path) -> Generator[str, None, None]:
        """Stream files using standard I/O with chunking."""
        previous_chunk_end = ""

        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break

                    # Add overlap from previous chunk
                    if previous_chunk_end:
                        chunk = previous_chunk_end + chunk

                    # Store overlap for next iteration
                    if len(chunk) > self.overlap:
                        previous_chunk_end = chunk[-self.overlap:]
                    else:
                        previous_chunk_end = chunk

                    yield chunk

        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    yield from self._stream_with_encoding(file_path, encoding)
                    return
                except UnicodeDecodeError:
                    continue

            self.logger.error(f"Could not decode file {file_path} with any encoding")
            raise

    def _stream_with_encoding(self, file_path: Path, encoding: str) -> Generator[str, None, None]:
        """Stream file with specific encoding."""
        previous_chunk_end = ""

        with open(file_path, encoding=encoding, errors='ignore') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break

                if previous_chunk_end:
                    chunk = previous_chunk_end + chunk

                if len(chunk) > self.overlap:
                    previous_chunk_end = chunk[-self.overlap:]
                else:
                    previous_chunk_end = chunk

                yield chunk

    def _stream_s3_content(self, storage, s3_path: str) -> Generator[str, None, None]:
        """Stream S3 file content in chunks."""
        try:
            # Get file size first
            metadata = storage.get_metadata(s3_path)
            file_size = metadata.get('size', 0)

            if file_size == 0:
                return

            # Stream in chunks using range requests
            position = 0
            previous_chunk_end = ""

            while position < file_size:
                # Calculate range
                end_pos = min(file_size - 1, position + self.chunk_size - 1)

                # Read chunk with range
                content = storage.read_range(s3_path, position, end_pos)

                if not content:
                    break

                # Decode content
                if isinstance(content, bytes):
                    chunk = content.decode('utf-8', errors='ignore')
                else:
                    chunk = content

                # Handle overlap
                if position > 0:
                    chunk = previous_chunk_end + chunk

                # Store overlap for next iteration
                if len(chunk) > self.overlap:
                    previous_chunk_end = chunk[-self.overlap:]

                yield chunk
                position = end_pos + 1

        except Exception as e:
            self.logger.error(f"Error streaming S3 file {s3_path}: {e}")
            # Fallback to reading entire file
            content = storage.read_file(s3_path)
            yield from self._chunk_content(content)

    def _chunk_content(self, content: str) -> Generator[str, None, None]:
        """Chunk already-loaded content."""
        position = 0
        content_len = len(content)
        previous_chunk_end = ""

        while position < content_len:
            end_pos = min(content_len, position + self.chunk_size)
            chunk = content[position:end_pos]

            if position > 0:
                chunk = previous_chunk_end + chunk

            if len(chunk) > self.overlap:
                previous_chunk_end = chunk[-self.overlap:]

            yield chunk
            position = end_pos

    def process_file_streaming(
        self,
        file_path: Union[str, Path],
        processor_func,
        combine_func=None,
    ) -> str:
        """
        Process a file using streaming with a processor function.

        Args:
            file_path: Path to file
            processor_func: Function to process each chunk
            combine_func: Function to combine processed chunks (default: concatenate)

        Returns:
            Combined processed content
        """
        if combine_func is None:
            def combine_func(chunks):
                return ''.join(chunks)

        processed_chunks = []

        try:
            for chunk in self.stream_file_content(file_path):
                processed_chunk = processor_func(chunk)
                if processed_chunk:
                    processed_chunks.append(processed_chunk)

                # Periodic memory cleanup
                if len(processed_chunks) % 50 == 0:
                    gc.collect()

            return combine_func(processed_chunks)

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            raise
        finally:
            # Final cleanup
            gc.collect()


class StreamingMarkdownProcessor(StreamingFileProcessor):
    """Specialized streaming processor for markdown files."""

    def __init__(self, chunk_size: int = 16384, overlap: int = 1024, **kwargs):
        """Initialize with markdown-optimized settings."""
        super().__init__(chunk_size=chunk_size, overlap=overlap, **kwargs)

    def process_markdown_streaming(self, file_path: Union[str, Path], cleaning_components) -> str:
        """
        Process markdown file using streaming with cleaning components.

        Args:
            file_path: Path to markdown file
            cleaning_components: List of cleaning components to apply

        Returns:
            Cleaned markdown content
        """
        def process_chunk(chunk: str) -> str:
            """Apply cleaning components to chunk."""
            processed = chunk
            for component in cleaning_components:
                try:
                    processed = component.process(processed, self.logger, Path(file_path).name)
                    if processed is None:
                        return ""
                except Exception as e:
                    self.logger.warning(f"Component {component.__class__.__name__} failed: {e}")
            return processed

        def combine_chunks(chunks) -> str:
            """Smart combination that handles markdown structure."""
            if not chunks:
                return ""

            # Simple concatenation for now - could be enhanced
            # to handle markdown structure preservation
            return ''.join(chunks)

        return self.process_file_streaming(file_path, process_chunk, combine_chunks)
