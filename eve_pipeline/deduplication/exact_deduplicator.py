"""Exact deduplication using hash-based matching."""

import threading
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus
from eve_pipeline.core.enums import HashAlgorithm
from eve_pipeline.core.utils import HashUtils


class ExactDeduplicator(ProcessorBase):
    """Exact deduplication processor using content hashing."""

    def __init__(
        self,
        hash_algorithm: Union[str, HashAlgorithm] = HashAlgorithm.MD5,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        **kwargs,
    ) -> None:
        """Initialize exact deduplicator.

        Args:
            hash_algorithm: Hash algorithm to use.
            normalize_whitespace: Whether to normalize whitespace before hashing.
            case_sensitive: Whether comparison should be case sensitive.
            **kwargs: Additional configuration.
        """
        super().__init__(name="ExactDeduplicator", **kwargs)

        if isinstance(hash_algorithm, str):
            try:
                self.hash_algorithm = HashAlgorithm(hash_algorithm.lower())
            except ValueError:
                raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
        else:
            self.hash_algorithm = hash_algorithm

        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive

        self._lock = threading.RLock()
        self.seen_hashes: set[str] = set()
        self.hash_to_file: dict[str, str] = {}

    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **_kwargs,
    ) -> ProcessorResult:
        """Process content for exact deduplication.

        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **_kwargs: Additional processing parameters.

        Returns:
            ProcessorResult indicating if content is duplicate.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )

        try:
            # Normalize content for comparison
            normalized_content = HashUtils.normalize_content_for_hashing(
                content, self.case_sensitive, self.normalize_whitespace,
            )

            # Calculate hash
            content_hash = HashUtils.calculate_hash(normalized_content, self.hash_algorithm)

            # Thread-safe check and update
            with self._lock:
                is_duplicate = content_hash in self.seen_hashes

                if is_duplicate:
                    duplicate_file = self.hash_to_file.get(content_hash, "unknown")
                    self.logger.info(f"Exact duplicate detected: {input_path} matches {duplicate_file}")

                    return ProcessorResult(
                        status=ProcessorStatus.SKIPPED,
                        input_path=input_path,
                        content=content,
                        metadata={
                            "is_duplicate": True,
                            "duplicate_of": duplicate_file,
                            "content_hash": content_hash,
                            "hash_algorithm": self.hash_algorithm.value,
                        },
                    )
                else:
                    # Mark as seen
                    self.seen_hashes.add(content_hash)
                    if input_path:
                        self.hash_to_file[content_hash] = str(input_path)

                    return ProcessorResult(
                        status=ProcessorStatus.SUCCESS,
                        input_path=input_path,
                        content=content,
                        metadata={
                            "is_duplicate": False,
                            "content_hash": content_hash,
                            "hash_algorithm": self.hash_algorithm.value,
                        },
                    )

        except Exception as e:
            self.logger.error(f"Exact deduplication failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )

    def _process_files_generator(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*.md",
        batch_size: int = 100,
    ) -> Generator[list[str], None, None]:
        """Generator that yields batches of files for memory-efficient processing.

        Args:
            input_dir: Path to input directory (local or S3).
            file_pattern: File pattern to match.
            batch_size: Number of files to process in each batch.

        Yields:
            Batches of file paths.
        """
        from eve_pipeline.storage.factory import StorageFactory

        input_dir_str = str(input_dir)
        storage = StorageFactory.get_storage_for_path(input_dir_str, **self.storage_config)

        # Get all files (this could be optimized further for very large directories)
        all_files = storage.list_files(input_dir_str, file_pattern)

        # Yield files in batches
        for i in range(0, len(all_files), batch_size):
            yield all_files[i:i + batch_size]

    def process_directory(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*.md",
        batch_size: int = 100,
    ) -> dict[str, list[str]]:
        """Find exact duplicates in a directory using memory-efficient processing.

        Args:
            input_dir: Path to input directory (local or S3).
            file_pattern: File pattern to match.
            batch_size: Number of files to process in each batch.

        Returns:
            Dictionary mapping hash to list of duplicate files.
        """
        hash_to_files: dict[str, list[str]] = {}
        total_processed = 0

        for file_batch in self._process_files_generator(input_dir, file_pattern, batch_size):
            self.logger.info(f"Processing batch of {len(file_batch)} files...")

            for file_path in file_batch:
                try:
                    content = self._read_file(file_path)

                    normalized_content = HashUtils.normalize_content_for_hashing(
                        content, self.case_sensitive, self.normalize_whitespace,
                    )
                    content_hash = HashUtils.calculate_hash(normalized_content, self.hash_algorithm)

                    with self._lock:
                        if content_hash not in hash_to_files:
                            hash_to_files[content_hash] = []
                        hash_to_files[content_hash].append(file_path)

                    total_processed += 1

                except Exception as e:
                    self.logger.warning(f"Failed to process {file_path}: {e}")
                    continue

        # Return only duplicates (hashes with multiple files)
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

        if duplicates:
            total_duplicate_files = sum(len(files) - 1 for files in duplicates.values())
            self.logger.info(
                f"Processed {total_processed} files, found {len(duplicates)} duplicate groups "
                f"containing {total_duplicate_files} duplicate files",
            )
        else:
            self.logger.info(f"Processed {total_processed} files, no exact duplicates found")

        return duplicates



    def reset(self) -> None:
        """Reset the deduplicator state in a thread-safe manner."""
        with self._lock:
            self.seen_hashes.clear()
            self.hash_to_file.clear()

    def get_statistics(self) -> dict[str, int]:
        """Get deduplication statistics in a thread-safe manner.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            return {
                "unique_hashes": len(self.seen_hashes),
                "files_processed": len(self.hash_to_file),
                "hash_algorithm": self.hash_algorithm.value,
            }
