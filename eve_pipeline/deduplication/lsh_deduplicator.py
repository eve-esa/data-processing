"""LSH-based approximate deduplication using MinHash."""

import time
from pathlib import Path
from typing import Optional, Union

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus


class LSHDeduplicator(ProcessorBase):
    """LSH-based approximate deduplication processor using MinHash."""

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_size: int = 3,
        batch_size: int = 1000,
        **kwargs,
    ) -> None:
        """Initialize LSH deduplicator.

        Args:
            threshold: Jaccard similarity threshold for near-duplicates.
            num_perm: Number of permutations for MinHash.
            shingle_size: Size of n-grams for comparison.
            batch_size: Process files in batches to manage memory.
            **kwargs: Additional configuration.
        """
        super().__init__(name="LSHDeduplicator", **kwargs)
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.batch_size = batch_size

        # Initialize LSH and storage
        self._initialize_lsh()
        self.file_hashes: dict[str, any] = {}
        self.processed_files: set[str] = set()

    def _initialize_lsh(self) -> None:
        """Initialize LSH components."""
        try:
            from datasketch import MinHash, MinHashLSH
            self.MinHash = MinHash
            self.MinHashLSH = MinHashLSH

            # Create LSH index
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

            self.logger.info(f"Initialized LSH with threshold={self.threshold}, num_perm={self.num_perm}")

        except ImportError as e:
            raise ImportError(f"datasketch not available: {e}")

    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **_kwargs,
    ) -> ProcessorResult:
        """Process content for LSH-based deduplication.

        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.

        Returns:
            ProcessorResult indicating if content is near-duplicate.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )

        try:
            # Create shingles from content
            shingles = self._create_shingles(content)

            if not shingles:
                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    content=content,
                    metadata={"skip_reason": "No shingles generated from content"},
                )

            # Create MinHash
            minhash = self.MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                minhash.update(shingle.encode('utf8'))

            # Check for near-duplicates
            file_key = str(input_path) if input_path else f"content_{hash(content)}"

            if file_key in self.processed_files:
                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    content=content,
                    metadata={"skip_reason": "File already processed"},
                )

            # Query LSH for similar documents
            candidates = self.lsh.query(minhash)

            # Filter out self-matches
            similar_files = [c for c in candidates if c != file_key]

            if similar_files:
                # Calculate actual similarities for reporting
                similarities = []
                for similar_file in similar_files:
                    if similar_file in self.file_hashes:
                        similarity = minhash.jaccard(self.file_hashes[similar_file])
                        similarities.append({
                            "file": similar_file,
                            "similarity": similarity,
                        })

                self.logger.info(f"Near-duplicate detected: {file_key} similar to {len(similar_files)} files")

                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    content=content,
                    metadata={
                        "is_near_duplicate": True,
                        "similar_files": similar_files,
                        "similarities": similarities,
                        "threshold": self.threshold,
                        "shingles_count": len(shingles),
                    },
                )
            else:
                # Add to LSH index
                self.lsh.insert(file_key, minhash)
                self.file_hashes[file_key] = minhash
                self.processed_files.add(file_key)

                return ProcessorResult(
                    status=ProcessorStatus.SUCCESS,
                    input_path=input_path,
                    content=content,
                    metadata={
                        "is_near_duplicate": False,
                        "threshold": self.threshold,
                        "shingles_count": len(shingles),
                    },
                )

        except Exception as e:
            self.logger.error(f"LSH deduplication failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )

    def _create_shingles(self, text: str) -> set[str]:
        """Create n-gram shingles from text."""
        try:
            # Clean and normalize text
            import re

            from nltk import ngrams
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            words = cleaned.split()

            if len(words) < self.shingle_size:
                return set()

            return {' '.join(gram) for gram in ngrams(words, self.shingle_size)}

        except ImportError:
            # Fallback without NLTK
            import re
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            words = cleaned.split()

            if len(words) < self.shingle_size:
                return set()

            shingles = set()
            for i in range(len(words) - self.shingle_size + 1):
                shingle = ' '.join(words[i:i + self.shingle_size])
                shingles.add(shingle)

            return shingles

    def process_directory(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*.md",
    ) -> list[list[str]]:
        """Find near-duplicates in a directory using LSH.

        Args:
            input_dir: Path to input directory (local or S3).
            file_pattern: File pattern to match.

        Returns:
            List of duplicate groups (each group is a list of similar files).
        """
        from eve_pipeline.storage.factory import StorageFactory

        input_dir_str = str(input_dir)
        storage = StorageFactory.get_storage_for_path(input_dir_str, **self.storage_config)
        files = storage.list_files(input_dir_str, file_pattern)

        if not files:
            self.logger.warning(f"No files found matching pattern {file_pattern}")
            return []

        self.logger.info(f"Processing {len(files)} files for near-duplicates")

        start_time = time.time()

        # Process files in batches
        for i in range(0, len(files), self.batch_size):
            batch = files[i:i + self.batch_size]
            self._process_batch(batch)

            if (i + self.batch_size) % (self.batch_size * 10) == 0:
                self.logger.info(f"Processed {min(i + self.batch_size, len(files))}/{len(files)} files")

        # Find duplicate groups
        duplicate_groups = self._find_duplicate_groups(files)

        end_time = time.time()
        self.logger.info(f"LSH processing completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Found {len(duplicate_groups)} groups of near-duplicates")

        return duplicate_groups

    def _process_batch(self, files: list[str]) -> None:
        """Process a batch of files."""
        for file_path in files:
            try:
                # Read file content (file_path is already a string from storage.list_files)
                content = self._read_file(file_path)

                # Process with LSH
                self.process(content, file_path)

            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                continue

    def _find_duplicate_groups(self, files: list[str]) -> list[list[str]]:
        """Find groups of near-duplicate files."""
        processed = set()
        groups = []

        for file_path in files:
            file_key = file_path  # file_path is already a string

            if file_key in processed or file_key not in self.file_hashes:
                continue

            # Query for similar files
            minhash = self.file_hashes[file_key]
            candidates = self.lsh.query(minhash)

            # Remove self and filter existing files
            candidates = [c for c in candidates if c != file_key and c in self.file_hashes]

            if candidates:
                # Create group including the current file and its near-duplicates
                group = [file_key, *candidates]
                # Sort for consistent ordering
                group = sorted(group)

                # Only add if not already added
                if group not in groups:
                    groups.append(group)

                # Mark all files in group as processed
                processed.update(group)

        return groups



    def reset(self) -> None:
        """Reset the deduplicator state."""
        self.lsh = self.MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.file_hashes.clear()
        self.processed_files.clear()

    def get_statistics(self) -> dict[str, int]:
        """Get deduplication statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "files_processed": len(self.processed_files),
            "unique_hashes": len(self.file_hashes),
            "lsh_threshold": self.threshold,
            "num_permutations": self.num_perm,
            "shingle_size": self.shingle_size,
        }
