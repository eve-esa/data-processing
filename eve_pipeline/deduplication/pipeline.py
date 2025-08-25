"""Deduplication pipeline orchestrator."""

from pathlib import Path
from typing import Optional, Union

from eve_pipeline.core.base import ProcessorResult, ProcessorStatus
from eve_pipeline.core.config import DeduplicationConfig
from eve_pipeline.deduplication.exact_deduplicator import ExactDeduplicator
from eve_pipeline.deduplication.lsh_deduplicator import LSHDeduplicator
from eve_pipeline.storage.factory import StorageFactory


class DeduplicationPipeline:
    """Pipeline for two-level deduplication: exact and approximate."""

    def __init__(
        self,
        config: Optional[DeduplicationConfig] = None,
        debug: bool = False,
        storage_config: Optional[dict] = None,
    ) -> None:
        """Initialize deduplication pipeline.

        Args:
            config: Deduplication configuration.
            debug: Enable debug logging.
            storage_config: Storage configuration for S3/local file operations.
        """
        self.config = config or DeduplicationConfig()
        self.debug = debug
        self.storage_config = storage_config or {}

        # Set up logging
        import logging
        self.logger = logging.getLogger("eve_pipeline.DeduplicationPipeline")
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Initialize deduplicators
        self.exact_deduplicator = None
        self.lsh_deduplicator = None

        if self.config.exact_deduplication:
            self.exact_deduplicator = ExactDeduplicator(
                debug=debug,
                storage_config=self.storage_config,
            )

        if self.config.lsh_deduplication:
            self.lsh_deduplicator = LSHDeduplicator(
                threshold=self.config.lsh_threshold,
                num_perm=self.config.lsh_num_perm,
                shingle_size=self.config.lsh_shingle_size,
                debug=debug,
                storage_config=self.storage_config,
            )

    def process_content(
        self,
        content: str,
        input_path: Optional[Path] = None,
    ) -> ProcessorResult:
        """Process content through the deduplication pipeline.

        Args:
            content: Input content to check for duplicates.
            input_path: Optional input file path.

        Returns:
            ProcessorResult with deduplication outcome.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )

        all_metadata = {}

        # Step 1: Exact deduplication
        if self.exact_deduplicator:
            exact_result = self.exact_deduplicator.process(content, input_path)
            all_metadata["exact_deduplication"] = exact_result.metadata or {}

            if exact_result.is_skipped:
                # Exact duplicate found
                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    content=content,
                    metadata={
                        "duplicate_type": "exact",
                        "deduplication_stage": "exact",
                        **all_metadata,
                    },
                )
            elif exact_result.is_failed:
                return ProcessorResult(
                    status=ProcessorStatus.FAILED,
                    input_path=input_path,
                    content=content,
                    error_message=f"Exact deduplication failed: {exact_result.error_message}",
                    metadata=all_metadata,
                )

        # Step 2: LSH-based approximate deduplication
        if self.lsh_deduplicator:
            lsh_result = self.lsh_deduplicator.process(content, input_path)
            all_metadata["lsh_deduplication"] = lsh_result.metadata or {}

            if lsh_result.is_skipped:
                # Near-duplicate found
                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    content=content,
                    metadata={
                        "duplicate_type": "near_duplicate",
                        "deduplication_stage": "lsh",
                        **all_metadata,
                    },
                )
            elif lsh_result.is_failed:
                return ProcessorResult(
                    status=ProcessorStatus.FAILED,
                    input_path=input_path,
                    content=content,
                    error_message=f"LSH deduplication failed: {lsh_result.error_message}",
                    metadata=all_metadata,
                )

        # No duplicates found
        return ProcessorResult(
            status=ProcessorStatus.SUCCESS,
            input_path=input_path,
            content=content,
            metadata={
                "duplicate_type": "unique",
                "deduplication_stage": "completed",
                **all_metadata,
            },
        )

    def process_directory(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*.md",
    ) -> dict[str, any]:
        """Process all files in a directory for deduplication.

        Args:
            input_dir: Path to input directory (local or S3).
            file_pattern: File pattern to match.

        Returns:
            Dictionary with deduplication results and statistics.
        """
        input_dir_str = str(input_dir)

        # Use storage factory to handle both local and S3 paths
        storage = StorageFactory.get_storage_for_path(input_dir_str, **self.storage_config)

        if not storage.exists(input_dir_str):
            return {
                "success": False,
                "error_message": f"Input directory does not exist: {input_dir}",
                "total_files": 0,
                "results": {},
            }

        # List files using storage backend
        self.logger.info(f"Searching for files in {input_dir} with pattern: {file_pattern}")
        files = storage.list_files(input_dir_str, file_pattern)

        if not files:
            self.logger.warning(f"No files found matching pattern {file_pattern} in {input_dir}")
            return {
                "success": False,
                "error_message": f"No files found matching pattern {file_pattern}",
                "total_files": 0,
                "results": {},
            }

        self.logger.info(f"Found {len(files)} files to process for deduplication")
        if self.debug:
            self.logger.debug(f"Files to process: {files[:10]}{'...' if len(files) > 10 else ''}")

        # Log which deduplication stages are enabled
        stages = []
        if self.config.exact_deduplication:
            stages.append("exact deduplication")
        if self.config.lsh_deduplication:
            stages.append("LSH near-duplicate detection")
        self.logger.info(f"Deduplication stages enabled: {stages}")

        results = {
            "total_files": len(files),
            "exact_duplicates": {},
            "near_duplicates": [],
            "unique_files": [],
            "failed_files": [],
        }

        # Step 1: Exact deduplication
        if self.config.exact_deduplication and self.exact_deduplicator:
            self.logger.info(f"Starting exact deduplication on {len(files)} files...")
            exact_duplicates = self.exact_deduplicator.process_directory(input_dir, file_pattern)
            results["exact_duplicates"] = exact_duplicates

            # Collect files that are exact duplicates (keep only first occurrence)
            exact_duplicate_files = set()
            for group in exact_duplicates.values():
                # Skip the first file (keep as original), mark others as duplicates
                for duplicate_file in group[1:]:
                    exact_duplicate_files.add(duplicate_file)
        else:
            exact_duplicate_files = set()

        # Step 2: LSH deduplication on remaining files
        if self.config.lsh_deduplication and self.lsh_deduplicator:
            # Reset LSH deduplicator for fresh analysis
            self.lsh_deduplicator.reset()

            # Filter out exact duplicates
            remaining_files = [f for f in files if str(f) not in exact_duplicate_files]

            if remaining_files:
                self.logger.info(f"Starting LSH near-duplicate detection on {len(remaining_files)} remaining files...")

                # Create temporary directory list for LSH processing
                _ = Path(input_dir)

                # Process remaining files
                near_duplicate_groups = []
                for file_path in remaining_files:
                    try:
                        content = self._read_file(file_path)
                        result = self.lsh_deduplicator.process(content, file_path)

                        if result.is_failed:
                            results["failed_files"].append({
                                "file": str(file_path),
                                "error": result.error_message,
                            })
                    except Exception as e:
                        results["failed_files"].append({
                            "file": str(file_path),
                            "error": str(e),
                        })

                # Get near-duplicate groups
                near_duplicate_groups = self.lsh_deduplicator._find_duplicate_groups(remaining_files)
                results["near_duplicates"] = near_duplicate_groups

                # Collect files that are near-duplicates
                near_duplicate_files = set()
                for group in near_duplicate_groups:
                    # Skip the first file (keep as original), mark others as near-duplicates
                    for duplicate_file in group[1:]:
                        near_duplicate_files.add(duplicate_file)
            else:
                self.logger.info("No files remaining after exact deduplication for LSH processing")
                near_duplicate_files = set()
        else:
            near_duplicate_files = set()

        # Determine unique files
        all_duplicate_files = exact_duplicate_files | near_duplicate_files
        failed_file_paths = {item["file"] for item in results["failed_files"]}

        results["unique_files"] = [
            str(f) for f in files
            if str(f) not in all_duplicate_files and str(f) not in failed_file_paths
        ]

        # Calculate statistics
        stats = self._calculate_statistics(results)
        results.update(stats)

        return {
            "success": True,
            "input_directory": str(input_dir),
            "file_pattern": file_pattern,
            **results,
        }

    def _calculate_statistics(self, results: dict[str, any]) -> dict[str, any]:
        """Calculate deduplication statistics."""
        total_files = results["total_files"]

        exact_duplicate_count = sum(
            len(group) - 1 for group in results["exact_duplicates"].values()
        )

        near_duplicate_count = sum(
            len(group) - 1 for group in results["near_duplicates"]
        )

        unique_count = len(results["unique_files"])
        failed_count = len(results["failed_files"])

        return {
            "statistics": {
                "total_files": total_files,
                "exact_duplicates": exact_duplicate_count,
                "near_duplicates": near_duplicate_count,
                "unique_files": unique_count,
                "failed_files": failed_count,
                "exact_duplicate_groups": len(results["exact_duplicates"]),
                "near_duplicate_groups": len(results["near_duplicates"]),
                "deduplication_rate": (exact_duplicate_count + near_duplicate_count) / total_files * 100 if total_files > 0 else 0,
            },
        }

    def _read_file(self, file_path: Union[str, Path]) -> str:
        """Read file using storage factory (supports both local and S3 paths)."""
        file_path_str = str(file_path)
        storage = StorageFactory.get_storage_for_path(file_path_str, **self.storage_config)

        try:
            return storage.read_text(file_path_str)
        except Exception as e:
            raise Exception(f"Cannot read file {file_path_str}: {e!s}")

    def reset(self) -> None:
        """Reset all deduplicators."""
        if self.exact_deduplicator:
            self.exact_deduplicator.reset()
        if self.lsh_deduplicator:
            self.lsh_deduplicator.reset()

    def get_statistics(self) -> dict[str, any]:
        """Get combined statistics from all deduplicators."""
        stats = {}

        if self.exact_deduplicator:
            stats["exact_deduplication"] = self.exact_deduplicator.get_statistics()

        if self.lsh_deduplicator:
            stats["lsh_deduplication"] = self.lsh_deduplicator.get_statistics()

        return stats
