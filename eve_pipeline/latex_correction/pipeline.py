"""LaTeX correction pipeline for mathematical formula validation and correction."""

import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Optional, Union

from eve_pipeline.cleaning.processors import LatexCorrector
from eve_pipeline.core.base import ProcessorResult, ProcessorStatus
from eve_pipeline.core.config import LatexConfig


class LatexCorrectionPipeline:
    """Pipeline for correcting LaTeX formulas using GPT-4o-mini."""

    def __init__(
        self,
        config: Optional[LatexConfig] = None,
        num_processes: Optional[int] = None,
        debug: bool = False,
        storage_config: Optional[dict] = None,
    ) -> None:
        """Initialize LaTeX correction pipeline.

        Args:
            config: LaTeX correction configuration.
            num_processes: Number of processes for parallel processing.
            debug: Enable debug logging.
            storage_config: Storage configuration for S3/local file operations.
        """
        self.config = config or LatexConfig()
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.debug = debug
        self.storage_config = storage_config or {}

        # Set up logging
        import logging
        self.logger = logging.getLogger("eve_pipeline.LatexCorrectionPipeline")
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Initialize the LaTeX corrector processor
        self.latex_corrector = LatexCorrector(
            debug=debug,
            storage_config=self.storage_config,
        )

    def process_content(self, content: str, input_path: Optional[Path] = None) -> ProcessorResult:
        """Process content through LaTeX correction.

        Args:
            content: Input content to process.
            input_path: Optional input file path.

        Returns:
            ProcessorResult with corrected content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )

        return self.latex_corrector.process(content, input_path)

    def process_file(self, input_path: Union[str, Path]) -> ProcessorResult:
        """Process a single file.

        Args:
            input_path: Path to input file (local or S3).

        Returns:
            ProcessorResult with processing results.
        """
        from eve_pipeline.storage.factory import StorageFactory

        input_path_str = str(input_path)

        try:
            # Use storage factory to handle both local and S3 paths
            storage = StorageFactory.get_storage_for_path(input_path_str, **self.storage_config)
            content = storage.read_text(input_path_str)

            return self.process_content(content, input_path)

        except Exception as e:
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message=f"Failed to read file: {e}",
            )

    def process_files(
        self,
        input_paths: list[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, ProcessorResult]:
        """Process multiple files.

        Args:
            input_paths: List of input file paths.
            output_dir: Optional output directory for results.

        Returns:
            Dictionary mapping file paths to processing results.
        """
        self.logger.info(f"Processing {len(input_paths)} files for LaTeX correction")
        if self.debug:
            self.logger.debug(f"Files to process: {[str(p) for p in input_paths[:10]]}{'...' if len(input_paths) > 10 else ''}")

        results = {}

        if self.num_processes == 1:
            # Sequential processing
            for input_path in input_paths:
                result = self.process_file(input_path)
                results[str(input_path)] = result

                if output_dir and result.status == ProcessorStatus.SUCCESS:
                    self._save_result(result, output_dir, input_path)
        else:
            # Parallel processing
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                    future_to_path = {
                        executor.submit(self.process_file, path): path
                        for path in input_paths
                    }

                    for future in concurrent.futures.as_completed(future_to_path):
                        input_path = future_to_path[future]
                        try:
                            result = future.result()
                            results[str(input_path)] = result

                            if output_dir and result.status == ProcessorStatus.SUCCESS:
                                self._save_result(result, output_dir, input_path)

                        except Exception as e:
                            results[str(input_path)] = ProcessorResult(
                                status=ProcessorStatus.FAILED,
                                input_path=Path(input_path),
                                error_message=f"Processing failed: {e}",
                            )
            except (OSError, RuntimeError) as e:
                # Fallback to sequential processing on multiprocessing errors (common on macOS)
                self.logger.warning(f"ProcessPoolExecutor failed ({e}), falling back to sequential processing")
                for input_path in input_paths:
                    result = self.process_file(input_path)
                    results[str(input_path)] = result

                    if output_dir and result.status == ProcessorStatus.SUCCESS:
                        self._save_result(result, output_dir, input_path)

        return results

    def _save_result(
        self,
        result: ProcessorResult,
        output_dir: Union[str, Path],
        input_path: Union[str, Path],
    ) -> None:
        """Save processing result to output directory.

        Args:
            result: Processing result to save.
            output_dir: Output directory (local or S3).
            input_path: Original input file path.
        """
        from eve_pipeline.storage.base import StorageBase
        from eve_pipeline.storage.factory import StorageFactory

        output_dir_str = str(output_dir)
        input_path_str = str(input_path)

        storage = StorageFactory.get_storage_for_path(output_dir_str, **self.storage_config)

        if StorageBase.is_s3_path(input_path_str):
            filename = input_path_str.split('/')[-1]
        else:
            filename = Path(input_path_str).name

        output_path = f"{output_dir_str.rstrip('/')}/{filename}"

        try:
            storage.write_text(output_path, result.content or "")
        except Exception as e:
            if self.debug:
                self.logger.debug(f"Failed to save result to {output_path}: {e}")

    @property
    def enabled(self) -> bool:
        """Check if LaTeX correction is enabled."""
        return self.config.enabled