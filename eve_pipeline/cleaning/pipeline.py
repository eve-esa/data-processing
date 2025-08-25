"""Cleaning pipeline orchestrator."""

import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Optional, Union

from eve_pipeline.cleaning.processors import (
    ArtifactRemover,
    NougatCorrector,
    OCRCorrector,
    OCRDeduplicator,
    RuleBasedCorrector,
)
from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus
from eve_pipeline.core.config import CleaningConfig


class CleaningPipeline:
    """Pipeline for cleaning markdown content using the 5-check process."""

    def __init__(
        self,
        config: Optional[CleaningConfig] = None,
        num_processes: Optional[int] = None,
        debug: bool = False,
        storage_config: Optional[dict] = None,
    ) -> None:
        """Initialize cleaning pipeline.

        Args:
            config: Cleaning configuration.
            num_processes: Number of processes for parallel processing.
            debug: Enable debug logging.
            storage_config: Storage configuration for processors.
        """
        self.config = config or CleaningConfig()
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.debug = debug
        self.storage_config = storage_config or {}

        # Initialize processors based on configuration
        self.processors = self._initialize_processors()

    def _initialize_processors(self) -> list[ProcessorBase]:
        """Initialize processors based on configuration."""
        processors = []

        if self.config.ocr_corrections:
            processors.append(OCRCorrector(debug=self.debug, storage_config=self.storage_config))

        if self.config.ocr_deduplication:
            processors.append(OCRDeduplicator(
                similarity_threshold=self.config.similarity_threshold,
                debug=self.debug,
                storage_config=self.storage_config,
            ))

        if self.config.nougat_correction:
            processors.append(NougatCorrector(debug=self.debug, storage_config=self.storage_config))

        if self.config.rule_based_corrections:
            processors.append(RuleBasedCorrector(debug=self.debug, storage_config=self.storage_config))

        if self.config.artifact_removal:
            processors.append(ArtifactRemover(debug=self.debug, storage_config=self.storage_config))

        return processors

    def process_content(self, content: str, input_path: Optional[Union[str, Path]] = None) -> ProcessorResult:
        """Process content through the cleaning pipeline.

        Args:
            content: Input content to clean.
            input_path: Optional input file path.

        Returns:
            ProcessorResult with cleaned content.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )

        current_content = content
        all_metadata = {}
        processing_steps = []

        for processor in self.processors:
            if not processor.enabled:
                continue

            try:
                result = processor.process(current_content, input_path)

                # Track processing step
                step_info = {
                    "processor": processor.name,
                    "status": result.status.value,
                    "processing_time": result.processing_time,
                }

                if result.metadata:
                    step_info["metadata"] = result.metadata

                if result.error_message:
                    step_info["error"] = result.error_message

                processing_steps.append(step_info)

                # Update metadata
                if result.metadata:
                    all_metadata[processor.name] = result.metadata

                # Check result status
                if result.is_failed:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        content=current_content,
                        error_message=f"Processing failed at {processor.name}: {result.error_message}",
                        metadata={"processing_steps": processing_steps, **all_metadata},
                    )
                elif result.is_success and result.content is not None:
                    current_content = result.content

            except Exception as e:
                self.logger.error(f"Cleaning processor {processor.name} failed: {e}")
                
                # Create helpful error message with context
                error_details = []
                error_details.append(f"Processor: {processor.name}")
                error_details.append(f"Content length: {len(current_content):,} characters")
                error_details.append(f"Completed processors: {len(processing_steps)}")
                
                # Add processor-specific hints
                error_str = str(e).lower()
                if "memory" in error_str:
                    error_details.append("Hint: Content may be too large for this processor")
                elif "regex" in error_str or "pattern" in error_str:
                    error_details.append("Hint: Content contains patterns that cause regex issues")
                elif "encoding" in error_str:
                    error_details.append("Hint: Content encoding issues detected")
                elif "openai" in error_str or "api" in error_str:
                    error_details.append("Hint: API connection or quota issues - check credentials")
                
                detailed_error = f"Cleaning failed in {processor.name}: {e!s}\n" + "\n".join(f"  • {detail}" for detail in error_details)
                
                return ProcessorResult(
                    status=ProcessorStatus.FAILED,
                    input_path=input_path,
                    content=current_content,
                    error_message=detailed_error,
                    metadata={"processing_steps": processing_steps, **all_metadata},
                )

        # Calculate overall statistics
        original_length = len(content)
        final_length = len(current_content)
        reduction_percent = (original_length - final_length) / original_length * 100 if original_length > 0 else 0

        return ProcessorResult(
            status=ProcessorStatus.SUCCESS,
            input_path=input_path,
            content=current_content,
            metadata={
                "processing_steps": processing_steps,
                "original_length": original_length,
                "final_length": final_length,
                "reduction_percent": reduction_percent,
                "processors_applied": len([s for s in processing_steps if s["status"] == "success"]),
                **all_metadata,
            },
        )

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> ProcessorResult:
        """Process a single file through the cleaning pipeline.

        Args:
            input_path: Path to input file.
            output_path: Optional path for output file.

        Returns:
            ProcessorResult with processing outcome.
        """
        # Don't convert S3 paths to Path objects as it corrupts the URLs
        if isinstance(input_path, str) and input_path.startswith('s3://'):
            input_path_obj = input_path
        else:
            input_path_obj = Path(input_path)

        try:
            # Read input file
            content = self._read_file(input_path_obj)

            # Process content
            result = self.process_content(content, input_path_obj)

            # Save output if provided and processing succeeded
            if output_path and result.is_success and result.content:
                # Don't convert S3 output paths to Path objects either
                if isinstance(output_path, str) and output_path.startswith('s3://'):
                    output_path_obj = output_path
                else:
                    output_path_obj = Path(output_path)
                self._write_file(output_path_obj, result.content)
                result.output_path = output_path_obj

            return result

        except Exception as e:
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path_obj,
                error_message=f"File processing failed: {e!s}",
            )

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        file_pattern: str = "*.md",
    ) -> dict[str, ProcessorResult]:
        """Process all files in a directory.

        Args:
            input_dir: Path to input directory.
            output_dir: Optional path to output directory.
            file_pattern: File pattern to match (default: "*.md").

        Returns:
            Dictionary mapping file paths to ProcessorResults.
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Find all matching files
        files = list(input_dir.rglob(file_pattern))

        if not files:
            return {}

        # Process files
        if self.num_processes > 1:
            return self._process_files_parallel(files, input_dir, output_dir)
        else:
            return self._process_files_sequential(files, input_dir, output_dir)

    def _process_files_sequential(
        self,
        files: list[Path],
        input_dir: Path,
        output_dir: Optional[Path],
    ) -> dict[str, ProcessorResult]:
        """Process files sequentially."""
        results = {}

        for file_path in files:
            # Create output path if output directory provided
            output_path = None
            if output_dir:
                relative_path = file_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

            result = self.process_file(file_path, output_path)
            results[str(file_path)] = result

        return results

    def _process_files_parallel(
        self,
        files: list[Path],
        input_dir: Path,
        output_dir: Optional[Path],
    ) -> dict[str, ProcessorResult]:
        """Process files in parallel using standardized parallel processor."""
        try:
            from eve_pipeline.core.parallel_processor import (
                ParallelProcessor,
                ProcessingConfig,
                ProcessingMode,
            )

            config = ProcessingConfig(
                max_workers=self.num_processes,
                mode=ProcessingMode.ADAPTIVE,
                enable_monitoring=True,
            )

            parallel_processor = ParallelProcessor(config)

            def process_single_file(file_path: Path) -> tuple[str, ProcessorResult]:
                # Create output path if output directory provided
                output_path = None
                if output_dir:
                    relative_path = file_path.relative_to(input_dir)
                    output_path = output_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                result = self.process_file(file_path, output_path)
                return str(file_path), result

            # Process files using standardized parallel processor
            file_results = parallel_processor.process_items(files, process_single_file)

            # Convert to expected format
            results = {}
            for i, file_path in enumerate(files):
                if i < len(file_results) and file_results[i] is not None:
                    file_key, result = file_results[i]
                    results[file_key] = result
                else:
                    results[str(file_path)] = ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=file_path,
                        error_message="Parallel processing returned no result",
                    )

            # Log performance report
            report = parallel_processor.get_performance_report()
            if 'error' not in report:
                self.logger.info(f"Parallel processing completed: {report['total_items_processed']} files in {report['total_processing_time']:.2f}s")

            return results

        except ImportError:
            self.logger.warning("Standardized parallel processor not available, falling back to ProcessPoolExecutor")
            return self._process_files_fallback(files, input_dir, output_dir)
        except Exception as e:
            self.logger.warning(f"Standardized parallel processing failed: {e}, falling back to ProcessPoolExecutor")
            return self._process_files_fallback(files, input_dir, output_dir)

    def _process_files_fallback(
        self,
        files: list[Path],
        input_dir: Path,
        output_dir: Optional[Path],
    ) -> dict[str, ProcessorResult]:
        """Fallback parallel processing using ProcessPoolExecutor."""
        results = {}

        def process_single_file(file_path: Path) -> tuple[str, ProcessorResult]:
            # Create output path if output directory provided
            output_path = None
            if output_dir:
                relative_path = file_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

            result = self.process_file(file_path, output_path)
            return str(file_path), result

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                future_to_file = {executor.submit(process_single_file, file_path): file_path
                                for file_path in files}

                for future in concurrent.futures.as_completed(future_to_file):
                    try:
                        file_key, result = future.result()
                        results[file_key] = result
                    except Exception as e:
                        file_path = future_to_file[future]
                        results[str(file_path)] = ProcessorResult(
                            status=ProcessorStatus.FAILED,
                            input_path=file_path,
                            error_message=f"Parallel processing failed: {e!s}",
                        )
        except (OSError, RuntimeError) as e:
            # Fallback to sequential processing on multiprocessing errors (common on macOS)
            self.logger.warning(f"ProcessPoolExecutor failed ({e}), falling back to sequential processing")
            return self._process_files_sequential(files, input_dir, output_dir)

        return results

    def _read_file(self, file_path: Path) -> str:
        """Read file with multiple encoding attempts."""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise Exception(f"Cannot decode file {file_path} with any supported encoding")

    def _write_file(self, file_path: Path, content: str) -> None:
        """Write content to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def get_summary_stats(self, results: dict[str, ProcessorResult]) -> dict[str, any]:
        """Generate summary statistics from processing results.

        Args:
            results: Dictionary of processing results.

        Returns:
            Summary statistics.
        """
        total_files = len(results)
        successful = sum(1 for r in results.values() if r.is_success)
        failed = sum(1 for r in results.values() if r.is_failed)

        total_original_length = sum(
            r.metadata.get("original_length", 0)
            for r in results.values()
            if r.metadata
        )

        total_final_length = sum(
            r.metadata.get("final_length", 0)
            for r in results.values()
            if r.metadata
        )

        overall_reduction = 0.0
        if total_original_length > 0:
            overall_reduction = (total_original_length - total_final_length) / total_original_length * 100

        return {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_files * 100 if total_files > 0 else 0,
            "total_original_length": total_original_length,
            "total_final_length": total_final_length,
            "overall_reduction_percent": overall_reduction,
        }
