"""Main pipeline orchestrator for the complete data processing workflow."""

import asyncio
import logging
import time
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional, Union, Iterator

from tqdm import tqdm

from eve_pipeline.cleaning.pipeline import CleaningPipeline
from eve_pipeline.core.base import ProcessorResult, ProcessorStatus
from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.deduplication.pipeline import DeduplicationPipeline
from eve_pipeline.extraction.factory import ExtractorFactory
from eve_pipeline.latex_correction.pipeline import LatexCorrectionPipeline
from eve_pipeline.pii_removal.processor import PIIRemover
from eve_pipeline.storage.base import StorageBase
from eve_pipeline.storage.factory import StorageFactory
from eve_pipeline.storage.async_s3 import AsyncS3Storage


class Pipeline:
    """Complete data processing pipeline orchestrator."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig()

        # Set up logging
        self._setup_logging()

        # Get storage configuration
        self.storage_config = self.config.storage.to_storage_kwargs()
        
        # Add performance settings to storage config
        if self.config.performance.use_async_s3:
            self.storage_config["max_concurrent"] = self.config.performance.s3_max_concurrent

        # Initialize stage processors
        self.extractor_factory = None
        self.cleaning_pipeline = None
        self.pii_remover = None
        self.deduplication_pipeline = None

        # Initialize enabled stages
        self._initialize_stages()

    def update_stage_configuration(self) -> None:
        """Update stage initialization based on current configuration.

        This should be called after configuration changes to ensure stages
        are properly enabled/disabled.
        """
        enabled_stages = self.config.enabled_stages

        # Update extraction stage
        if "extraction" in enabled_stages:
            if not hasattr(self, 'extractor_factory') or self.extractor_factory is None:
                self.extractor_factory = ExtractorFactory()
                self.logger.info("Extraction stage initialized")
        else:
            if hasattr(self, 'extractor_factory') and self.extractor_factory is not None:
                self.extractor_factory = None
                self.logger.info("Extraction stage disabled")

        if "cleaning" in enabled_stages:
            if not hasattr(self, 'cleaning_pipeline') or self.cleaning_pipeline is None:
                if "latex_correction" not in enabled_stages:
                    self.config.cleaning.latex_correction = False
                self.cleaning_pipeline = CleaningPipeline(
                    config=self.config.cleaning,
                    num_processes=1,
                    debug=self.config.debug,
                    storage_config=self.storage_config,
                )
                self.logger.info("Cleaning stage initialized")
        else:
            if hasattr(self, 'cleaning_pipeline') and self.cleaning_pipeline is not None:
                self.cleaning_pipeline = None
                self.logger.info("Cleaning stage disabled")

        if "pii_removal" in enabled_stages:
            if not hasattr(self, 'pii_remover') or self.pii_remover is None:
                self.pii_remover = PIIRemover(
                    entities=self.config.pii_removal.entities,
                    score_threshold=self.config.pii_removal.score_threshold,
                    use_presidio=self.config.pii_removal.use_presidio,
                    use_flair=self.config.pii_removal.use_flair,
                    server_url=self.config.pii_removal.server_url,
                    debug=self.config.debug,
                    storage_config=self.storage_config,
                )
                self.logger.info("PII removal stage initialized")
        else:
            if hasattr(self, 'pii_remover') and self.pii_remover is not None:
                self.pii_remover = None
                self.logger.info("PII removal stage disabled")

        if "deduplication" in enabled_stages:
            if not hasattr(self, 'deduplication_pipeline') or self.deduplication_pipeline is None:
                self.deduplication_pipeline = DeduplicationPipeline(
                    config=self.config.deduplication,
                    debug=self.config.debug,
                    storage_config=self.storage_config,
                )
                self.logger.info("Deduplication stage initialized")
        else:
            if hasattr(self, 'deduplication_pipeline') and self.deduplication_pipeline is not None:
                self.deduplication_pipeline = None
                self.logger.info("Deduplication stage disabled")

        if "latex_correction" in enabled_stages:
            if not hasattr(self, 'latex_correction_pipeline') or self.latex_correction_pipeline is None:
                self.latex_correction_pipeline = LatexCorrectionPipeline(
                    config=self.config.latex_correction,
                    debug=self.config.debug,
                    storage_config=self.storage_config,
                )
                self.logger.info("LaTeX correction stage initialized")
        else:
            if hasattr(self, 'latex_correction_pipeline') and self.latex_correction_pipeline is not None:
                self.latex_correction_pipeline = None
                self.logger.info("LaTeX correction stage disabled")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = self.config.log_level.value if hasattr(self.config.log_level, 'value') else self.config.log_level
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger("eve_pipeline.Pipeline")

    def _initialize_stages(self) -> None:
        """Initialize enabled pipeline stages."""
        enabled_stages = self.config.enabled_stages

        if "extraction" in enabled_stages:
            self.extractor_factory = ExtractorFactory()
            self.logger.info("Extraction stage initialized")

        if "cleaning" in enabled_stages:
            self.cleaning_pipeline = CleaningPipeline(
                config=self.config.cleaning,
                num_processes=1,  # Will handle multiprocessing at pipeline level
                debug=self.config.debug,
                storage_config=self.storage_config,
            )
            self.logger.info("Cleaning stage initialized")

        if "pii_removal" in enabled_stages:
            self.pii_remover = PIIRemover(
                entities=self.config.pii_removal.entities,
                score_threshold=self.config.pii_removal.score_threshold,
                use_presidio=self.config.pii_removal.use_presidio,
                use_flair=self.config.pii_removal.use_flair,
                server_url=self.config.pii_removal.server_url,
                debug=self.config.debug,
                storage_config=self.storage_config,
            )
            self.logger.info("PII removal stage initialized")

        if "deduplication" in enabled_stages:
            self.deduplication_pipeline = DeduplicationPipeline(
                config=self.config.deduplication,
                debug=self.config.debug,
                storage_config=self.storage_config,
            )
            self.logger.info("Deduplication stage initialized")

        if "latex_correction" in enabled_stages:
            self.latex_correction_pipeline = LatexCorrectionPipeline(
                config=self.config.latex_correction,
                debug=self.config.debug,
                storage_config=self.storage_config,
            )
            self.logger.info("LaTeX correction stage initialized")
        else:
            self.latex_correction_pipeline = None

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> ProcessorResult:
        """Process a single file through the entire pipeline.

        Args:
            input_path: Path to input file.
            output_path: Optional path for final output.

        Returns:
            ProcessorResult with final processing outcome.
        """
        input_path_str = str(input_path)
        start_time = time.time()

        # Check if input exists using appropriate storage backend
        input_storage = StorageFactory.get_storage_for_path(input_path_str, **self.storage_config)
        if not input_storage.exists(input_path_str):
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message=f"Input file does not exist: {input_path}",
            )

        current_content = None
        all_metadata = {}
        processing_steps = []

        try:
            # Stage 1: Extraction
            if self.extractor_factory and self.config.extraction.enabled:
                step_start = time.time()

                extractor = self.extractor_factory.get_extractor(input_path_str)
                if not extractor:
                    # Get file extension and supported formats for helpful error message
                    file_ext = Path(input_path_str).suffix.lower() if hasattr(Path(input_path_str), 'suffix') else ""
                    supported_formats = self.extractor_factory.get_supported_formats()
                    
                    error_msg = f"Unsupported file type '{file_ext}' for file: {input_path_str}"
                    error_msg += f"\nSupported formats: {', '.join(sorted(supported_formats))}"
                    error_msg += f"\nHint: Ensure the file has a supported extension and is not corrupted"
                    
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        error_message=error_msg,
                    )

                extraction_result = extractor.process_file(input_path_str)

                step_info = {
                    "stage": "extraction",
                    "status": extraction_result.status.value,
                    "processing_time": time.time() - step_start,
                    "metadata": extraction_result.metadata or {},
                }

                if extraction_result.error_message:
                    step_info["error"] = extraction_result.error_message

                processing_steps.append(step_info)

                if extraction_result.is_failed:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        processing_time=time.time() - start_time,
                        error_message=f"Extraction failed: {extraction_result.error_message}",
                        metadata={"processing_steps": processing_steps},
                    )

                current_content = extraction_result.content
                all_metadata["extraction"] = extraction_result.metadata or {}
            else:
                # Read file content directly if extraction disabled
                current_content = self._read_file(input_path_str)

            # Stage 2: Cleaning
            if self.cleaning_pipeline and self.config.cleaning.enabled:
                step_start = time.time()

                cleaning_result = self.cleaning_pipeline.process_content(current_content, input_path)

                step_info = {
                    "stage": "cleaning",
                    "status": cleaning_result.status.value,
                    "processing_time": time.time() - step_start,
                    "metadata": cleaning_result.metadata or {},
                }

                if cleaning_result.error_message:
                    step_info["error"] = cleaning_result.error_message

                processing_steps.append(step_info)

                if cleaning_result.is_failed:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        processing_time=time.time() - start_time,
                        error_message=f"Cleaning failed: {cleaning_result.error_message}",
                        metadata={"processing_steps": processing_steps, **all_metadata},
                    )

                if cleaning_result.content:
                    current_content = cleaning_result.content
                all_metadata["cleaning"] = cleaning_result.metadata or {}

            # Stage 3: PII Removal
            if self.pii_remover and self.config.pii_removal.enabled:
                step_start = time.time()

                pii_result = self.pii_remover.process(current_content, input_path)

                step_info = {
                    "stage": "pii_removal",
                    "status": pii_result.status.value,
                    "processing_time": time.time() - step_start,
                    "metadata": pii_result.metadata or {},
                }

                if pii_result.error_message:
                    step_info["error"] = pii_result.error_message

                processing_steps.append(step_info)

                if pii_result.is_failed:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        processing_time=time.time() - start_time,
                        error_message=f"PII removal failed: {pii_result.error_message}",
                        metadata={"processing_steps": processing_steps, **all_metadata},
                    )

                if pii_result.content:
                    current_content = pii_result.content
                all_metadata["pii_removal"] = pii_result.metadata or {}

            # Stage 4: Deduplication (check only, don't modify content)
            if self.deduplication_pipeline and self.config.deduplication.enabled:
                step_start = time.time()

                dedup_result = self.deduplication_pipeline.process_content(current_content, input_path)

                step_info = {
                    "stage": "deduplication",
                    "status": dedup_result.status.value,
                    "processing_time": time.time() - step_start,
                    "metadata": dedup_result.metadata or {},
                }

                if dedup_result.error_message:
                    step_info["error"] = dedup_result.error_message

                processing_steps.append(step_info)

                if dedup_result.is_skipped:
                    # File is duplicate, return skipped status
                    return ProcessorResult(
                        status=ProcessorStatus.SKIPPED,
                        input_path=input_path,
                        content=current_content,
                        processing_time=time.time() - start_time,
                        metadata={"processing_steps": processing_steps, **all_metadata},
                    )
                elif dedup_result.is_failed:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        processing_time=time.time() - start_time,
                        error_message=f"Deduplication failed: {dedup_result.error_message}",
                        metadata={"processing_steps": processing_steps, **all_metadata},
                    )

                all_metadata["deduplication"] = dedup_result.metadata or {}

            # Stage 5: LaTeX Correction
            if self.latex_correction_pipeline and self.config.latex_correction.enabled:
                step_start = time.time()

                latex_result = self.latex_correction_pipeline.process_content(current_content, input_path)

                step_info = {
                    "stage": "latex_correction",
                    "status": latex_result.status.value,
                    "processing_time": time.time() - step_start,
                    "metadata": latex_result.metadata or {},
                }

                if latex_result.error_message:
                    step_info["error"] = latex_result.error_message

                processing_steps.append(step_info)

                if latex_result.is_failed:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        processing_time=time.time() - start_time,
                        error_message=f"LaTeX correction failed: {latex_result.error_message}",
                        metadata={"processing_steps": processing_steps, **all_metadata},
                    )

                current_content = latex_result.content or current_content
                all_metadata["latex_correction"] = latex_result.metadata or {}

            # Save final output if path provided
            if output_path and current_content:
                output_path_str = str(output_path)
                self._write_file(output_path_str, current_content)

            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                output_path=output_path,
                content=current_content,
                processing_time=time.time() - start_time,
                metadata={
                    "processing_steps": processing_steps,
                    "stages_completed": len(processing_steps),
                    **all_metadata,
                },
            )

        except Exception as e:
            self.logger.error(f"Pipeline processing failed for {input_path}: {e}")
            
            # Create a more helpful error message
            error_context = []
            if processing_steps:
                last_step = processing_steps[-1]
                error_context.append(f"Failed during '{last_step.get('stage', 'unknown')}' stage")
            
            # Include the file path and stage information
            error_context.append(f"File: {input_path}")
            error_context.append(f"Stages completed: {len(processing_steps)}")
            
            # Add troubleshooting hints based on error type
            error_str = str(e).lower()
            if "memory" in error_str or "out of memory" in error_str:
                error_context.append("Hint: Try reducing batch size or processing smaller files")
            elif "permission" in error_str or "access" in error_str:
                error_context.append("Hint: Check file permissions and access rights")
            elif "timeout" in error_str:
                error_context.append("Hint: File may be too large or complex, try increasing timeout")
            elif "import" in error_str or "module" in error_str:
                error_context.append("Hint: Missing dependencies - check installation requirements")
            
            detailed_error = f"Pipeline processing failed: {e!s}\n" + "\n".join(f"  • {ctx}" for ctx in error_context)
            
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                processing_time=time.time() - start_time,
                error_message=detailed_error,
                metadata={"processing_steps": processing_steps, **all_metadata},
            )

        finally:
            # Clean up resources to prevent memory leaks
            self._cleanup_resources()

    async def _discover_files_async(
        self,
        input_dir: str,
        file_patterns: list[str],
    ) -> list[str]:
        """Discover files using async operations for better performance.

        Args:
            input_dir: Path to input directory (local or S3).
            file_patterns: List of file patterns to process.

        Returns:
            List of discovered file paths.
        """
        input_storage = StorageFactory.get_async_storage_for_path(input_dir, **self.storage_config)

        # Use async S3 storage for concurrent pattern matching if available and enabled
        if isinstance(input_storage, AsyncS3Storage) and self.config.performance.use_async_s3:
            async with input_storage:
                files = await input_storage.list_files_batch_async(input_dir, file_patterns)
                self.logger.info(f"Found {len(files)} files using async S3 pattern matching")
                return files
        else:
            # Fallback to sync for local storage, but still run concurrently
            loop = asyncio.get_event_loop()
            
            async def list_pattern_async(pattern: str) -> list[str]:
                return await loop.run_in_executor(None, input_storage.list_files, input_dir, pattern)

            tasks = [list_pattern_async(pattern) for pattern in file_patterns]
            pattern_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results and remove duplicates
            all_files = []
            for i, result in enumerate(pattern_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Pattern '{file_patterns[i]}' matching failed: {result}")
                    continue
                if isinstance(result, list):
                    all_files.extend(result)

            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for file_path in all_files:
                if file_path not in seen:
                    seen.add(file_path)
                    unique_files.append(file_path)

            return unique_files

    def _discover_files_sync(
        self,
        input_dir: str,
        file_patterns: list[str],
    ) -> list[str]:
        """Synchronous file discovery with concurrent pattern matching where possible."""
        input_storage = StorageFactory.get_storage_for_path(input_dir, **self.storage_config)

        # Use concurrent pattern matching if supported and enabled
        if hasattr(input_storage, 'list_files_batch') and self.config.performance.concurrent_pattern_matching:
            self.logger.info(f"Searching for files in {input_dir} with patterns: {file_patterns} (concurrent)")
            files = input_storage.list_files_batch(input_dir, file_patterns)
            self.logger.info(f"Found {len(files)} files using concurrent pattern matching")
            return files
        else:
            # Fallback to sequential pattern matching
            self.logger.info(f"Searching for files in {input_dir} with patterns: {file_patterns} (sequential)")
            files = []
            for pattern in file_patterns:
                pattern_files = input_storage.list_files(input_dir, pattern)
                files.extend(pattern_files)
                self.logger.debug(f"Found {len(pattern_files)} files matching pattern '{pattern}'")

            files = list(set(files))  # Remove duplicates
            return files

    def _process_files_streaming(
        self,
        files: list[str],
        input_dir: str,
        output_dir: Optional[Union[str, Path]],
        batch_size: int = 50,
    ) -> Iterator[ProcessorResult]:
        """Process files in streaming batches to reduce memory usage.

        Args:
            files: List of file paths to process.
            input_dir: Input directory path.
            output_dir: Optional output directory path.
            batch_size: Number of files to process in each batch.

        Yields:
            ProcessorResult for each processed file.
        """
        total_files = len(files)
        
        for i in range(0, total_files, batch_size):
            batch_files = files[i:i + batch_size]
            batch_end = min(i + batch_size, total_files)
            
            self.logger.info(f"Processing batch {i//batch_size + 1}: files {i+1}-{batch_end} of {total_files}")

            if self.config.num_processes > 1:
                # Process batch in parallel
                results = self._process_files_parallel(batch_files, input_dir, output_dir)
                for result in results:
                    yield result
            else:
                # Process batch sequentially
                for file_path in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}"):
                    output_path = self._generate_output_path(file_path, input_dir, output_dir)
                    result = self.process_file(file_path, output_path)
                    yield result

            # Force garbage collection between batches
            import gc
            gc.collect()

    def _generate_output_path(
        self,
        file_path: str,
        input_dir: str,
        output_dir: Optional[Union[str, Path]],
    ) -> Optional[str]:
        """Generate output path for a file."""
        if not output_dir:
            return None

        # Create output path if output directory provided
        if StorageBase.is_s3_path(file_path) and StorageBase.is_s3_path(input_dir):
            # S3 to S3 - use key-based relative path
            input_key = file_path.replace(input_dir, "").lstrip("/")
            if "." in input_key:
                base_name = ".".join(input_key.split(".")[:-1])
                output_path = f"{str(output_dir).rstrip('/')}/{base_name}.md"
            else:
                output_path = f"{str(output_dir).rstrip('/')}/{input_key}.md"
        elif not StorageBase.is_s3_path(file_path) and not StorageBase.is_s3_path(str(output_dir)):
            relative_path = Path(file_path).relative_to(Path(input_dir))
            output_path = Path(output_dir) / relative_path.with_suffix('.md')
        else:
            if isinstance(file_path, str) and file_path.startswith('s3://'):
                filename = file_path.split('/')[-1]
            else:
                filename = Path(file_path).name
            if "." in filename:
                base_name = ".".join(filename.split(".")[:-1])
                output_path = f"{str(output_dir).rstrip('/')}/{base_name}.md"
            else:
                output_path = f"{str(output_dir).rstrip('/')}/{filename}.md"

        return str(output_path)

    def _cleanup_resources(self) -> None:
        """Clean up pipeline resources to prevent memory leaks."""
        try:
            # Clean up PII remover if it exists
            if hasattr(self, 'pii_remover') and self.pii_remover and hasattr(self.pii_remover, 'cleanup'):
                self.pii_remover.cleanup()

            # Clean up cleaning pipeline if it exists
            if hasattr(self, 'cleaning_pipeline') and self.cleaning_pipeline and hasattr(self.cleaning_pipeline, 'cleanup'):
                self.cleaning_pipeline.cleanup()

            # Force garbage collection
            import gc
            gc.collect()

        except Exception as e:
            self.logger.warning(f"Error during pipeline cleanup: {e}")

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        file_patterns: Optional[list[str]] = None,
    ) -> dict[str, any]:
        """Process all files in a directory through the pipeline.

        Args:
            input_dir: Path to input directory (local or S3).
            output_dir: Optional path to output directory (local or S3).
            file_patterns: List of file patterns to process.

        Returns:
            Dictionary with processing results and statistics.
        """
        input_dir_str = str(input_dir)
        input_storage = StorageFactory.get_storage_for_path(input_dir_str, **self.storage_config)

        if not input_storage.exists(input_dir_str):
            return {
                "success": False,
                "error_message": f"Input directory does not exist: {input_dir}",
            }

        if output_dir:
            output_dir_str = str(output_dir)
            StorageFactory.get_storage_for_path(output_dir_str, **self.storage_config)

        # Default file patterns based on enabled stages
        if file_patterns is None:
            if self.config.extraction.enabled:
                file_patterns = [f"*.{fmt}" for fmt in self.config.extraction.supported_formats]
            else:
                file_patterns = ["*.md"]  # Process only markdown if no extraction

        # Discover files using async operations for better performance if enabled
        if self.config.performance.use_async_s3 and StorageBase.is_s3_path(input_dir_str):
            try:
                files = asyncio.run(self._discover_files_async(input_dir_str, file_patterns))
            except Exception as e:
                self.logger.warning(f"Async file discovery failed, falling back to sync: {e}")
                files = self._discover_files_sync(input_dir_str, file_patterns)
        else:
            files = self._discover_files_sync(input_dir_str, file_patterns)

        if not files:
            self.logger.warning(f"No files found matching patterns {file_patterns} in {input_dir}")
            return {
                "success": True,
                "message": f"No files found matching patterns {file_patterns}",
                "total_files": 0,
                "results": [],
            }

        self.logger.info(f"Found {len(files)} files to process from {input_dir}")
        if self.config.debug:
            self.logger.debug(f"Files to process: {files[:10]}{'...' if len(files) > 10 else ''}")

        # Log enabled stages
        enabled_stages = self.config.enabled_stages
        self.logger.info(f"Pipeline stages enabled: {enabled_stages}")

        # Process files using streaming for memory efficiency
        # Calculate batch size based on configuration or auto-calculate
        if self.config.performance.streaming_batch_size is not None:
            stream_batch_size = self.config.performance.streaming_batch_size
        else:
            # Auto-calculate based on extraction batch size and system resources
            stream_batch_size = max(50, self.config.extraction.batch_size * 4)
        
        results = []
        start_time = time.time()
        
        for result in self._process_files_streaming(files, input_dir_str, output_dir, stream_batch_size):
            results.append(result)
            
            # Log progress periodically
            if len(results) % 100 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                self.logger.info(f"Processed {len(results)}/{len(files)} files ({rate:.1f} files/sec)")

        # Generate statistics
        stats = self._generate_statistics(results)

        return {
            "success": True,
            "input_directory": input_dir_str,
            "output_directory": str(output_dir) if output_dir else None,
            "file_patterns": file_patterns,
            "total_files": len(files),
            "results": results,
            **stats,
        }

    def _process_files_sequential(
        self,
        files: list[str],
        input_dir: str,
        output_dir: Optional[Union[str, Path]],
    ) -> list[ProcessorResult]:
        """Process files sequentially."""
        results = []

        for file_path in tqdm(files, desc="Processing files"):
            output_path = self._generate_output_path(file_path, input_dir, output_dir)
            result = self.process_file(file_path, output_path)
            results.append(result)

        return results

    def _process_files_parallel(
        self,
        files: list[str],
        input_dir: str,
        output_dir: Optional[Union[str, Path]],
    ) -> list[ProcessorResult]:
        """Process files in parallel using thread-based parallelism to avoid pickle issues."""
        results = []

        from concurrent.futures import ThreadPoolExecutor

        def process_single_file(file_path: str) -> ProcessorResult:
            output_path = self._generate_output_path(file_path, input_dir, output_dir)
            return self.process_file(file_path, output_path)

        with ThreadPoolExecutor(max_workers=self.config.num_processes) as executor:
            future_to_file = {executor.submit(process_single_file, file_path): file_path
                            for file_path in files}

            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Processing files"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    results.append(ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=file_path,
                        error_message=f"Parallel processing failed: {e!s}",
                    ))

        return results

    def _generate_statistics(self, results: list[ProcessorResult]) -> dict[str, any]:
        """Generate processing statistics."""
        total_files = len(results)
        successful = sum(1 for r in results if r.is_success)
        failed = sum(1 for r in results if r.is_failed)
        skipped = sum(1 for r in results if r.is_skipped)

        total_processing_time = sum(r.processing_time for r in results)

        # Stage-specific statistics
        stage_stats = {}
        for stage in self.config.enabled_stages:
            stage_stats[stage] = {
                "enabled": True,
                "processed": sum(1 for r in results
                               if r.metadata and
                               stage in r.metadata and
                               r.metadata[stage]),
            }

        return {
            "statistics": {
                "total_files": total_files,
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "success_rate": successful / total_files * 100 if total_files > 0 else 0,
                "total_processing_time": total_processing_time,
                "average_processing_time": total_processing_time / total_files if total_files > 0 else 0,
                "enabled_stages": self.config.enabled_stages,
                "stage_statistics": stage_stats,
            },
        }

    def _read_file(self, file_path: str) -> str:
        """Read file with encoding detection."""
        storage = StorageFactory.get_storage_for_path(file_path, **self.storage_config)
        return storage.read_text(file_path)

    def _write_file(self, file_path: str, content: str) -> None:
        """Write content to file."""
        storage = StorageFactory.get_storage_for_path(file_path, **self.storage_config)
        storage.write_text(file_path, content)
