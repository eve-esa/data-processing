"""Main pipeline orchestrator for the complete data processing workflow."""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from eve_pipeline.core.base import ProcessorResult, ProcessorStatus
from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.extraction.factory import ExtractorFactory
from eve_pipeline.cleaning.pipeline import CleaningPipeline
from eve_pipeline.pii_removal.processor import PIIRemover
from eve_pipeline.deduplication.pipeline import DeduplicationPipeline
from eve_pipeline.latex_correction.pipeline import LatexCorrectionPipeline


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
        
        # Initialize stage processors
        self.extractor_factory = None
        self.cleaning_pipeline = None
        self.pii_remover = None
        self.deduplication_pipeline = None
        
        # Initialize enabled stages
        self._initialize_stages()
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
            )
            self.logger.info("PII removal stage initialized")
        
        if "deduplication" in enabled_stages:
            self.deduplication_pipeline = DeduplicationPipeline(
                config=self.config.deduplication,
                debug=self.config.debug,
            )
            self.logger.info("Deduplication stage initialized")
        
        if "latex_correction" in enabled_stages:
            self.latex_correction_pipeline = LatexCorrectionPipeline(
                config=self.config.latex_correction,
                debug=self.config.debug,
            )
            self.logger.info("LaTeX correction stage initialized")
    
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
        input_path = Path(input_path)
        start_time = time.time()
        
        if not input_path.exists():
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
                
                extractor = self.extractor_factory.get_extractor(input_path)
                if not extractor:
                    return ProcessorResult(
                        status=ProcessorStatus.FAILED,
                        input_path=input_path,
                        error_message=f"No extractor available for file type: {input_path.suffix}",
                    )
                
                extraction_result = extractor.process_file(input_path)
                
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
                current_content = self._read_file(input_path)
            
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
                output_path = Path(output_path)
                self._write_file(output_path, current_content)
            
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
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                processing_time=time.time() - start_time,
                error_message=f"Pipeline processing failed: {str(e)}",
                metadata={"processing_steps": processing_steps, **all_metadata},
            )
        
        finally:
            # Clean up resources to prevent memory leaks
            self._cleanup_resources()
    
    def _cleanup_resources(self) -> None:
        """Clean up pipeline resources to prevent memory leaks."""
        try:
            # Clean up PII remover if it exists
            if hasattr(self, 'pii_remover') and self.pii_remover:
                if hasattr(self.pii_remover, 'cleanup'):
                    self.pii_remover.cleanup()
            
            # Clean up cleaning pipeline if it exists  
            if hasattr(self, 'cleaning_pipeline') and self.cleaning_pipeline:
                if hasattr(self.cleaning_pipeline, 'cleanup'):
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
        file_patterns: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Process all files in a directory through the pipeline.
        
        Args:
            input_dir: Path to input directory.
            output_dir: Optional path to output directory.
            file_patterns: List of file patterns to process.
            
        Returns:
            Dictionary with processing results and statistics.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            return {
                "success": False,
                "error_message": f"Input directory does not exist: {input_dir}",
            }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default file patterns based on enabled stages
        if file_patterns is None:
            if self.config.extraction.enabled:
                file_patterns = [f"*.{fmt}" for fmt in self.config.extraction.supported_formats]
            else:
                file_patterns = ["*.md"]  # Assume markdown if no extraction
        
        # Find all matching files
        files = []
        for pattern in file_patterns:
            files.extend(input_dir.rglob(pattern))
        
        files = list(set(files))  # Remove duplicates
        
        if not files:
            return {
                "success": True,
                "message": f"No files found matching patterns {file_patterns}",
                "total_files": 0,
                "results": [],
            }
        
        self.logger.info(f"Processing {len(files)} files from {input_dir}")
        
        # Process files
        if self.config.num_processes > 1:
            results = self._process_files_parallel(files, input_dir, output_dir)
        else:
            results = self._process_files_sequential(files, input_dir, output_dir)
        
        # Generate statistics
        stats = self._generate_statistics(results)
        
        return {
            "success": True,
            "input_directory": str(input_dir),
            "output_directory": str(output_dir) if output_dir else None,
            "file_patterns": file_patterns,
            "total_files": len(files),
            "results": results,
            **stats,
        }
    
    def _process_files_sequential(
        self,
        files: List[Path],
        input_dir: Path,
        output_dir: Optional[Path],
    ) -> List[ProcessorResult]:
        """Process files sequentially."""
        results = []
        
        for file_path in tqdm(files, desc="Processing files"):
            # Create output path if output directory provided
            output_path = None
            if output_dir:
                relative_path = file_path.relative_to(input_dir)
                output_path = output_dir / relative_path.with_suffix('.md')
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result = self.process_file(file_path, output_path)
            results.append(result)
        
        return results
    
    def _process_files_parallel(
        self,
        files: List[Path],
        input_dir: Path,
        output_dir: Optional[Path],
    ) -> List[ProcessorResult]:
        """Process files in parallel."""
        results = []
        
        def process_single_file(file_path: Path) -> ProcessorResult:
            # Create output path if output directory provided
            output_path = None
            if output_dir:
                relative_path = file_path.relative_to(input_dir)
                output_path = output_dir / relative_path.with_suffix('.md')
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            return self.process_file(file_path, output_path)
        
        with ProcessPoolExecutor(max_workers=self.config.num_processes) as executor:
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
                        error_message=f"Parallel processing failed: {str(e)}",
                    ))
        
        return results
    
    def _generate_statistics(self, results: List[ProcessorResult]) -> Dict[str, any]:
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
            }
        }
    
    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise Exception(f"Cannot decode file {file_path} with any supported encoding")
    
    def _write_file(self, file_path: Path, content: str) -> None:
        """Write content to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)