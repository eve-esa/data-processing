"""LaTeX correction pipeline for mathematical formula validation and correction."""

import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Union

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus
from eve_pipeline.core.config import LatexConfig
from eve_pipeline.cleaning.processors import LatexCorrector


class LatexCorrectionPipeline:
    """Pipeline for correcting LaTeX formulas using GPT-4o-mini."""
    
    def __init__(
        self,
        config: Optional[LatexConfig] = None,
        num_processes: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """Initialize LaTeX correction pipeline.
        
        Args:
            config: LaTeX correction configuration.
            num_processes: Number of processes for parallel processing.
            debug: Enable debug logging.
        """
        self.config = config or LatexConfig()
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.debug = debug
        
        # Initialize the LaTeX corrector processor
        self.latex_corrector = LatexCorrector(debug=debug)
    
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
            input_path: Path to input file.
            
        Returns:
            ProcessorResult with processing results.
        """
        input_path = Path(input_path)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.process_content(content, input_path)
            
        except Exception as e:
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message=f"Failed to read file: {e}",
            )
    
    def process_files(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, ProcessorResult]:
        """Process multiple files.
        
        Args:
            input_paths: List of input file paths.
            output_dir: Optional output directory for results.
            
        Returns:
            Dictionary mapping file paths to processing results.
        """
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
            output_dir: Output directory.
            input_path: Original input file path.
        """
        output_dir = Path(output_dir)
        input_path = Path(input_path)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file path
        output_path = output_dir / input_path.name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.content or "")
        except Exception as e:
            if self.debug:
                print(f"Failed to save result to {output_path}: {e}")
    
    @property
    def enabled(self) -> bool:
        """Check if LaTeX correction is enabled."""
        return self.config.enabled