"""Comprehensive cleaning step that applies all data cleaning components."""

from pathlib import Path
from typing import List, Tuple

from eve.base_step import PipelineStep
from eve.steps.cleaning.ocr_corrections import OCRCorrections
from eve.steps.cleaning.ocr_duplicate_remover import OCRDuplicateRemover
from eve.steps.cleaning.nougat_correction import NougatCorrection
from eve.steps.cleaning.rule_based_corrections import RuleBasedCorrections
from eve.steps.cleaning.nougat_artifact_removal import NougatArtifactRemovalComponent
from eve.steps.cleaning.latex_correction import LatexCorrectionComponent


class CleaningStep(PipelineStep):
    """
    Comprehensive cleaning step that applies multiple data cleaning components.
    
    This step processes extracted text through various cleaning components to:
    - Fix OCR-induced errors
    - Remove OCR duplicates
    - Apply Nougat corrections
    - Apply rule-based corrections
    - Remove Nougat artifacts
    - Correct LaTeX syntax errors (optional)
    """

    def __init__(self, config: dict):
        """Initialize the cleaning step with configuration.
        
        Args:
            config: Configuration dictionary with component settings.
                   Expected keys:
                   - ocr_threshold: float (default 0.99) - OCR duplicate threshold
                   - min_words: int (default 2) - Minimum words for processing
                   - enable_latex_correction: bool (default False) - Enable LaTeX correction
                   - openrouter_api_key: str (optional) - API key for LaTeX correction
                   - openrouter_model: str (default "anthropic/claude-3-haiku") - Model for corrections
                   - debug: bool (default False) - Enable debug output
        """
        super().__init__(config, name="CleaningStep")
        
        self.debug = config.get("debug", False)
        ocr_threshold = config.get("ocr_threshold", 0.99)
        min_words = config.get("min_words", 2)
        enable_latex = config.get("enable_latex_correction", False)
        openrouter_key = config.get("openrouter_api_key")
        openrouter_model = config.get("openrouter_model", "anthropic/claude-3-haiku")
        
        self.components = [
            OCRCorrections(debug=self.debug),
            OCRDuplicateRemover(threshold=ocr_threshold, min_words=min_words, debug=self.debug),
            NougatCorrection(debug=self.debug),
            RuleBasedCorrections(debug=self.debug),
            NougatArtifactRemovalComponent(debug=self.debug),
        ]
        
        if enable_latex:
            self.components.append(
                LatexCorrectionComponent(
                    debug=self.debug,
                    api_key=openrouter_key,
                    model=openrouter_model
                )
            )

    async def execute(self, input_data: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
        """Execute the cleaning step on extracted text data.
        
        Args:
            input_data: List of tuples containing (file_path, extracted_text).
            
        Returns:
            List of tuples containing (file_path, cleaned_text).
        """
        self.logger.info(f"Executing cleaning step on {len(input_data)} files")
        
        if not input_data:
            self.logger.warning("No input data provided to cleaning step")
            return []
        
        result = []
        processed_count = 0
        failed_count = 0
        
        for file_path, content in input_data:
            filename = file_path.name
            
            if not content:
                self.logger.warning(f"{filename} - Empty content, skipping cleaning")
                result.append((file_path, content))
                failed_count += 1
                continue
            
            try:
                cleaned_content = content
                
                for component in self.components:
                    try:
                        cleaned_content = await component.process(cleaned_content, filename)
                        
                        if cleaned_content is None:
                            self.logger.error(f"{filename} - Component {component.__class__.__name__} returned None")
                            cleaned_content = content
                            break
                            
                    except Exception as e:
                        self.logger.error(f"{filename} - Component {component.__class__.__name__} failed: {str(e)}")
                        continue
                
                if content and cleaned_content:
                    original_length = len(content)
                    cleaned_length = len(cleaned_content)
                    reduction_percent = ((original_length - cleaned_length) / original_length) * 100
                    
                    if reduction_percent > 0:
                        self.logger.info(f"{filename} - Cleaned: {reduction_percent:.2f}% text removed ({original_length} -> {cleaned_length} chars)")
                    else:
                        self.logger.info(f"{filename} - Cleaned: No significant changes")
                
                result.append((file_path, cleaned_content))
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"{filename} - Cleaning failed: {str(e)}")
                result.append((file_path, content))
                failed_count += 1
        
        self.logger.info(f"Cleaning step completed: {processed_count} processed, {failed_count} failed")
        return result

    def _get_applicable_formats(self) -> List[str]:
        """Get list of formats that these cleaning components apply to.
        
        Returns:
            List of file formats that can be processed by cleaning components.
        """
        return [
            "md",
            "txt",
            "tex",
            "html",
            "xml",
        ]

    def get_component_info(self) -> dict:
        """Get information about enabled cleaning components.
        
        Returns:
            Dictionary with component information.
        """
        component_info = {
            "total_components": len(self.components),
            "components": [component.__class__.__name__ for component in self.components],
            "applicable_formats": self._get_applicable_formats(),
            "debug_enabled": self.debug
        }
        
        latex_enabled = any(isinstance(comp, LatexCorrectionComponent) for comp in self.components)
        component_info["latex_correction_enabled"] = latex_enabled
        
        return component_info
