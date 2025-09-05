"""Comprehensive cleaning step that applies all data cleaning components."""

from pathlib import Path
from typing import List, Union, Tuple

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.cleaning.processors import (
    OCRProcessor,
    DuplicateRemovalProcessor,
    NougatProcessor,
    RuleBasedProcessor,
    LaTeXProcessor,
)


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
        
        ocr_threshold = config.get("ocr_threshold", 0.99)
        min_words = config.get("min_words", 2)
        enable_latex = config.get("enable_latex_correction", False)
        openrouter_key = config.get("openrouter_api_key")
        openrouter_model = config.get("openrouter_model", "anthropic/claude-3-haiku")
        
        self.processors = [
            OCRProcessor(debug=self.debug),
            DuplicateRemovalProcessor(threshold=ocr_threshold, min_words=min_words, debug=self.debug),
            NougatProcessor(debug=self.debug),
            RuleBasedProcessor(debug=self.debug),
        ]
        
        if enable_latex:
            self.processors.append(
                LaTeXProcessor(
                    debug=self.debug,
                    api_key=openrouter_key,
                    model=openrouter_model
                )
            )

    async def execute(self, input_data: Union[List[Document], List[Tuple[Path, str]]]) -> List[Document]:
        """Execute the cleaning step on input data.
        
        Args:
            input_data: List of Documents or list of tuples containing (file_path, extracted_text).
            
        Returns:
            List of cleaned Documents.
        """
        # Convert tuple format to Document objects if needed
        documents = []
        if input_data and isinstance(input_data[0], tuple):
            documents = [Document.from_tuple(item) for item in input_data]
        else:
            documents = input_data
        
        self.logger.info(f"Executing cleaning step on {len(documents)} documents")
        
        if not documents:
            self.logger.warning("No input data provided to cleaning step")
            return []
        
        result = []
        processed_count = 0
        failed_count = 0
        
        for document in documents:
            if document.is_empty():
                self.logger.warning(f"{document.filename} - Empty content, skipping cleaning")
                result.append(document)
                failed_count += 1
                continue
            
            try:
                processed_document = document
                original_length = document.content_length
                
                for processor in self.processors:
                    try:
                        processed_document = await processor.process(processed_document)
                        
                        if processed_document is None:
                            self.logger.error(f"{document.filename} - Processor {processor.__class__.__name__} returned None")
                            processed_document = document
                            break
                            
                    except Exception as e:
                        self.logger.error(f"{document.filename} - Processor {processor.__class__.__name__} failed: {str(e)}")
                        continue
                
                if original_length > 0 and processed_document.content_length != original_length:
                    reduction_percent = ((original_length - processed_document.content_length) / original_length) * 100
                    
                    if reduction_percent > 0:
                        self.logger.info(f"{document.filename} - Cleaned: {reduction_percent:.2f}% text removed ({original_length} -> {processed_document.content_length} chars)")
                    else:
                        self.logger.info(f"{document.filename} - Cleaned: No significant changes")
                
                result.append(processed_document)
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"{document.filename} - Cleaning failed: {str(e)}")
                result.append(document)
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
        """Get information about enabled cleaning processors.
        
        Returns:
            Dictionary with processor information.
        """
        component_info = {
            "total_processors": len(self.processors),
            "processors": [processor.__class__.__name__ for processor in self.processors],
            "applicable_formats": self._get_applicable_formats(),
            "debug_enabled": self.debug
        }
        
        latex_enabled = any(isinstance(proc, LaTeXProcessor) for proc in self.processors)
        component_info["latex_correction_enabled"] = latex_enabled
        
        return component_info
