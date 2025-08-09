"""Base extractor classes and interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import mimetypes

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus


class ExtractorBase(ProcessorBase):
    """Base class for all extractors."""
    
    def __init__(
        self,
        supported_formats: Optional[List[str]] = None,
        output_format: str = "markdown",
        **kwargs,
    ) -> None:
        """Initialize extractor.
        
        Args:
            supported_formats: List of supported file extensions (without dots).
            output_format: Output format (markdown, text, etc.).
            **kwargs: Additional configuration.
        """
        super().__init__(**kwargs)
        self.supported_formats = supported_formats or []
        self.output_format = output_format
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """Check if extractor can handle the file.
        
        Args:
            file_path: Path to file (local or S3).
            
        Returns:
            True if extractor can handle the file.
        """
        # Handle S3 paths without converting to Path objects
        if isinstance(file_path, str) and file_path.startswith('s3://'):
            # Extract extension from S3 URL
            extension = file_path.split(".")[-1].lower() if "." in file_path else ""
        else:
            # Handle local paths
            file_path_obj = Path(file_path)
            extension = file_path_obj.suffix.lower().lstrip(".")
        
        return extension in self.supported_formats
    
    @abstractmethod
    def extract_content(self, file_path: Union[str, Path]) -> str:
        """Extract content from file.
        
        Args:
            file_path: Path to file (local or S3).
            
        Returns:
            Extracted content as string.
        """
        pass
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content (for extractors, this extracts from file path).
        
        Args:
            content: Ignored for extractors.
            input_path: Path to input file.
            **kwargs: Additional parameters.
            
        Returns:
            ProcessorResult with extracted content.
        """
        if not input_path:
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                error_message="Input path is required for extraction",
            )
        
        # Don't convert S3 paths to Path objects as it corrupts the URLs
        if isinstance(input_path, str) and input_path.startswith('s3://'):
            input_path_obj = input_path  # Keep as string for S3 paths
        else:
            input_path_obj = Path(input_path)
        
        try:
            # Check if we can handle this file
            if not self.can_extract(input_path_obj):
                # Get file extension safely for both S3 and local paths
                if isinstance(input_path_obj, str):
                    file_suffix = "." + input_path_obj.split(".")[-1] if "." in input_path_obj else ""
                else:
                    file_suffix = input_path_obj.suffix
                
                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    error_message=f"Unsupported format: {file_suffix}",
                )
            
            # Extract content
            extracted_content = self.extract_content(input_path_obj)
            
            if not extracted_content or not extracted_content.strip():
                return ProcessorResult(
                    status=ProcessorStatus.FAILED,
                    input_path=input_path,
                    error_message="No content extracted",
                )
            
            # Get file extension safely for both S3 and local paths
            if isinstance(input_path_obj, str):
                original_format = "." + input_path_obj.split(".")[-1].lower() if "." in input_path_obj else ""
            else:
                original_format = input_path_obj.suffix.lower()
            
            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=input_path,
                content=extracted_content,
                metadata={
                    "original_format": original_format,
                    "output_format": self.output_format,
                    "content_length": len(extracted_content),
                },
            )
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {input_path}: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message=str(e),
            )
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type of file.
        
        Args:
            file_path: Path to file.
            
        Returns:
            MIME type string.
        """
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    def _create_markdown_header(self, file_path: Path) -> str:
        """Create markdown header for extracted content.
        
        Args:
            file_path: Path to original file.
            
        Returns:
            Markdown header string.
        """
        return f"# Extracted from {file_path.name}\n\n"