"""Text file extraction."""

from pathlib import Path
from typing import Any, Dict, Union

from eve_pipeline.extraction.base import ExtractorBase


class TextExtractor(ExtractorBase):
    """Plain text file extractor."""
    
    def __init__(
        self,
        preserve_formatting: bool = True,
        **kwargs,
    ) -> None:
        """Initialize text extractor.
        
        Args:
            preserve_formatting: Whether to preserve original formatting.
            **kwargs: Additional configuration.
        """
        super().__init__(
            supported_formats=["txt", "text", "md", "markdown"],
            output_format="markdown",
            **kwargs,
        )
        self.preserve_formatting = preserve_formatting
    
    def extract_content(self, file_path: Union[str, Path]) -> str:
        """Extract content from text file.
        
        Args:
            file_path: Path to text file (local or S3).
            
        Returns:
            Text content as markdown.
        """
        try:
            # Read file content
            content = self._read_file(file_path)
            
            if not content.strip():
                raise RuntimeError("Empty text file")
            
            # Get file extension safely for both S3 and local paths
            if isinstance(file_path, str) and file_path.startswith('s3://'):
                file_suffix = "." + file_path.split(".")[-1].lower() if "." in file_path else ""
            else:
                file_path_obj = Path(file_path) if isinstance(file_path, str) else file_path
                file_suffix = file_path_obj.suffix.lower()
            
            # Process based on file type
            if file_suffix in [".md", ".markdown"]:
                # Already markdown
                processed_content = content
            else:
                # Convert plain text to markdown
                processed_content = self._convert_to_markdown(content, file_path)
            
            return processed_content
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise
    
    def _convert_to_markdown(self, content: str, file_path: Union[str, Path]) -> str:
        """Convert plain text to markdown format.
        
        Args:
            content: Plain text content.
            file_path: Path to original file.
            
        Returns:
            Content formatted as markdown.
        """
        # Add header
        markdown_content = self._create_markdown_header(file_path)
        
        if self.preserve_formatting:
            # Wrap in code block to preserve formatting
            markdown_content += f"```\n{content}\n```"
        else:
            # Basic markdown formatting
            lines = content.split("\n")
            processed_lines = []
            
            for line in lines:
                line = line.rstrip()
                
                # Detect potential headers (all caps, standalone lines)
                if (line.isupper() and 
                    len(line.split()) > 1 and 
                    len(line) < 100 and
                    line.replace(" ", "").isalnum()):
                    processed_lines.append(f"## {line.title()}")
                # Detect list items
                elif line.strip().startswith(("-", "*", "•")):
                    processed_lines.append(line)
                # Regular content
                else:
                    processed_lines.append(line)
            
            markdown_content += "\n".join(processed_lines)
        
        return markdown_content
    
    def get_text_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get text file metadata.
        
        Args:
            file_path: Path to text file.
            
        Returns:
            Dictionary with metadata.
        """
        try:
            content = self._read_file(file_path)
            
            lines = content.split("\n")
            words = content.split()
            
            # For S3 files, we can't use stat(), so calculate file size from content
            try:
                file_size = file_path.stat().st_size if not str(file_path).startswith('s3://') else len(content.encode('utf-8'))
            except (OSError, AttributeError):
                file_size = len(content.encode('utf-8'))
            
            metadata = {
                "file_size": file_size,
                "content_length": len(content),
                "line_count": len(lines),
                "word_count": len(words),
                "non_empty_lines": len([line for line in lines if line.strip()]),
                "encoding": self._detect_encoding(file_path),
            }
            
            # Detect if it's already markdown
            if isinstance(file_path, str) and file_path.startswith('s3://'):
                file_suffix = "." + file_path.split(".")[-1].lower() if "." in file_path else ""
            else:
                file_path_obj = Path(file_path) if isinstance(file_path, str) else file_path
                file_suffix = file_path_obj.suffix.lower()
            
            if file_suffix in [".md", ".markdown"]:
                metadata["is_markdown"] = True
                metadata["header_count"] = len([line for line in lines if line.strip().startswith("#")])
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract text metadata: {e}")
            try:
                file_size = file_path.stat().st_size if not str(file_path).startswith('s3://') else 0
            except (OSError, AttributeError):
                file_size = 0
            return {"file_size": file_size}
    
    def _detect_encoding(self, file_path: Union[str, Path]) -> str:
        """Detect file encoding.
        
        Args:
            file_path: Path to file (local or S3).
            
        Returns:
            Detected encoding.
        """
        return "utf-8"