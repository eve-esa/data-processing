"""Base classes and interfaces for pipeline processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import time
from enum import Enum

from eve_pipeline.storage.factory import StorageFactory
from eve_pipeline.storage.base import StorageBase


class ProcessorStatus(Enum):
    """Status of a processor operation."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class ProcessorResult:
    """Result of a processor operation."""
    
    status: ProcessorStatus
    input_path: Optional[Union[str, Path]] = None
    output_path: Optional[Union[str, Path]] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.warnings is None:
            self.warnings = []
    
    @property
    def is_success(self) -> bool:
        """Check if processing was successful."""
        return self.status == ProcessorStatus.SUCCESS
    
    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ProcessorStatus.FAILED
    
    @property
    def is_skipped(self) -> bool:
        """Check if processing was skipped."""
        return self.status == ProcessorStatus.SKIPPED


class ProcessorBase(ABC):
    """Base class for all pipeline processors."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        debug: bool = False,
        storage_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize processor.
        
        Args:
            name: Processor name. If None, uses class name.
            enabled: Whether processor is enabled.
            debug: Enable debug logging.
            storage_config: Configuration for storage backends.
            **kwargs: Additional configuration.
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.debug = debug
        self.config = kwargs
        self.storage_config = storage_config or {}
        
        # Set up logging
        self.logger = logging.getLogger(f"eve_pipeline.{self.name}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    @abstractmethod
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """Process content.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with processing outcome.
        """
        pass
    
    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """Process a file.
        
        Args:
            input_path: Path to input file (local or S3).
            output_path: Optional path for output file (local or S3).
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult with processing outcome.
        """
        start_time = time.time()
        input_path_str = str(input_path)
        
        try:
            # Read input file using appropriate storage backend
            content = self._read_file(input_path_str)
            
            # Process content
            result = self.process(content, input_path, **kwargs)
            
            # Update timing
            result.processing_time = time.time() - start_time
            result.input_path = input_path
            
            # Save output if path provided and processing succeeded
            if output_path and result.is_success and result.content:
                output_path_str = str(output_path)
                self._write_file(output_path_str, result.content)
                result.output_path = output_path
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing file {input_path}: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )
    
    def _read_file(self, file_path: Union[str, Path]) -> str:
        """Read file with multiple encoding attempts.
        
        Args:
            file_path: Path to file (local or S3).
            
        Returns:
            File content as string.
            
        Raises:
            Exception: If file cannot be read with any encoding.
        """
        file_path_str = str(file_path)
        
        # Get appropriate storage backend
        storage = StorageFactory.get_storage_for_path(file_path_str, **self.storage_config)
        
        try:
            return storage.read_text(file_path_str)
        except Exception as e:
            raise Exception(f"Cannot read file {file_path_str}: {str(e)}")
    
    def _write_file(self, file_path: Union[str, Path], content: str) -> None:
        """Write content to file.
        
        Args:
            file_path: Output file path (local or S3).
            content: Content to write.
        """
        file_path_str = str(file_path)
        
        # Get appropriate storage backend
        storage = StorageFactory.get_storage_for_path(file_path_str, **self.storage_config)
        
        try:
            storage.write_text(file_path_str, content)
        except Exception as e:
            raise Exception(f"Cannot write to file {file_path_str}: {str(e)}")
    
    def should_skip(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> bool:
        """Check if processing should be skipped.
        
        Args:
            input_path: Input file path (local or S3).
            output_path: Optional output file path (local or S3).
            
        Returns:
            True if processing should be skipped.
        """
        if not self.enabled:
            return True
        
        # Skip if output already exists (basic implementation)
        if output_path:
            output_path_str = str(output_path)
            storage = StorageFactory.get_storage_for_path(output_path_str, **self.storage_config)
            if storage.exists(output_path_str):
                return True
        
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"