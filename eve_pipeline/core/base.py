"""Base classes and interfaces for pipeline processors."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from eve_pipeline.core.enums import ProcessorStatus
from eve_pipeline.core.logging import LoggerManager
from eve_pipeline.core.utils import PathUtils
from eve_pipeline.storage.factory import StorageFactory


@dataclass
class ProcessorResult:
    """Result of a processor operation."""

    status: ProcessorStatus
    input_path: Optional[Union[str, Path]] = None
    output_path: Optional[Union[str, Path]] = None
    content: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    warnings: Optional[list[str]] = None

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

    def get_user_metadata(self) -> dict[str, Any]:
        """Get minimal metadata for user display (non-debugging)."""
        if not self.metadata:
            return {}
        
        user_meta = {}
        
        if "stages_completed" in self.metadata:
            user_meta["stages_completed"] = self.metadata["stages_completed"]
        
        if "duplicate_type" in self.metadata:
            user_meta["duplicate_type"] = self.metadata["duplicate_type"]
        
        for key in ["content_length", "original_format", "output_format"]:
            if key in self.metadata:
                user_meta[key] = self.metadata[key]
        
        if "processing_steps" in self.metadata:
            failed_stages = [step.get("stage") for step in self.metadata["processing_steps"] if step.get("status") == "FAILED"]
            if failed_stages:
                user_meta["failed_stages"] = failed_stages
        
        return user_meta

    def get_debug_metadata(self) -> dict[str, Any]:
        """Get full metadata for debugging purposes."""
        return self.metadata or {}


class ProcessorBase(ABC):
    """Base class for all pipeline processors."""

    def __init__(
        self,
        name: Optional[str] = None,
        enabled: bool = True,
        debug: bool = False,
        storage_config: Optional[dict[str, Any]] = None,
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

        self.logger = LoggerManager.get_logger(
            self.name,
            level="DEBUG" if debug else None,
        )

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
        raise NotImplementedError("Subclasses must implement process method")

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """Process a file synchronously.

        Args:
            input_path: Path to input file (local or S3).
            output_path: Optional path for output file (local or S3).
            **kwargs: Additional processing parameters.

        Returns:
            ProcessorResult with processing outcome.
        """
        return asyncio.run(self.process_file_async(input_path, output_path, **kwargs))

    async def process_file_async(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """Process a file asynchronously.

        Args:
            input_path: Path to input file (local or S3).
            output_path: Optional path for output file (local or S3).
            **kwargs: Additional processing parameters.

        Returns:
            ProcessorResult with processing outcome.
        """
        start_time = time.time()
        if isinstance(input_path, str) and input_path.startswith('s3://'):
            input_path_obj = input_path
        else:
            input_path_obj = Path(input_path) if not isinstance(input_path, Path) else input_path

        try:
            content = await self._read_file_async(input_path_obj)

            result = await asyncio.get_event_loop().run_in_executor(
                None, self.process, content, input_path_obj, **kwargs,
            )

            # Update timing
            result.processing_time = time.time() - start_time
            result.input_path = input_path_obj

            # Save output if path provided and processing succeeded
            if output_path and result.is_success and result.content:
                if isinstance(output_path, str) and output_path.startswith('s3://'):
                    output_path_obj = output_path
                else:
                    output_path_obj = Path(output_path) if not isinstance(output_path, Path) else output_path
                await self._write_file_async(output_path_obj, result.content)
                result.output_path = output_path_obj

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
        """Read file synchronously (legacy support).

        Args:
            file_path: Path to file (local or S3).

        Returns:
            File content as string.
        """
        return asyncio.run(self._read_file_async(file_path))

    async def _read_file_async(self, file_path: Union[str, Path]) -> str:
        """Read file asynchronously with multiple encoding attempts.

        Args:
            file_path: Path to file (local or S3).

        Returns:
            File content as string.

        Raises:
            Exception: If file cannot be read with any encoding.
        """
        if isinstance(file_path, str) and file_path.startswith('s3://'):
            file_path_str = file_path
        else:
            file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
            file_path_str = str(file_path)

        # Get appropriate storage backend
        storage = StorageFactory.get_storage_for_path(file_path_str, **self.storage_config)

        try:
            content = await asyncio.get_event_loop().run_in_executor(
                None, storage.read_text, file_path_str,
            )
            return content
        except Exception as e:
            raise Exception(f"Cannot read file {file_path}: {e!s}")

    def _write_file(self, file_path: Union[str, Path], content: str) -> None:
        """Write content to file synchronously (legacy support).

        Args:
            file_path: Output file path (local or S3).
            content: Content to write.
        """
        asyncio.run(self._write_file_async(file_path, content))

    async def _write_file_async(self, file_path: Union[str, Path], content: str) -> None:
        """Write content to file asynchronously.

        Args:
            file_path: Output file path (local or S3).
            content: Content to write.
        """
        if isinstance(file_path, str) and file_path.startswith('s3://'):
            file_path_str = file_path
        else:
            file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
            file_path_str = str(file_path)
            PathUtils.ensure_path_exists(file_path, is_file=True)

        # Get appropriate storage backend
        storage = StorageFactory.get_storage_for_path(file_path_str, **self.storage_config)

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, storage.write_text, file_path_str, content,
            )
        except Exception as e:
            raise Exception(f"Cannot write to file {file_path}: {e!s}")

    def should_skip(self, _input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> bool:
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
