"""Tests for core base classes."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus


class MockProcessor(ProcessorBase):
    """Mock processor for testing."""

    def process(self, content: str, input_path=None, **_kwargs) -> ProcessorResult:
        """Mock process implementation."""
        if not content:
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content",
            )

        return ProcessorResult(
            status=ProcessorStatus.SUCCESS,
            input_path=input_path,
            content=content.upper(),
            metadata={"processed": True},
        )


class TestProcessorBase:
    """Test cases for ProcessorBase."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = MockProcessor(name="TestProcessor", debug=True)

        assert processor.name == "TestProcessor"
        assert processor.enabled is True
        assert processor.debug is True
        assert processor.logger is not None

    def test_processor_initialization_defaults(self):
        """Test processor initialization with defaults."""
        processor = MockProcessor()

        assert processor.name == "MockProcessor"
        assert processor.enabled is True
        assert processor.debug is False

    def test_process_success(self):
        """Test successful processing."""
        processor = MockProcessor()
        result = processor.process("hello world")

        assert result.is_success
        assert result.content == "HELLO WORLD"
        assert result.metadata["processed"] is True

    def test_process_failure(self):
        """Test failed processing."""
        processor = MockProcessor()
        result = processor.process("")

        assert result.is_failed
        assert result.error_message == "Empty content"

    @patch('eve_pipeline.storage.factory.StorageFactory.get_storage_for_path')
    def test_read_file(self, mock_get_storage):
        """Test file reading."""
        mock_storage = Mock()
        mock_storage.read_text.return_value = "file content"
        mock_get_storage.return_value = mock_storage

        processor = MockProcessor()
        content = processor._read_file("test.txt")

        assert content == "file content"
        mock_storage.read_text.assert_called_once_with("test.txt")

    @patch('eve_pipeline.storage.factory.StorageFactory.get_storage_for_path')
    def test_write_file(self, mock_get_storage):
        """Test file writing."""
        mock_storage = Mock()
        mock_get_storage.return_value = mock_storage

        processor = MockProcessor()
        processor._write_file("output.txt", "content")

        mock_storage.write_text.assert_called_once_with("output.txt", "content")

    @pytest.mark.asyncio
    @patch('eve_pipeline.storage.factory.StorageFactory.get_storage_for_path')
    async def test_async_file_operations(self, mock_get_storage):
        """Test async file operations."""
        mock_storage = Mock()
        mock_storage.read_text.return_value = "async content"
        mock_get_storage.return_value = mock_storage

        processor = MockProcessor()

        # Test async read
        content = await processor._read_file_async("test.txt")
        assert content == "async content"

        # Test async write
        await processor._write_file_async("output.txt", "async content")
        mock_storage.write_text.assert_called_with("output.txt", "async content")

    @pytest.mark.asyncio
    @patch('eve_pipeline.storage.factory.StorageFactory.get_storage_for_path')
    async def test_process_file_async(self, mock_get_storage):
        """Test async file processing."""
        mock_storage = Mock()
        mock_storage.read_text.return_value = "hello world"
        mock_get_storage.return_value = mock_storage

        processor = MockProcessor()
        result = await processor.process_file_async("input.txt", "output.txt")

        assert result.is_success
        assert result.content == "HELLO WORLD"
        assert result.input_path == Path("input.txt")
        assert result.output_path == Path("output.txt")


class TestProcessorResult:
    """Test cases for ProcessorResult."""

    def test_result_properties(self):
        """Test result status properties."""
        # Test success
        result = ProcessorResult(status=ProcessorStatus.SUCCESS)
        assert result.is_success
        assert not result.is_failed
        assert not result.is_skipped

        # Test failure
        result = ProcessorResult(status=ProcessorStatus.FAILED)
        assert not result.is_success
        assert result.is_failed
        assert not result.is_skipped

        # Test skipped
        result = ProcessorResult(status=ProcessorStatus.SKIPPED)
        assert not result.is_success
        assert not result.is_failed
        assert result.is_skipped

    def test_result_initialization(self):
        """Test result initialization with defaults."""
        result = ProcessorResult(status=ProcessorStatus.SUCCESS)

        assert result.metadata == {}
        assert result.warnings == []
        assert result.processing_time == 0.0
        assert result.error_message is None
