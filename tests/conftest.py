"""Pytest configuration and fixtures for eve-pipeline tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest

from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.core.logging import LoggerManager


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Set up logging for tests."""
    LoggerManager.setup_logging(level="DEBUG", force=True)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """# Sample Document

This is a sample document with some LaTeX formulas:
- Inline formula: $E = mc^2$
- Display formula: $$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$

The document also contains some PII:
- Email: john.doe@example.com
- Phone: +1-555-123-4567

This text has some duplicated content.
This text has some duplicated content.
"""


@pytest.fixture
def sample_config() -> PipelineConfig:
    """Sample pipeline configuration for testing."""
    return PipelineConfig(
        num_processes=1,
        debug=True,
        retry_failed_files=False,
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenRouter/OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "corrected formula"

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_storage():
    """Mock storage backend for testing."""
    mock_storage = Mock()
    mock_storage.exists.return_value = True
    mock_storage.read_text.return_value = "sample content"
    mock_storage.write_text.return_value = None
    mock_storage.list_files.return_value = ["file1.txt", "file2.txt"]
    return mock_storage
