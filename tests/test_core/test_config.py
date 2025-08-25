"""Tests for configuration management."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from eve_pipeline.core.config import (
    DeduplicationConfig,
    ExtractionConfig,
    PipelineConfig,
)
from eve_pipeline.core.enums import ExtractionMethod, HashAlgorithm, LogLevel


class TestPipelineConfig:
    """Test cases for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()

        assert config.num_processes >= 1
        assert config.debug is False
        assert config.log_level == LogLevel.INFO
        assert config.retry_failed_files is False
        assert config.max_retries == 3

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        config = PipelineConfig(
            num_processes=4,
            retry_failed_files=True,
            max_retries=5,
        )
        assert config.num_processes == 4
        assert config.retry_failed_files is True
        assert config.max_retries == 5

    def test_invalid_num_processes(self):
        """Test invalid number of processes."""
        with pytest.raises(ValidationError):
            PipelineConfig(num_processes=0)

        with pytest.raises(ValidationError):
            PipelineConfig(num_processes=-1)

    def test_retry_validation(self):
        """Test retry configuration validation."""
        # Should raise error when retries enabled but max_retries is 0
        with pytest.raises(ValidationError):
            PipelineConfig(retry_failed_files=True, max_retries=0)

    def test_log_level_validation(self):
        """Test log level validation."""
        # Test string conversion
        config = PipelineConfig(log_level="debug")
        assert config.log_level == LogLevel.DEBUG

        # Test enum directly
        config = PipelineConfig(log_level=LogLevel.ERROR)
        assert config.log_level == LogLevel.ERROR

        # Test invalid log level
        with pytest.raises(ValidationError):
            PipelineConfig(log_level="invalid")

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = PipelineConfig(
            num_processes=2,
            debug=True,
            log_level=LogLevel.DEBUG,
        )

        config_dict = config.to_dict()
        assert config_dict["num_processes"] == 2
        assert config_dict["debug"] is True
        assert config_dict["log_level"] == "DEBUG"

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "num_processes": 4,
            "debug": True,
            "log_level": "ERROR",
            "retry_failed_files": True,
            "max_retries": 5,
        }

        config = PipelineConfig.from_dict(config_dict)
        assert config.num_processes == 4
        assert config.debug is True
        assert config.log_level == LogLevel.ERROR
        assert config.retry_failed_files is True
        assert config.max_retries == 5

    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        config = PipelineConfig(
            num_processes=4,
            debug=True,
            log_level=LogLevel.DEBUG,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test YAML
            yaml_path = Path(tmp_dir) / "config.yaml"
            config.save(yaml_path)
            assert yaml_path.exists()

            loaded_config = PipelineConfig.from_file(yaml_path)
            assert loaded_config.num_processes == 4
            assert loaded_config.debug is True
            assert loaded_config.log_level == LogLevel.DEBUG

            # Test JSON
            json_path = Path(tmp_dir) / "config.json"
            config.save(json_path)
            assert json_path.exists()

            loaded_config = PipelineConfig.from_file(json_path)
            assert loaded_config.num_processes == 4

    def test_enabled_stages(self):
        """Test enabled stages property."""
        config = PipelineConfig()
        enabled = config.enabled_stages

        # By default, most stages should be enabled (deduplication is disabled by default)
        expected_stages = ["extraction", "cleaning", "pii_removal", "latex_correction"]
        assert all(stage in enabled for stage in expected_stages)
        # Deduplication is disabled by default
        assert "deduplication" not in enabled

        # Test with some stages disabled
        config.extraction.enabled = False
        config.pii_removal.enabled = False
        enabled = config.enabled_stages

        assert "extraction" not in enabled
        assert "pii_removal" not in enabled
        assert "cleaning" in enabled


class TestExtractionConfig:
    """Test cases for ExtractionConfig."""

    def test_default_extraction_config(self):
        """Test default extraction configuration."""
        config = ExtractionConfig()

        assert config.enabled is True
        assert config.pdf_extractor == ExtractionMethod.NOUGAT
        assert config.batch_size == 32
        assert config.skip_existing is True
        assert "pdf" in config.supported_formats

    def test_pdf_extractor_validation(self):
        """Test PDF extractor validation."""
        # Test string conversion
        config = ExtractionConfig(pdf_extractor="marker")
        assert config.pdf_extractor == ExtractionMethod.MARKER

        # Test enum directly
        config = ExtractionConfig(pdf_extractor=ExtractionMethod.NOUGAT)
        assert config.pdf_extractor == ExtractionMethod.NOUGAT

        # Test invalid extractor
        with pytest.raises(ValidationError):
            ExtractionConfig(pdf_extractor="invalid")

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch size
        config = ExtractionConfig(batch_size=8)
        assert config.batch_size == 8

        # Invalid batch size
        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=0)

        with pytest.raises(ValidationError):
            ExtractionConfig(batch_size=-1)


class TestDeduplicationConfig:
    """Test cases for DeduplicationConfig."""

    def test_default_deduplication_config(self):
        """Test default deduplication configuration."""
        config = DeduplicationConfig()

        assert config.enabled is False
        assert config.exact_deduplication is True
        assert config.lsh_deduplication is True
        assert config.lsh_threshold == 0.8
        assert config.hash_algorithm == HashAlgorithm.MD5

    def test_threshold_validation(self):
        """Test LSH threshold validation."""
        # Valid thresholds
        config = DeduplicationConfig(lsh_threshold=0.5)
        assert config.lsh_threshold == 0.5

        config = DeduplicationConfig(lsh_threshold=1.0)
        assert config.lsh_threshold == 1.0

        # Invalid thresholds
        with pytest.raises(ValidationError):
            DeduplicationConfig(lsh_threshold=-0.1)

        with pytest.raises(ValidationError):
            DeduplicationConfig(lsh_threshold=1.1)

    def test_hash_algorithm_validation(self):
        """Test hash algorithm validation."""
        # Test string conversion
        config = DeduplicationConfig(hash_algorithm="sha256")
        assert config.hash_algorithm == HashAlgorithm.SHA256

        # Test enum directly
        config = DeduplicationConfig(hash_algorithm=HashAlgorithm.SHA1)
        assert config.hash_algorithm == HashAlgorithm.SHA1

        # Test invalid algorithm
        with pytest.raises(ValidationError):
            DeduplicationConfig(hash_algorithm="invalid")

    def test_numeric_validation(self):
        """Test numeric field validation."""
        # Valid values
        config = DeduplicationConfig(
            lsh_num_perm=256,
            lsh_shingle_size=5,
        )
        assert config.lsh_num_perm == 256
        assert config.lsh_shingle_size == 5

        # Invalid values
        with pytest.raises(ValidationError):
            DeduplicationConfig(lsh_num_perm=0)

        with pytest.raises(ValidationError):
            DeduplicationConfig(lsh_shingle_size=0)
