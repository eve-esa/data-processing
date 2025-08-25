"""Pipeline configuration management."""

import os
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from eve_pipeline.core.enums import ExtractionMethod, HashAlgorithm, LogLevel


class PerformanceConfig(BaseModel):
    """Configuration for performance optimizations."""
    
    use_async_s3: bool = Field(default=True, description="Use async S3 operations for better performance")
    s3_max_concurrent: int = Field(default=16, gt=0, description="Maximum concurrent S3 operations")
    streaming_batch_size: Optional[int] = Field(default=None, description="Batch size for streaming processing (auto if None)")
    concurrent_pattern_matching: bool = Field(default=True, description="Use concurrent pattern matching for file discovery")


class StorageConfig(BaseModel):
    """Storage configuration."""

    save_to_local: bool = True
    save_to_s3: bool = False
    local_base_dir: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    aws_region: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    aws_access_key_id: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY"))
    aws_session_token: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_SESSION_TOKEN"))

    def to_storage_kwargs(self) -> dict[str, Any]:
        """Convert to storage factory kwargs."""
        return {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_region": self.aws_region,
            "aws_session_token": self.aws_session_token,
            "base_path": self.local_base_dir,
        }



class ExtractionConfig(BaseModel):
    """Configuration for extraction stage."""

    enabled: bool = True
    supported_formats: list[str] = Field(
        default=["pdf", "xml", "html", "htm", "txt", "text", "csv", "tsv", "md", "markdown"],
        description="List of supported file formats for extraction",
    )
    pdf_extractor: ExtractionMethod = Field(
        default=ExtractionMethod.NOUGAT,
        description="PDF extraction method to use",
    )
    nougat_checkpoint: Optional[str] = Field(
        default_factory=lambda: os.getenv("NOUGAT_CHECKPOINT"),
        description="Path to Nougat model checkpoint",
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for processing")
    skip_existing: bool = Field(default=True, description="Skip processing if output already exists")

    @field_validator("pdf_extractor", mode="before")
    @classmethod
    def validate_pdf_extractor(cls, v):
        """Validate PDF extractor method."""
        if isinstance(v, str):
            try:
                return ExtractionMethod(v.lower())
            except ValueError:
                raise ValueError(f"Invalid PDF extractor method: {v}")
        return v


class CleaningConfig(BaseModel):
    """Configuration for cleaning stage."""

    enabled: bool = True
    ocr_corrections: bool = True
    ocr_deduplication: bool = True
    nougat_correction: bool = True
    rule_based_corrections: bool = True
    artifact_removal: bool = True
    similarity_threshold: float = 0.99
    skip_existing: bool = True


class PIIConfig(BaseModel):
    """Configuration for PII removal stage."""

    enabled: bool = True
    entities: list[str] = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    score_threshold: float = 0.35
    use_flair: bool = True
    use_presidio: bool = True
    server_url: str = "http://localhost:8000"
    skip_existing: bool = True


class DeduplicationConfig(BaseModel):
    """Configuration for deduplication stage."""

    enabled: bool = False
    exact_deduplication: bool = True
    lsh_deduplication: bool = True
    lsh_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="LSH similarity threshold")
    lsh_num_perm: int = Field(default=128, gt=0, description="Number of permutations for MinHash")
    lsh_shingle_size: int = Field(default=3, gt=0, description="Size of n-grams for comparison")
    hash_algorithm: HashAlgorithm = Field(default=HashAlgorithm.MD5, description="Hash algorithm for exact deduplication")
    skip_existing: bool = True

    @field_validator("hash_algorithm", mode="before")
    @classmethod
    def validate_hash_algorithm(cls, v):
        """Validate hash algorithm."""
        if isinstance(v, str):
            try:
                return HashAlgorithm(v.lower())
            except ValueError:
                raise ValueError(f"Invalid hash algorithm: {v}")
        return v


class LatexConfig(BaseModel):
    """Configuration for LaTeX correction stage."""

    enabled: bool = True
    normalize_equations: bool = True
    fix_symbols: bool = True
    clean_artifacts: bool = True
    skip_existing: bool = True


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    # General settings
    num_processes: int = Field(default_factory=lambda: os.cpu_count() or 1, gt=0)
    debug: bool = False
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level for the pipeline")
    retry_failed_files: bool = Field(default=False, description="Enable retry mechanism for failed files")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries for failed files")

    # Input/Output
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None

    # Stage configurations
    storage: StorageConfig = Field(default_factory=StorageConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    pii_removal: PIIConfig = Field(default_factory=PIIConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    latex_correction: LatexConfig = Field(default_factory=LatexConfig)

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        if isinstance(v, str):
            try:
                return LogLevel(v.upper())
            except ValueError:
                raise ValueError(f"Invalid log level: {v}")
        return v

    @model_validator(mode="after")
    def validate_config_consistency(self):
        """Validate configuration consistency."""
        if self.retry_failed_files and self.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0 when retry_failed_files is enabled")

        if self.num_processes > 1 and self.retry_failed_files:
            pass

        return self

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Load config from file (JSON/YAML)."""
        config_path = Path(config_path)

        if config_path.suffix.lower() == ".json":
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            import yaml
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        data = self.model_dump()

        # Convert enum values to strings for serialization
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif hasattr(obj, 'value'):  # It's an enum
                return obj.value
            else:
                return obj

        return convert_enums(data)

    def save(self, config_path: Union[str, Path]) -> None:
        """Save config to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.suffix.lower() == ".json":
            import json
            with open(config_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:

            import yaml

            # Create a custom yaml representer for enums
            def enum_representer(dumper, data):
                return dumper.represent_scalar('tag:yaml.org,2002:str', data.value)

            # Register representer for all enum types used in the config
            yaml.add_representer(LogLevel, enum_representer)
            yaml.add_representer(ExtractionMethod, enum_representer)
            yaml.add_representer(HashAlgorithm, enum_representer)

            with open(config_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    @property
    def enabled_stages(self) -> list[str]:
        """Get list of enabled pipeline stages."""
        stages = []
        if self.extraction.enabled:
            stages.append("extraction")
        if self.cleaning.enabled:
            stages.append("cleaning")
        if self.pii_removal.enabled:
            stages.append("pii_removal")
        if self.deduplication.enabled:
            stages.append("deduplication")
        if self.latex_correction.enabled:
            stages.append("latex_correction")
        return stages
