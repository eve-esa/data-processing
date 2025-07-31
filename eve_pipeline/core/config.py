"""Pipeline configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Storage configuration."""
    
    save_to_local: bool = True
    save_to_s3: bool = False
    local_base_dir: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    aws_region: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_REGION"))
    aws_access_key: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_ACCESS_KEY"))
    aws_secret_key: Optional[str] = Field(default_factory=lambda: os.getenv("AWS_SECRET_KEY"))


class DatabaseConfig(BaseModel):
    """Database configuration for logging."""
    
    enabled: bool = True
    host: str = Field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("DB_PORT", "3306")))
    database: str = Field(default_factory=lambda: os.getenv("DB_NAME", "eve_pipeline"))
    user: str = Field(default_factory=lambda: os.getenv("DB_USER", "root"))
    password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))


class ExtractionConfig(BaseModel):
    """Configuration for extraction stage."""
    
    enabled: bool = True
    supported_formats: List[str] = ["pdf", "xml", "html", "txt", "csv"]
    pdf_extractor: str = "nougat"  # nougat, marker, pypdf
    nougat_checkpoint: Optional[str] = Field(default_factory=lambda: os.getenv("NOUGAT_CHECKPOINT"))
    batch_size: int = 4
    skip_existing: bool = True


class CleaningConfig(BaseModel):
    """Configuration for cleaning stage."""
    
    enabled: bool = True
    ocr_corrections: bool = True
    ocr_deduplication: bool = True
    nougat_correction: bool = True
    rule_based_corrections: bool = True
    artifact_removal: bool = True
    latex_correction: bool = True
    openai_api_key: Optional[str] = None
    similarity_threshold: float = 0.99
    skip_existing: bool = True


class PIIConfig(BaseModel):
    """Configuration for PII removal stage."""
    
    enabled: bool = True
    entities: List[str] = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    score_threshold: float = 0.35
    use_flair: bool = True
    use_presidio: bool = True
    server_url: str = "http://localhost:8000"
    skip_existing: bool = True


class DeduplicationConfig(BaseModel):
    """Configuration for deduplication stage."""
    
    enabled: bool = True
    exact_deduplication: bool = True
    lsh_deduplication: bool = True
    lsh_threshold: float = 0.8
    lsh_num_perm: int = 128
    lsh_shingle_size: int = 3
    skip_existing: bool = True


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
    num_processes: int = Field(default_factory=lambda: os.cpu_count() or 1)
    debug: bool = False
    log_level: str = "INFO"
    
    # Input/Output
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Stage configurations
    storage: StorageConfig = Field(default_factory=StorageConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    pii_removal: PIIConfig = Field(default_factory=PIIConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    latex_correction: LatexConfig = Field(default_factory=LatexConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()
    
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
            with open(config_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    @property
    def enabled_stages(self) -> List[str]:
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