"""Eve Data Processing Pipeline - A scalable, modular data processing pipeline."""

__version__ = "0.1.0"
__author__ = "Eve Data Processing Team"

from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.core.enums import ExtractionMethod, HashAlgorithm, LogLevel
from eve_pipeline.core.logging import LoggerManager
from eve_pipeline.core.pipeline import Pipeline

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "ExtractionMethod",
    "HashAlgorithm",
    "LogLevel",
    "LoggerManager",
]
