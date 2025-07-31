"""Eve Data Processing Pipeline - A scalable, modular data processing pipeline."""

__version__ = "0.1.0"
__author__ = "Eve Data Processing Team"

from eve_pipeline.core.pipeline import Pipeline
from eve_pipeline.core.config import PipelineConfig

__all__ = ["Pipeline", "PipelineConfig"]