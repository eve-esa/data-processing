"""Core pipeline components and interfaces."""

from eve_pipeline.core.base import ProcessorBase, ProcessorResult
from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.core.pipeline import Pipeline

__all__ = ["ProcessorBase", "ProcessorResult", "PipelineConfig", "Pipeline"]