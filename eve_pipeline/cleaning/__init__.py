"""Data cleaning module for the pipeline."""

from eve_pipeline.cleaning.pipeline import CleaningPipeline
from eve_pipeline.cleaning.processors import (
    ArtifactRemover,
    LatexCorrector,
    NougatCorrector,
    OCRCorrector,
    OCRDeduplicator,
    RuleBasedCorrector,
)

__all__ = [
    "OCRCorrector",
    "OCRDeduplicator",
    "NougatCorrector",
    "RuleBasedCorrector",
    "ArtifactRemover",
    "LatexCorrector",
    "CleaningPipeline",
]
