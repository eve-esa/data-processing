"""Deduplication module for exact and approximate duplicate detection."""

from eve_pipeline.deduplication.exact_deduplicator import ExactDeduplicator
from eve_pipeline.deduplication.lsh_deduplicator import LSHDeduplicator
from eve_pipeline.deduplication.pipeline import DeduplicationPipeline

__all__ = [
    "ExactDeduplicator",
    "LSHDeduplicator",
    "DeduplicationPipeline",
]
