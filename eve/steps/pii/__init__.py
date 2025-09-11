"""
PII removal module for the EVE pipeline.

This module provides PII (Personally Identifiable Information) removal using
Microsoft Presidio with support for both local and remote server processing.
"""

from .pii_step import PIIStep, create_pii_step
from .pii_processors import (
    PIIProcessor,
    LocalPresidioProcessor, 
    RemoteServerProcessor,
)

__all__ = [
    "PIIStep",
    "create_pii_step", 
    "PIIProcessor",
    "LocalPresidioProcessor",
    "RemoteServerProcessor"
]
