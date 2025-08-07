"""Storage module for handling various storage backends."""

from eve_pipeline.storage.s3 import S3Storage
from eve_pipeline.storage.local import LocalStorage
from eve_pipeline.storage.factory import StorageFactory

__all__ = [
    "S3Storage",
    "LocalStorage", 
    "StorageFactory",
]