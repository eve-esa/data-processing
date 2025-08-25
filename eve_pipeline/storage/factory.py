"""Storage factory for creating appropriate storage backends."""

import os
from typing import Optional

from eve_pipeline.storage.base import StorageBase
from eve_pipeline.storage.local import LocalStorage
from eve_pipeline.storage.s3 import S3Storage
from eve_pipeline.storage.async_s3 import AsyncS3Storage


class StorageFactory:
    """Factory for creating storage backends based on path or configuration."""

    @staticmethod
    def create_storage(path: Optional[str] = None, **config) -> StorageBase:
        """Create appropriate storage backend based on path or configuration.

        Args:
            path: Path to determine storage type (s3:// for S3, local path for local storage).
            **config: Storage configuration parameters.

        Returns:
            StorageBase instance.
        """
        # Determine storage type from path
        if path and StorageBase.is_s3_path(path):
            return StorageFactory.create_s3_storage(**config)
        else:
            return StorageFactory.create_local_storage(**config)

    @staticmethod
    def create_local_storage(base_path: Optional[str] = None, **kwargs) -> LocalStorage:
        """Create local storage backend.

        Args:
            base_path: Optional base path for relative operations.
            **kwargs: Additional configuration.

        Returns:
            LocalStorage instance.
        """
        return LocalStorage(base_path=base_path, **kwargs)

    @staticmethod
    def create_s3_storage(
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        **kwargs,
    ) -> S3Storage:
        """Create S3 storage backend.

        Args:
            aws_access_key_id: AWS access key ID. Falls back to env vars.
            aws_secret_access_key: AWS secret access key. Falls back to env vars.
            aws_region: AWS region. Falls back to env vars.
            aws_session_token: AWS session token for temporary credentials.
            **kwargs: Additional configuration.

        Returns:
            S3Storage instance.
        """
        # Use provided credentials or fall back to environment variables
        access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = aws_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")

        return S3Storage(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_region=region,
            aws_session_token=session_token,
            **kwargs,
        )

    @staticmethod
    def get_storage_for_path(path: str, **config) -> StorageBase:
        """Get appropriate storage backend for a specific path.

        Args:
            path: Path to determine storage type.
            **config: Storage configuration.

        Returns:
            StorageBase instance.
        """
        return StorageFactory.create_storage(path=path, **config)

    @staticmethod
    def create_mixed_storage(
        input_path: str,
        output_path: Optional[str] = None,
        **config,
    ) -> tuple[StorageBase, Optional[StorageBase]]:
        """Create storage backends for input and output paths.

        Args:
            input_path: Input path.
            output_path: Optional output path.
            **config: Storage configuration.

        Returns:
            Tuple of (input_storage, output_storage).
        """
        input_storage = StorageFactory.get_storage_for_path(input_path, **config)

        output_storage = None
        if output_path:
            output_storage = StorageFactory.get_storage_for_path(output_path, **config)

        return input_storage, output_storage

    @staticmethod
    def create_async_s3_storage(
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        max_concurrent: int = 16,
        **kwargs,
    ) -> AsyncS3Storage:
        """Create async S3 storage backend.

        Args:
            aws_access_key_id: AWS access key ID. Falls back to env vars.
            aws_secret_access_key: AWS secret access key. Falls back to env vars.
            aws_region: AWS region. Falls back to env vars.
            aws_session_token: AWS session token for temporary credentials.
            max_concurrent: Maximum concurrent operations.
            **kwargs: Additional configuration.

        Returns:
            AsyncS3Storage instance.
        """
        access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        region = aws_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")

        return AsyncS3Storage(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_region=region,
            aws_session_token=session_token,
            max_concurrent=max_concurrent,
            **kwargs,
        )

    @staticmethod 
    def get_async_storage_for_path(path: str, **config) -> StorageBase:
        """Get appropriate async storage backend for a specific path.

        Args:
            path: Path to determine storage type.
            **config: Storage configuration.

        Returns:
            StorageBase instance (AsyncS3Storage for S3 paths, LocalStorage for local).
        """
        if StorageBase.is_s3_path(path):
            return StorageFactory.create_async_s3_storage(**config)
        else:
            return StorageFactory.create_local_storage(**config)
