"""
Asynchronous S3 storage backend with connection pooling for better performance.
"""

import asyncio
import contextlib
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import aioboto3
import aiofiles

from eve_pipeline.storage.base import StorageBase


class AsyncS3Storage(StorageBase):
    """
    Asynchronous S3 storage backend with connection pooling and batch operations.
    """

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        max_concurrent: int = 16,
        chunk_size: int = 8192,
        **kwargs,
    ):
        """
        Initialize async S3 storage.

        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_region: AWS region
            aws_session_token: AWS session token
            max_concurrent: Maximum concurrent operations
            chunk_size: Chunk size for streaming operations
        """
        super().__init__(**kwargs)

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region or "us-east-1"
        self.aws_session_token = aws_session_token
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size

        # Connection management
        self.session = None
        self.semaphore = None
        self.executor = None

        # Fallback sync client
        self.sync_client = None
        self._init_sync_client()

        self.logger = logging.getLogger(__name__)

    def _init_sync_client(self):
        """Initialize synchronous S3 client as fallback."""
        try:
            import boto3
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region,
            )
            self.sync_client = session.client('s3')
        except Exception as e:
            self.logger.error(f"Failed to initialize sync S3 client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self):
        """Ensure async session is initialized."""
        if self.session is None:
            try:
                self.session = aioboto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_session_token=self.aws_session_token,
                    region_name=self.aws_region,
                )
                self.semaphore = asyncio.Semaphore(self.max_concurrent)
                self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
                self.logger.info("Async S3 session initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize async session: {e}")
                self.session = None

    def _parse_s3_path(self, path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and key."""
        if not path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {path}")

        path_parts = path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ""

        return bucket, key

    async def download_batch(
        self,
        paths: list[str],
        local_dir: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """
        Download multiple S3 objects in parallel.

        Args:
            paths: List of S3 paths to download
            local_dir: Optional local directory to save files

        Returns:
            List of download results with metadata
        """
        await self._ensure_session()

        if not self.session:
            return await self._download_batch_sync(paths, local_dir)

        results = []

        async def download_single(path: str) -> dict[str, Any]:
            async with self.semaphore:
                try:
                    start_time = time.time()

                    if local_dir:
                        local_path = local_dir / Path(path).name
                        content = await self.read_text_async(path)
                        async with aiofiles.open(local_path, 'w', encoding='utf-8') as f:
                            await f.write(content)
                        result = {
                            'path': path,
                            'local_path': str(local_path),
                            'success': True,
                            'size': len(content),
                            'download_time': time.time() - start_time,
                        }
                    else:
                        content = await self.read_text_async(path)
                        result = {
                            'path': path,
                            'content': content,
                            'success': True,
                            'size': len(content),
                            'download_time': time.time() - start_time,
                        }

                    return result

                except Exception as e:
                    self.logger.error(f"Failed to download {path}: {e}")
                    return {
                        'path': path,
                        'success': False,
                        'error': str(e),
                        'download_time': time.time() - start_time,
                    }

        # Process in batches to manage memory
        batch_size = min(self.max_concurrent, 50)
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_tasks = [download_single(path) for path in batch_paths]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions in results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        'success': False,
                        'error': str(result),
                        'download_time': 0,
                    })
                else:
                    results.append(result)

            # Memory cleanup between batches
            if i > 0 and i % (batch_size * 5) == 0:
                gc.collect()

        return results

    async def _download_batch_sync(
        self,
        paths: list[str],
        local_dir: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Fallback synchronous batch download."""
        def download_single_sync(path: str) -> dict[str, Any]:
            try:
                start_time = time.time()
                bucket, key = self._parse_s3_path(path)

                response = self.sync_client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read().decode('utf-8', errors='ignore')

                if local_dir:
                    local_path = local_dir / Path(path).name
                    with open(local_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return {
                        'path': path,
                        'local_path': str(local_path),
                        'success': True,
                        'size': len(content),
                        'download_time': time.time() - start_time,
                    }
                else:
                    return {
                        'path': path,
                        'content': content,
                        'success': True,
                        'size': len(content),
                        'download_time': time.time() - start_time,
                    }

            except Exception as e:
                return {
                    'path': path,
                    'success': False,
                    'error': str(e),
                    'download_time': time.time() - start_time,
                }

        # Use ThreadPoolExecutor for parallel sync operations
        loop = asyncio.get_event_loop()
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

        tasks = [
            loop.run_in_executor(self.executor, download_single_sync, path)
            for path in paths
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def read_text_async(self, path: str) -> str:
        """Asynchronously read text from S3."""
        await self._ensure_session()

        if not self.session:
            # Fallback to sync
            return await self._read_text_sync(path)

        bucket, key = self._parse_s3_path(path)

        try:
            async with self.session.client('s3') as s3:
                response = await s3.get_object(Bucket=bucket, Key=key)
                content = await response['Body'].read()

                # Try multiple encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return content.decode(encoding)
                    except UnicodeDecodeError:
                        continue

                return content.decode('utf-8', errors='replace')

        except Exception as e:
            self.logger.error(f"Failed to read {path}: {e}")
            raise

    async def _read_text_sync(self, path: str) -> str:
        """Fallback synchronous read."""
        def read_sync():
            bucket, key = self._parse_s3_path(path)
            response = self.sync_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()

            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue

            return content.decode('utf-8', errors='replace')

        loop = asyncio.get_event_loop()
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

        return await loop.run_in_executor(self.executor, read_sync)

    async def upload_batch(
        self,
        files: list[dict[str, str]],
        progress_callback=None,
    ) -> list[dict[str, Any]]:
        """
        Upload multiple files to S3 in parallel.

        Args:
            files: List of dicts with 'local_path' and 's3_path' keys
            progress_callback: Optional callback for progress updates

        Returns:
            List of upload results
        """
        await self._ensure_session()

        results = []

        async def upload_single(file_info: dict[str, str]) -> dict[str, Any]:
            async with self.semaphore:
                try:
                    start_time = time.time()
                    local_path = file_info['local_path']
                    s3_path = file_info['s3_path']

                    bucket, key = self._parse_s3_path(s3_path)

                    # Read file content
                    async with aiofiles.open(local_path, 'r', encoding='utf-8') as f:
                        content = await f.read()

                    # Upload to S3
                    if self.session:
                        async with self.session.client('s3') as s3:
                            await s3.put_object(
                                Bucket=bucket,
                                Key=key,
                                Body=content.encode('utf-8'),
                                ContentType='text/plain',
                            )
                    else:
                        # Fallback to sync
                        self.sync_client.put_object(
                            Bucket=bucket,
                            Key=key,
                            Body=content.encode('utf-8'),
                            ContentType='text/plain',
                        )

                    result = {
                        'local_path': local_path,
                        's3_path': s3_path,
                        'success': True,
                        'size': len(content),
                        'upload_time': time.time() - start_time,
                    }

                    if progress_callback:
                        await progress_callback(result)

                    return result

                except Exception as e:
                    self.logger.error(f"Failed to upload {file_info}: {e}")
                    return {
                        'local_path': file_info.get('local_path'),
                        's3_path': file_info.get('s3_path'),
                        'success': False,
                        'error': str(e),
                        'upload_time': time.time() - start_time,
                    }

        # Process in batches
        batch_size = min(self.max_concurrent, 20)
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_tasks = [upload_single(file_info) for file_info in batch_files]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        'success': False,
                        'error': str(result),
                        'upload_time': 0,
                    })
                else:
                    results.append(result)

        return results

    async def close(self):
        """Close all connections and cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        if self.session:
            with contextlib.suppress(Exception):
                await self.session.close()
            self.session = None

        self.semaphore = None
        self.logger.info("Async S3 storage closed")

    def exists(self, path: str) -> bool:
        """Check if S3 object exists (sync method)."""
        try:
            bucket, key = self._parse_s3_path(path)
            self.sync_client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text from S3 (sync method)."""
        bucket, key = self._parse_s3_path(path)
        response = self.sync_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read()
        return content.decode(encoding, errors='replace')

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text to S3 (sync method)."""
        bucket, key = self._parse_s3_path(path)
        self.sync_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode(encoding),
            ContentType='text/plain',
        )

    def list_files(self, directory: str, pattern: str = "*") -> list[str]:
        """List files in S3 directory (sync method)."""
        import fnmatch

        bucket, prefix = self._parse_s3_path(directory)
        if not prefix.endswith('/'):
            prefix += '/'

        files = []
        paginator = self.sync_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    if fnmatch.fnmatch(filename, pattern):
                        files.append(f"s3://{bucket}/{key}")

        return files

    async def list_files_async(self, directory: str, pattern: str = "*") -> list[str]:
        """List files in S3 directory asynchronously."""
        import fnmatch
        
        await self._ensure_session()

        if not self.session:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.list_files, directory, pattern
            )

        bucket, prefix = self._parse_s3_path(directory)
        if not prefix.endswith('/'):
            prefix += '/'

        files = []

        try:
            async with self.session.client('s3') as s3:
                paginator = s3.get_paginator('list_objects_v2')
                
                async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            filename = key.split('/')[-1]
                            if fnmatch.fnmatch(filename, pattern):
                                files.append(f"s3://{bucket}/{key}")
        
        except Exception as e:
            self.logger.error(f"Error listing files in {directory}: {e}")
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.list_files, directory, pattern
            )

        return files

    async def list_files_batch_async(self, directory: str, patterns: list[str]) -> list[str]:
        """List files matching multiple patterns concurrently."""
        if not patterns:
            return []

        tasks = [self.list_files_async(directory, pattern) for pattern in patterns]
        pattern_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_files = []
        for result in pattern_results:
            if isinstance(result, Exception):
                self.logger.error(f"Pattern matching failed: {result}")
                continue
            if isinstance(result, list):
                all_files.extend(result)

        seen = set()
        unique_files = []
        for file_path in all_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)

        return unique_files
