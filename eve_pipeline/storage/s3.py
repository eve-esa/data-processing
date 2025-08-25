"""S3 storage backend using boto3."""

import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from eve_pipeline.storage.base import StorageBase


class S3Storage(StorageBase):
    """S3 storage backend using boto3."""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize S3 storage.

        Args:
            aws_access_key_id: AWS access key ID. Falls back to env var or default profile.
            aws_secret_access_key: AWS secret access key. Falls back to env var or default profile.
            aws_region: AWS region. Falls back to env var or default region.
            aws_session_token: AWS session token for temporary credentials.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)

        # Initialize S3 client with provided credentials or defaults
        session_kwargs = {}

        if aws_access_key_id:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if aws_region:
            session_kwargs['region_name'] = aws_region
        if aws_session_token:
            session_kwargs['aws_session_token'] = aws_session_token

        # Create session and client
        self.session = boto3.Session(**session_kwargs)
        self.s3_client = self.session.client('s3')

        self.logger.info("S3 storage initialized")

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and key.

        Args:
            s3_path: S3 path in format s3://bucket/key

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If path is not a valid S3 path.
        """
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}")

        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"No bucket specified in S3 path: {s3_path}")

        return bucket, key

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists in S3."""
        try:
            bucket, key = self._parse_s3_path(path)

            if not key:
                # Check if bucket exists
                self.s3_client.head_bucket(Bucket=bucket)
                return True

            # Check if exact key exists (file)
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # Check if it's a "directory" (prefix with objects)
                    response = self.s3_client.list_objects_v2(
                        Bucket=bucket,
                        Prefix=key if key.endswith('/') else key + '/',
                        MaxKeys=1,
                    )
                    return response.get('KeyCount', 0) > 0
                raise

        except (ClientError, ValueError):
            return False

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text content from an S3 object."""
        bucket, key = self._parse_s3_path(path)

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()

            # Try multiple encodings if specified encoding fails
            encodings = [encoding, "utf-8", "latin-1", "cp1252", "iso-8859-1"]

            for enc in encodings:
                try:
                    return content.decode(enc)
                except UnicodeDecodeError:
                    continue

            raise Exception(f"Cannot decode S3 object {path} with any supported encoding")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: {path}")
            elif error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {bucket}")
            else:
                raise Exception(f"Error reading from S3: {e}")

    def read_range(self, path: str, start: int, end: int) -> str:
        """Read a range of bytes from S3 object.

        Args:
            path: S3 path to read.
            start: Start byte position (inclusive).
            end: End byte position (inclusive).

        Returns:
            Content in the specified range as string.
        """
        bucket, key = self._parse_s3_path(path)

        try:
            response = self.s3_client.get_object(
                Bucket=bucket,
                Key=key,
                Range=f'bytes={start}-{end}',
            )
            content = response['Body'].read()

            # Try multiple encodings
            encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
            for encoding in encodings:
                try:
                    return content.decode(encoding, errors='ignore')
                except UnicodeDecodeError:
                    continue

            # If all fail, return with replacement characters
            return content.decode('utf-8', errors='replace')

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: {path}")
            else:
                raise Exception(f"Error reading range from S3: {e}")

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """Write text content to an S3 object."""
        bucket, key = self._parse_s3_path(path)

        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode(encoding),
                ContentType='text/plain',
            )
            self.logger.debug(f"Written text to S3: {path}")

        except ClientError as e:
            raise Exception(f"Error writing to S3: {e}")

    def read_bytes(self, path: str) -> bytes:
        """Read binary content from an S3 object."""
        bucket, key = self._parse_s3_path(path)

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: {path}")
            elif error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {bucket}")
            else:
                raise Exception(f"Error reading from S3: {e}")

    def write_bytes(self, path: str, content: bytes) -> None:
        """Write binary content to an S3 object."""
        bucket, key = self._parse_s3_path(path)

        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content,
            )
            self.logger.debug(f"Written bytes to S3: {path}")

        except ClientError as e:
            raise Exception(f"Error writing to S3: {e}")

    def list_files(self, path: str, pattern: Optional[str] = None) -> list[str]:
        """List files in an S3 prefix."""
        bucket, prefix = self._parse_s3_path(path)

        # Ensure prefix ends with / for directory listing
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')

        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']

                    # Skip directories (keys ending with /)
                    if key.endswith('/'):
                        continue

                    s3_path = f"s3://{bucket}/{key}"

                    # Apply pattern filter if provided
                    if pattern:
                        # Convert glob pattern to regex for matching
                        if fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(Path(key).name, pattern):
                            files.append(s3_path)
                    else:
                        files.append(s3_path)

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise FileNotFoundError(f"S3 bucket not found: {bucket}")
            else:
                self.logger.warning(f"Error listing S3 objects: {e}")
                return []

        return sorted(files)

    def list_files_batch(self, directory: str, patterns: list[str], max_workers: int = 4) -> list[str]:
        """List files matching multiple patterns concurrently.

        Args:
            directory: S3 directory path.
            patterns: List of file patterns to match.
            max_workers: Maximum number of concurrent workers.

        Returns:
            List of matching file paths.
        """
        if not patterns:
            return []

        def list_pattern(pattern: str) -> list[str]:
            return self.list_files(directory, pattern)

        all_files = []
        
        # Use ThreadPoolExecutor for concurrent pattern matching
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pattern = {executor.submit(list_pattern, pattern): pattern 
                               for pattern in patterns}

            for future in as_completed(future_to_pattern):
                pattern = future_to_pattern[future]
                try:
                    pattern_files = future.result()
                    all_files.extend(pattern_files)
                except Exception as e:
                    self.logger.error(f"Error listing files for pattern '{pattern}': {e}")

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file_path in all_files:
            if file_path not in seen:
                seen.add(file_path)
                unique_files.append(file_path)

        return unique_files

    def delete(self, path: str) -> None:
        """Delete an S3 object or all objects with a prefix."""
        bucket, key = self._parse_s3_path(path)

        try:
            if not key:
                raise ValueError("Cannot delete entire bucket")

            # Check if it's a single object
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
                # It's a single object
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                self.logger.debug(f"Deleted S3 object: {path}")
                return
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise

            # Check if it's a prefix (directory)
            prefix = key if key.endswith('/') else key + '/'
            objects_to_delete = []

            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    objects_to_delete.append({'Key': obj['Key']})

            if objects_to_delete:
                # Delete in batches of 1000 (S3 limit)
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i:i+1000]
                    self.s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': batch},
                    )

                self.logger.debug(f"Deleted {len(objects_to_delete)} S3 objects with prefix: {prefix}")

        except ClientError as e:
            raise Exception(f"Error deleting from S3: {e}")

    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy an S3 object."""
        src_bucket, src_key = self._parse_s3_path(src_path)
        dst_bucket, dst_key = self._parse_s3_path(dst_path)

        try:
            copy_source = {'Bucket': src_bucket, 'Key': src_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dst_bucket,
                Key=dst_key,
            )
            self.logger.debug(f"Copied S3 object from {src_path} to {dst_path}")

        except ClientError as e:
            raise Exception(f"Error copying S3 object: {e}")

    def get_metadata(self, path: str) -> dict[str, Any]:
        """Get S3 object metadata."""
        bucket, key = self._parse_s3_path(path)

        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)

            return {
                "size": response.get('ContentLength', 0),
                "modified_time": response.get('LastModified'),
                "etag": response.get('ETag', '').strip('"'),
                "content_type": response.get('ContentType', ''),
                "storage_class": response.get('StorageClass', 'STANDARD'),
                "is_file": True,
                "is_dir": False,
                "bucket": bucket,
                "key": key,
                "path": path,
            }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return {}
            else:
                raise Exception(f"Error getting S3 metadata: {e}")

    def is_dir(self, path: str) -> bool:
        """Check if S3 path represents a directory (prefix with objects)."""
        bucket, key = self._parse_s3_path(path)

        if not key:
            # Root bucket is considered a directory
            return True

        # Check if there are objects with this prefix
        prefix = key if key.endswith('/') else key + '/'

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=1,
            )
            return response.get('KeyCount', 0) > 0
        except ClientError:
            return False

    def upload_file(self, local_path: str, s3_path: str) -> None:
        """Upload a local file to S3.

        Args:
            local_path: Local file path.
            s3_path: S3 destination path.
        """
        bucket, key = self._parse_s3_path(s3_path)

        try:
            self.s3_client.upload_file(local_path, bucket, key)
            self.logger.debug(f"Uploaded {local_path} to {s3_path}")

        except ClientError as e:
            raise Exception(f"Error uploading to S3: {e}")

    def download_file(self, s3_path: str, local_path: str) -> None:
        """Download an S3 object to local file.

        Args:
            s3_path: S3 source path.
            local_path: Local destination path.
        """
        bucket, key = self._parse_s3_path(s3_path)

        # Create local directory if needed
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client.download_file(bucket, key, local_path)
            self.logger.debug(f"Downloaded {s3_path} to {local_path}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise FileNotFoundError(f"S3 object not found: {s3_path}")
            else:
                raise Exception(f"Error downloading from S3: {e}")

    def get_presigned_url(self, path: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for S3 object access.

        Args:
            path: S3 path.
            expires_in: URL expiration time in seconds.

        Returns:
            Presigned URL.
        """
        bucket, key = self._parse_s3_path(path)

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in,
            )
            return url

        except ClientError as e:
            raise Exception(f"Error generating presigned URL: {e}")
