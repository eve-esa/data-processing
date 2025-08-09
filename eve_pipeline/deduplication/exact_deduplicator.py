"""Exact deduplication using hash-based matching."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from eve_pipeline.core.base import ProcessorBase, ProcessorResult, ProcessorStatus


class ExactDeduplicator(ProcessorBase):
    """Exact deduplication processor using content hashing."""
    
    def __init__(
        self,
        hash_algorithm: str = "md5",
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        **kwargs,
    ) -> None:
        """Initialize exact deduplicator.
        
        Args:
            hash_algorithm: Hash algorithm to use (md5, sha1, sha256).
            normalize_whitespace: Whether to normalize whitespace before hashing.
            case_sensitive: Whether comparison should be case sensitive.
            **kwargs: Additional configuration.
        """
        super().__init__(name="ExactDeduplicator", **kwargs)
        self.hash_algorithm = hash_algorithm.lower()
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive
        
        # Validate hash algorithm
        if self.hash_algorithm not in ["md5", "sha1", "sha256"]:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
        
        # Track seen content hashes
        self.seen_hashes: Set[str] = set()
        self.hash_to_file: Dict[str, str] = {}
    
    def process(
        self,
        content: str,
        input_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ProcessorResult:
        """Process content for exact deduplication.
        
        Args:
            content: Input content to process.
            input_path: Optional input file path.
            **kwargs: Additional processing parameters.
            
        Returns:
            ProcessorResult indicating if content is duplicate.
        """
        if not content or not content.strip():
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                error_message="Empty content provided",
            )
        
        try:
            # Normalize content for comparison
            normalized_content = self._normalize_content(content)
            
            # Calculate hash
            content_hash = self._calculate_hash(normalized_content)
            
            # Check if duplicate
            is_duplicate = content_hash in self.seen_hashes
            
            if is_duplicate:
                duplicate_file = self.hash_to_file.get(content_hash, "unknown")
                self.logger.info(f"Exact duplicate detected: {input_path} matches {duplicate_file}")
                
                return ProcessorResult(
                    status=ProcessorStatus.SKIPPED,
                    input_path=input_path,
                    content=content,
                    metadata={
                        "is_duplicate": True,
                        "duplicate_of": duplicate_file,
                        "content_hash": content_hash,
                        "hash_algorithm": self.hash_algorithm,
                    },
                )
            else:
                # Mark as seen
                self.seen_hashes.add(content_hash)
                if input_path:
                    self.hash_to_file[content_hash] = str(input_path)
                
                return ProcessorResult(
                    status=ProcessorStatus.SUCCESS,
                    input_path=input_path,
                    content=content,
                    metadata={
                        "is_duplicate": False,
                        "content_hash": content_hash,
                        "hash_algorithm": self.hash_algorithm,
                    },
                )
                
        except Exception as e:
            self.logger.error(f"Exact deduplication failed: {e}")
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=input_path,
                content=content,
                error_message=str(e),
            )
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent comparison."""
        normalized = content
        
        # Case normalization
        if not self.case_sensitive:
            normalized = normalized.lower()
        
        # Whitespace normalization
        if self.normalize_whitespace:
            import re
            # Replace multiple whitespace with single space
            normalized = re.sub(r'\s+', ' ', normalized)
            # Remove leading/trailing whitespace
            normalized = normalized.strip()
        
        return normalized
    
    def _calculate_hash(self, content: str) -> str:
        """Calculate hash of content."""
        content_bytes = content.encode('utf-8')
        
        if self.hash_algorithm == "md5":
            return hashlib.md5(content_bytes).hexdigest()
        elif self.hash_algorithm == "sha1":
            return hashlib.sha1(content_bytes).hexdigest()
        elif self.hash_algorithm == "sha256":
            return hashlib.sha256(content_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*.md",
    ) -> Dict[str, List[str]]:
        """Find exact duplicates in a directory.
        
        Args:
            input_dir: Path to input directory (local or S3).
            file_pattern: File pattern to match.
            
        Returns:
            Dictionary mapping hash to list of duplicate files.
        """
        from eve_pipeline.storage.factory import StorageFactory
        
        input_dir_str = str(input_dir)
        storage = StorageFactory.get_storage_for_path(input_dir_str, **self.storage_config)
        files = storage.list_files(input_dir_str, file_pattern)
        
        hash_to_files: Dict[str, List[str]] = {}
        
        for file_path in files:
            try:
                # Read file content (file_path is already a string from storage.list_files)
                content = self._read_file(file_path)
                
                # Calculate hash
                normalized_content = self._normalize_content(content)
                content_hash = self._calculate_hash(normalized_content)
                
                # Group files by hash
                if content_hash not in hash_to_files:
                    hash_to_files[content_hash] = []
                hash_to_files[content_hash].append(file_path)
                
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        # Return only duplicates (hashes with multiple files)
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        
        if duplicates:
            total_duplicate_files = sum(len(files) - 1 for files in duplicates.values())
            self.logger.info(f"Found {len(duplicates)} duplicate groups containing {total_duplicate_files} duplicate files")
        else:
            self.logger.info("No exact duplicates found")
        
        return duplicates
    

    
    def reset(self) -> None:
        """Reset the deduplicator state."""
        self.seen_hashes.clear()
        self.hash_to_file.clear()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get deduplication statistics.
        
        Returns:
            Dictionary with statistics.
        """
        return {
            "unique_hashes": len(self.seen_hashes),
            "files_processed": len(self.hash_to_file),
        }