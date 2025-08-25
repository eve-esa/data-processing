"""Tests for exact deduplication."""

import threading
from unittest.mock import Mock, patch

import pytest

from eve_pipeline.core.enums import HashAlgorithm
from eve_pipeline.deduplication.exact_deduplicator import ExactDeduplicator


class TestExactDeduplicator:
    """Test cases for ExactDeduplicator."""

    def test_deduplicator_initialization(self):
        """Test deduplicator initialization."""
        dedup = ExactDeduplicator(
            hash_algorithm=HashAlgorithm.SHA256,
            normalize_whitespace=True,
            case_sensitive=False,
        )

        assert dedup.hash_algorithm == HashAlgorithm.SHA256
        assert dedup.normalize_whitespace is True
        assert dedup.case_sensitive is False
        assert dedup.name == "ExactDeduplicator"

    def test_string_hash_algorithm_conversion(self):
        """Test string to enum conversion for hash algorithm."""
        dedup = ExactDeduplicator(hash_algorithm="sha1")
        assert dedup.hash_algorithm == HashAlgorithm.SHA1

        with pytest.raises(ValueError):
            ExactDeduplicator(hash_algorithm="invalid")

    def test_process_success(self):
        """Test successful processing of unique content."""
        dedup = ExactDeduplicator()
        result = dedup.process("Hello world", "file1.txt")

        assert result.is_success
        assert result.metadata["is_duplicate"] is False
        assert "content_hash" in result.metadata
        assert result.metadata["hash_algorithm"] == "md5"

    def test_process_duplicate(self):
        """Test processing of duplicate content."""
        dedup = ExactDeduplicator()

        # Process first file
        result1 = dedup.process("Hello world", "file1.txt")
        assert result1.is_success
        assert not result1.metadata["is_duplicate"]

        # Process duplicate content
        result2 = dedup.process("Hello world", "file2.txt")
        assert result2.is_skipped
        assert result2.metadata["is_duplicate"] is True
        assert result2.metadata["duplicate_of"] == "file1.txt"

    def test_process_empty_content(self):
        """Test processing of empty content."""
        dedup = ExactDeduplicator()
        result = dedup.process("", "empty.txt")

        assert result.is_failed
        assert "Empty content" in result.error_message

    def test_case_sensitivity(self):
        """Test case sensitivity handling."""
        # Case insensitive (default)
        dedup = ExactDeduplicator(case_sensitive=False)
        result1 = dedup.process("Hello World", "file1.txt")
        result2 = dedup.process("hello world", "file2.txt")

        assert result1.is_success
        assert result2.is_skipped  # Should be duplicate

        # Case sensitive
        dedup = ExactDeduplicator(case_sensitive=True)
        result1 = dedup.process("Hello World", "file1.txt")
        result2 = dedup.process("hello world", "file2.txt")

        assert result1.is_success
        assert result2.is_success  # Should not be duplicate

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        # With normalization (default)
        dedup = ExactDeduplicator(normalize_whitespace=True)
        result1 = dedup.process("Hello   world", "file1.txt")
        result2 = dedup.process("Hello world", "file2.txt")

        assert result1.is_success
        assert result2.is_skipped  # Should be duplicate after normalization

        # Without normalization
        dedup = ExactDeduplicator(normalize_whitespace=False)
        result1 = dedup.process("Hello   world", "file1.txt")
        result2 = dedup.process("Hello world", "file2.txt")

        assert result1.is_success
        assert result2.is_success  # Should not be duplicate

    def test_different_hash_algorithms(self):
        """Test different hash algorithms."""
        content = "Test content"

        for algorithm in [HashAlgorithm.MD5, HashAlgorithm.SHA1, HashAlgorithm.SHA256]:
            dedup = ExactDeduplicator(hash_algorithm=algorithm)
            result = dedup.process(content, "test.txt")

            assert result.is_success
            assert result.metadata["hash_algorithm"] == algorithm.value
            assert "content_hash" in result.metadata

    def test_thread_safety(self):
        """Test thread safety of deduplicator."""
        dedup = ExactDeduplicator()
        results = {}
        errors = []

        def process_content(thread_id):
            try:
                content = f"Content {thread_id}"
                result = dedup.process(content, f"file{thread_id}.txt")
                results[thread_id] = result
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_content, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # All results should be successful (different content)
        for result in results.values():
            assert result.is_success
            assert not result.metadata["is_duplicate"]

        # Check statistics
        stats = dedup.get_statistics()
        assert stats["unique_hashes"] == 10
        assert stats["files_processed"] == 10

    def test_concurrent_duplicate_detection(self):
        """Test concurrent processing of duplicate content."""
        dedup = ExactDeduplicator()
        results = {}
        errors = []

        def process_duplicate_content(thread_id):
            try:
                # All threads process the same content
                result = dedup.process("Same content", f"file{thread_id}.txt")
                results[thread_id] = result
            except Exception as e:
                errors.append(e)

        # Create multiple threads processing same content
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_duplicate_content, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

        # One should be original, others should be duplicates
        success_count = sum(1 for r in results.values() if r.is_success)
        duplicate_count = sum(1 for r in results.values() if r.is_skipped)

        assert success_count == 1
        assert duplicate_count == 4

    @patch('eve_pipeline.storage.factory.StorageFactory')
    def test_process_directory(self, mock_storage_factory):
        """Test directory processing."""
        mock_storage = Mock()
        mock_storage.list_files.return_value = ["file1.md", "file2.md", "file3.md"]
        mock_storage_factory.get_storage_for_path.return_value = mock_storage

        dedup = ExactDeduplicator()

        # Mock file reading to return different content for each file
        def mock_read_file(file_path):
            if "file1" in file_path:
                return "Content A"
            elif "file2" in file_path:
                return "Content A"  # Duplicate of file1
            else:
                return "Content B"

        with patch.object(dedup, '_read_file', side_effect=mock_read_file):
            duplicates = dedup.process_directory("/test/dir", "*.md")

        # Should find one duplicate group (file1 and file2)
        assert len(duplicates) == 1

        # Check that the duplicate group contains file1 and file2
        duplicate_group = next(iter(duplicates.values()))
        assert len(duplicate_group) == 2
        assert "file1.md" in duplicate_group
        assert "file2.md" in duplicate_group

    def test_reset_functionality(self):
        """Test reset functionality."""
        dedup = ExactDeduplicator()

        # Process some content
        dedup.process("Content 1", "file1.txt")
        dedup.process("Content 2", "file2.txt")

        # Check statistics before reset
        stats = dedup.get_statistics()
        assert stats["unique_hashes"] == 2
        assert stats["files_processed"] == 2

        # Reset and check
        dedup.reset()
        stats = dedup.get_statistics()
        assert stats["unique_hashes"] == 0
        assert stats["files_processed"] == 0

        # Process same content again - should not be duplicate after reset
        result = dedup.process("Content 1", "file1.txt")
        assert result.is_success
        assert not result.metadata["is_duplicate"]

    def test_memory_efficient_processing(self):
        """Test memory-efficient batch processing."""
        dedup = ExactDeduplicator()

        with patch.object(dedup, '_process_files_generator') as mock_generator:
            # Mock generator to yield batches
            mock_generator.return_value = [
                ["batch1_file1.md", "batch1_file2.md"],
                ["batch2_file1.md", "batch2_file2.md"],
            ]

            def mock_read_file(file_path):
                return f"Content for {file_path}"

            with patch.object(dedup, '_read_file', side_effect=mock_read_file):
                duplicates = dedup.process_directory("/test/dir", batch_size=2)

            # Should process in batches
            mock_generator.assert_called_once_with("/test/dir", "*.md", 2)

            # No duplicates expected (all different content)
            assert len(duplicates) == 0
