import hashlib
import os
from collections import defaultdict


class ExactDuplication:
    """this class does exact duplication by -
    1. calculate size as a first filter to save computation.
    2. calcuates checksum and finds the duplicates.
    """

    def _validate(self):
        if len(self.input_data) < 2:
            raise ValueError("need at least 2 files for duplication")

    def __init__(self, input_data: list):
        self.input_data = input_data
        self.duplicates = []

        self._validate()

    @staticmethod
    def _calculate_sha256(file: str) -> str:
        """calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _calculate_size(file: str) -> int:
        """calculate file size"""
        return os.path.getsize(file)

    def find_duplicates(self) -> list[list[str]]:
        """Find duplicate files based on size and SHA-256 checksum."""

        # stage 1: Group files by size
        size_groups = defaultdict(list)

        for file in self.input_data:
            file_size = self._calculate_size(file)
            size_groups[file_size].append(file)

        # stage 2: only calculate checksum for files with matching sizes
        file_map = defaultdict(list)

        for size, files in size_groups.items():
            if len(files) < 2:
                # skip files with unique sizes they can't be duplicates
                continue

            # only calculate SHA-256 for files that might be duplicates
            for file in files:
                checksum = self._calculate_sha256(file)
                key = (size, checksum)
                file_map[key].append(file)

        self.duplicates = {key: paths for key, paths in file_map.items() if len(paths) > 1}
        return list(self.duplicates.values())
