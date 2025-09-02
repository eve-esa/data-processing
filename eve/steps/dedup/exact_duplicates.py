import aiofiles
import asyncio
import hashlib
from pathlib import Path
from collections import defaultdict

from eve.utils import read_in_chunks

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
    async def _calculate_sha256(file: str) -> str:
        """calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        async for chunk in read_in_chunks(file, 'rb'):
            sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    async def _calculate_size(file: str) -> int:
        """calculate file size"""
        stat = await asyncio.to_thread(lambda: Path(file).stat()) # the stat() is blocking so run on a different thread
        return stat.st_size

    async def find_duplicates(self) -> list[list[str]]:
        """Find duplicate files based on size and SHA-256 checksum."""

        # stage 1: group files by size 
        size_tasks = [self._calculate_size(file) for file in self.input_data]
        sizes = await asyncio.gather(*size_tasks)
        
        size_groups = defaultdict(list)
        for file, size in zip(self.input_data, sizes):
            size_groups[size].append(file)

        # stage 2: calculate checksums for potential duplicates
        checksum_tasks = []
        file_info = []
        
        for size, files in size_groups.items():
            if len(files) >= 2:  # Only files with matching sizes
                for file in files:
                    checksum_tasks.append(self._calculate_sha256(file))
                    file_info.append((file, size))

        if not checksum_tasks:
            return []

        checksums = await asyncio.gather(*checksum_tasks)
        
        file_map = defaultdict(list)
        for (file, size), checksum in zip(file_info, checksums):
            key = (size, checksum)
            file_map[key].append(file)

        self.duplicates = {key: paths for key, paths in file_map.items() if len(paths) > 1}
        return list(self.duplicates.values())
