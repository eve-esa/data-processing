import asyncio
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List

from eve.model.document import Document
from eve.utils import read_in_chunks


class ExactDuplication:
    """this class does exact duplication by -
    
    1. calculate size as a first filter to save computation.
    2. calcuates checksum and finds the duplicates.
    """

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.duplicates = []

        self._validate()

    def _validate(self):
        if len(self.documents) < 2:
            raise ValueError("need at least 2 files for duplication")

    @staticmethod
    async def _calculate_sha256(file_path: Path) -> str:
        """calculate SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        async for chunk in read_in_chunks(file_path, 'rb'):
            sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    async def _calculate_size(file_path: Path) -> int:
        """calculate file size"""
        stat = await asyncio.to_thread(lambda: file_path.stat())  # run blocking stat in thread
        return stat.st_size

    async def find_duplicates(self) -> list[list[Document]]:
        """Find duplicate files based on size and SHA-256 checksum."""

        # stage 1: group files by size
        size_tasks = [self._calculate_size(doc.file_path) for doc in self.documents]
        sizes = await asyncio.gather(*size_tasks)

        size_groups = defaultdict(list)
        for doc, size in zip(self.documents, sizes):
            size_groups[size].append(doc)
        
        # stage 2: calculate checksums for potential duplicates
        checksum_tasks = []
        file_info = []

        for size, docs in size_groups.items():
            if len(docs) >= 2:  # Only consider docs with matching sizes
                for doc in docs:
                    checksum_tasks.append(self._calculate_sha256(doc.file_path))
                    file_info.append((doc, size))

        if not checksum_tasks:
            return []

        checksums = await asyncio.gather(*checksum_tasks)

        file_map = defaultdict(list)
        for (doc, size), checksum in zip(file_info, checksums):
            key = (size, checksum)
            file_map[key].append(doc)

        self.duplicates = {key: docs for key, docs in file_map.items() if len(docs) > 1}
        return list(self.duplicates.values())
