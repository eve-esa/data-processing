# REFERENCE - https://github.com/ekzhu/datasketch

'''
Adjust NUM_PERM: Higher values increase accuracy but use more memory.
Adjust THRESHOLD: Higher values find closer duplicates but may miss some.
Adjust SHINGLE_SIZE: Larger shingles are more specific but increase computation.
'''
from typing import Any

from datasketch import MinHash, MinHashLSH
from nltk import ngrams
from tqdm.auto import tqdm


class LSH:

    def _validate(self):
        if len(self.input_data) < 2:
            raise ValueError("need at least 2 files for duplication")

    def __init__(self, input_data: list, shingle_size: int = 3, num_perm: int = 128, threshold: float = 0.8):

        self.input_data = input_data
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.threshold = threshold
        self.file_hashes = {}
        self.duplicates = []

        self._validate()

    def create_shingles(self, text: str) -> set:
        words = text.lower().split()
        return {' '.join(gram) for gram in ngrams(words, self.shingle_size)}

    def _do_lsh(self) -> Any:
        lsh = MinHashLSH(threshold = self.threshold, num_perm = self.num_perm)

        for file in tqdm(self.input_data, total=len(self.input_data)):
            with open(file) as f:
                text = f.read()
                shingles = self.create_shingles(text)

                m = MinHash(num_perm=self.num_perm)
                for shingle in shingles:
                    m.update(shingle.encode('utf8'))

                lsh.insert(file, m)
                self.file_hashes[file] = m

        return lsh

    def find_duplicates(self) -> list[list[str]]:
        """find near-duplicate files using LSH"""

        file_hashes = self._do_lsh()
        processed = set()

        for file in self.input_data:
            if file in processed:
                continue
            m = self.file_hashes[file]
            candidates = file_hashes.query(m)
            # Exclude the file itself and ensure there are other similar files
            candidates = [c for c in candidates if c != file]
            if candidates:  # If there are near-duplicates
                # Create a group including the current file and its near-duplicates
                group = [file, *candidates]
                # Sort the group to ensure consistent ordering
                group = sorted(group)
                # Only add the group if it hasn't been added before
                if group not in self.duplicates:
                    self.duplicates.append(group)
                # Mark all files in the group as processed
                processed.update(group)
        return self.duplicates
