from typing import List, Any

from datasketch import MinHash, MinHashLSH
from nltk import ngrams
from tqdm.auto import tqdm

from eve.model.document import Document


class LSH:
    def __init__(
        self,
        documents: List[Document],
        shingle_size: int = 3,
        num_perm: int = 128,
        threshold: float = 0.8,
    ):
        self.documents = documents
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.threshold = threshold
        self.doc_hashes = {}   # map: document -> minHash
        self.duplicates = []

        self._validate()

    def _validate(self):
        if len(self.documents) < 2:
            raise ValueError("need at least 2 files for duplication")

    def create_shingles(self, text: str) -> set[str]:
        """Create shingles (word n-grams) from text."""
        words = text.lower().split()
        return {" ".join(gram) for gram in ngrams(words, self.shingle_size)}

    def _do_lsh(self) -> Any:
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        for doc in tqdm(self.documents, total=len(self.documents)):
            shingles = self.create_shingles(doc.content)

            m = MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                m.update(shingle.encode("utf8"))

            # Use file_path as the LSH key, but keep mapping to Document
            lsh.insert(str(doc.file_path), m)
            self.doc_hashes[doc] = m

        return lsh

    def find_duplicates(self) -> list[list[Document]]:
        """Find near-duplicate documents using LSH."""
        file_hashes = self._do_lsh()
        processed = set()

        for doc in self.documents:
            if doc in processed:
                continue

            m = self.doc_hashes[doc]
            candidates = file_hashes.query(m)

            # Convert LSH string keys back to Document objects
            candidate_docs = [
                d for d in self.documents if str(d.file_path) in candidates and d != doc
            ]

            if candidate_docs:
                group = [doc, *candidate_docs]
                group = sorted(group, key = lambda d: str(d.file_path))  # consistent ordering
                if group not in self.duplicates:
                    self.duplicates.append(group)
                processed.update(group)

        return self.duplicates

