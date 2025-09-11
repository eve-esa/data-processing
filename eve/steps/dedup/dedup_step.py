from pathlib import Path
from typing import List, Union

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.dedup.exact_duplicates import ExactDuplication
from eve.steps.dedup.minhash import LSH

class DuplicationStep(PipelineStep):
    async def _exact_deduplication(self, documents: List[Document]) -> List[Document]:
        finder = ExactDuplication(documents)
        duplicates = await finder.find_duplicates()
        return duplicates

    async def _lsh_deduplication(self, documents: List[Document]) -> List[Document]:
        shingle_size = self.config.get("shingle_size", 3)
        num_perm = self.config.get("num_perm", 128)
        threshold = self.config.get("threshold", 0.8)
        lsh = LSH(documents, shingle_size, num_perm, threshold)
        duplicates = lsh.find_duplicates() 
        return duplicates

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute deduplication on input files or documents.
        
        Args:
            input_data: List of file paths or Document objects to deduplicate
            
        Returns:
            List of Document objects with duplicates removed
        """
        method = self.config.get("method", "exact")  # default to exact
        
        self.logger.info(f"Executing duplication step with method: {method} file count: {len(documents)}")

        if method == "exact":
            duplicates = await self._exact_deduplication(documents)
        elif method == "lsh":
            duplicates = await self._lsh_deduplication(documents)
        else:
            self.logger.error(f"Invalid deduplication method: {method}")
            raise ValueError(f"Invalid deduplication method: {method}")
        
        # Remove duplicates from documents
        duplicate_docs = set()
        duplicates_removed = 0
        for group in duplicates:
            # Keep the first doc in each group, mark the rest as duplicates
            for doc in group[1:]:
                duplicate_docs.add(doc)
                duplicates_removed += 1

        # Filter out duplicates, keeping the first occurrence
        result_documents = []
        for doc in documents:
            if doc not in duplicate_docs:
                result_documents.append(doc)

        self.logger.info(
            f"Deduplication complete: {len(result_documents)} files remaining, {duplicates_removed} duplicates removed"
        )
        return result_documents