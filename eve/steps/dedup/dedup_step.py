from pathlib import Path
from typing import List, Union

from eve.base_step import PipelineStep
from eve.model.document import Document
from eve.steps.dedup.exact_duplicates import ExactDuplication
from eve.steps.dedup.minhash import LSH

class DuplicationStep(PipelineStep):
    async def _exact_deduplication(self, input_data: List[Path]) -> List[List[Path]]:
        finder = ExactDuplication(input_data)
        duplicates = await finder.find_duplicates()
        return duplicates

    async def _lsh_deduplication(self, input_data: List[Path]) -> List[List[Path]]:
        shingle_size = self.config.get("shingle_size", 3)
        num_perm = self.config.get("num_perm", 128)
        threshold = self.config.get("threshold", 0.8)
        lsh = LSH(input_data, shingle_size, num_perm, threshold)
        duplicates = lsh.find_duplicates() 
        return duplicates

    async def execute(self, input_data: Union[List[Path], List[Document]]) -> List[Document]:
        """Execute deduplication on input files or documents.
        
        Args:
            input_data: List of file paths or Document objects to deduplicate
            
        Returns:
            List of Document objects with duplicates removed
        """
        method = self.config.get("method", "exact")  # default to exact
        
        # Handle both Path and Document inputs
        if input_data and isinstance(input_data[0], Document):
            documents = input_data
            file_paths = [doc.file_path for doc in documents]
            # Create mapping from Path to Document
            path_to_doc = {doc.file_path: doc for doc in documents}
        else:
            file_paths = input_data
            # Create basic documents for path-only inputs
            documents = [Document.from_path_and_content(path, "") for path in file_paths]
            path_to_doc = {doc.file_path: doc for doc in documents}
        
        self.logger.info(f"Executing duplication step with method: {method} file count: {len(file_paths)}")

        if method == "exact":
            duplicates = await self._exact_deduplication(file_paths)
        elif method == "lsh":
            duplicates = await self._lsh_deduplication(file_paths)
        else:
            self.logger.error(f"Invalid deduplication method: {method}")
            raise ValueError(f"Invalid deduplication method: {method}")

        # Remove duplicates from input_data
        duplicate_paths = set()
        duplicates_removed = 0
        for group in duplicates:
            # Keep the first file in each group, remove the rest
            for file_path in group[1:]:
                duplicate_paths.add(file_path)
                duplicates_removed += 1

        # Filter out duplicates, keeping the first occurrence
        result_documents = []
        for file_path in file_paths:
            if file_path not in duplicate_paths:
                document = path_to_doc[file_path]
                # Add deduplication metadata
                document.add_metadata('deduplication_method', method)
                document.add_metadata('is_duplicate', False)
                result_documents.append(document)
            else:
                # Mark as duplicate in metadata for reference
                document = path_to_doc[file_path]
                document.add_metadata('deduplication_method', method)
                document.add_metadata('is_duplicate', True)
        
        self.logger.info(f"Deduplication complete: {len(result_documents)} files remaining, {duplicates_removed} duplicates removed")
        return result_documents