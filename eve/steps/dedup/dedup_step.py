from pathlib import Path
from typing import List

from eve.base_step import PipelineStep
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

    async def execute(self, input_data: List[Path]) -> List[Path]:
        method = self.config.get("method", "exact")  # default to exact
        self.logger.info(f"Executing duplication step with method: {method} file count: {len(input_data)}")

        if method == "exact":
            duplicates = await self._exact_deduplication(input_data)
        elif method == "lsh":
            duplicates = await self._lsh_deduplication(input_data)
        else:
            self.logger.error(f"Invalid deduplication method: {method}")
            raise ValueError(f"Invalid deduplication method: {method}")

        # Remove duplicates from input_data
        duplicate_paths = set()
        for group in duplicates:
            # Keep the first file in each group, remove the rest
            for file_path in group[1:]:
                duplicate_paths.add(file_path)

        # Filter out duplicates, keeping the first occurrence
        result = [file_path for file_path in input_data if file_path not in duplicate_paths]
        
        self.logger.info(f"File count: {len(result)} after {method} duplication step.")
        return result