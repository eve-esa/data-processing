from eve.base_step import PipelineStep
from eve.steps.dedup.exact_duplicates import ExactDuplication
from eve.steps.dedup.minhash import LSH


class DuplicationStep(PipelineStep):

    def _exact_deduplication(self, input_data):
        finder = ExactDuplication(input_data)
        duplicates = finder.find_duplicates()
        return duplicates

    def _lsh_deduplication(self, input_data):
        shingle_size = self.config.get("shingle_size", 3)
        num_perm = self.config.get("num_perm", 128)
        threshold = self.config.get("threshold", 0.8)
        lsh = LSH(input_data, shingle_size, num_perm, threshold)
        duplicates = lsh.find_duplicates()
        return duplicates

    def execute(self, input_data) -> list:
        method = self.config.get("method", "exact")  # fefault to exact
        self.logger.info(f"Executing duplication step with method: {method} file count: {len(input_data)}")

        if method == "exact":
            duplicates = self._exact_deduplication(input_data)
        elif method == "lsh":
            duplicates = self._lsh_deduplication(input_data)
        else:
            self.logger.error(f"invalid deduplication method: {method}")
            raise ValueError(f"invalid deduplication method: {method}")

        for pair in duplicates: # remove the duplicates from the processing
            for file in pair:
                input_data.remove(file)
        self.logger.info(f"file count: {len(input_data)} after {method} duplication step.")
        return input_data

