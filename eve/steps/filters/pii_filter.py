from typing import Any, List
from eve.base_step import PipelineStep
from eve.model.document import Document
import re

# Regex to match any markdown header level for "Reference" or "References"


class PiiFilterStep(PipelineStep):

    def __init__(self, config: dict):
        super().__init__(config, name="PiiFilter")
        # Percentage
        self.threshold = config.get("threshold")
        self.special_tokens = ["[PERSON]", "[EMAIL_ADDRESS]"]
        self.apply_filter = config.get("apply_filter", False)
        self.action = config.get("action", "keep")
        self.reference_header_regex = re.compile(
            r"^#{1,6}\s*references?\b.*$", re.IGNORECASE | re.MULTILINE
        )

    def reference_filter(self, doc: Document):
        """
        Return true if the header "Reference" appears in the text or in the extracted headers
        """
        regex_in_body = bool(self.reference_header_regex.search(doc.content))
        reference_in_headers = lambda headers: any(
            h.strip().lower() in ["reference", "references"] for h in headers
        )
        return regex_in_body or reference_in_headers(doc.metadata["headers"])

    async def execute(self, documents: List[Document]) -> Any:
        for document in documents:
            try:
                total_words = len(document.content.split(" "))
                special_tokens_count = 0
                for special_token in self.special_tokens:
                    special_tokens_count += document.content.count(special_token)
                # Percentage of pii_tokens overall
                document.metadata["pii_tokens"] = special_tokens_count / total_words
            except Exception as e:
                self.logger.error(
                    f"Error processing {document.filename}, exception {e}"
                )

        if self.apply_filter:
            original_len = len(documents)
            if self.action == "keep":
                condition = lambda doc: doc.metadata[
                    "pii_tokens"
                ] >= self.threshold or not self.reference_filter(doc)
                documents = [doc for doc in documents if condition(doc)]
                log_str = f"""Total docs: {original_len} Remaining docs: {len(documents)} Percentage kept: {len(documents)/original_len * 100:.2f}%"""
            else:
                condition = lambda doc: doc.metadata[
                    "pii_tokens"
                ] < self.threshold or self.reference_filter(doc)
                documents = [doc for doc in documents if condition(doc)]
                log_str = f"""Total docs: {original_len} Remaining docs: {len(documents)} Percentage filtered: {1-len(documents)/original_len * 100:.2f}%"""

            self.logger.info(log_str)

        return documents
