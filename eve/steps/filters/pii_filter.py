"""PII-based document filtering step with abstract/introduction exceptions."""

from typing import List
from eve.base_step import PipelineStep
from eve.model.document import Document
import re


class PiiFilterStep(PipelineStep):
    """Filter documents based on PII (Personally Identifiable Information) token percentage.

    This step calculates the percentage of PII tokens ([PERSON], [EMAIL_ADDRESS]) in documents
    and filters them based on a threshold. Documents with "abstract" or "introduction" in their
    headers or text are kept regardless of PII percentage.

    Config parameters:

        - threshold (float): PII token percentage threshold (e.g., 0.03 for 3%)
        - action (str): Either "keep" or "discard" (default: "discard")
            - "discard": Remove documents with PII >= threshold (except abstract/intro)
            - "keep": Keep only documents with PII >= threshold (except abstract/intro)
        - apply_filter (bool): Whether to apply filtering (default: True)

    Examples:
        # Remove documents with >= 3% PII tokens (but keep abstracts/intros)
        config: {threshold: 0.03, action: "discard", apply_filter: true}

        # Only calculate PII percentage without filtering
        config: {threshold: 0.03, apply_filter: false}
    """

    def __init__(self, config: dict):
        super().__init__(config, name="PiiFilter")

        # Validate required config
        if "threshold" not in config:
            raise ValueError("PiiFilterStep requires 'threshold' parameter in config")

        self.threshold = config.get("threshold")
        self.special_tokens = ["[PERSON]", "[EMAIL_ADDRESS]"]
        self.apply_filter = config.get("apply_filter", True)
        self.action = config.get("action", "discard").lower()

        # Validate action parameter
        if self.action not in ["keep", "discard"]:
            raise ValueError(
                f"Invalid action '{self.action}'. Must be 'keep' or 'discard'"
            )

        # Regex patterns for detecting abstract and introduction
        # Allow for leading whitespace before the header
        self.abstract_header_regex = re.compile(
            r"^\s*#{1,6}\s*abstract\b.*$", re.IGNORECASE | re.MULTILINE
        )
        self.introduction_header_regex = re.compile(
            r"^\s*#{1,6}\s*introduction\b.*$", re.IGNORECASE | re.MULTILINE
        )

        self.logger.info(
            f"Initialized PiiFilter: {self.action} documents with PII >= {self.threshold} "
            f"(except abstract/introduction sections), apply_filter={self.apply_filter}"
        )

    def _has_abstract_or_introduction(self, doc: Document) -> bool:
        """Check if document has abstract or introduction in headers or text.

        Args:
            doc: Document to check

        Returns:
            True if document contains abstract or introduction
        """
        # Check in text content
        has_abstract_in_text = bool(self.abstract_header_regex.search(doc.content))
        has_introduction_in_text = bool(
            self.introduction_header_regex.search(doc.content)
        )

        # Check in headers metadata
        has_abstract_in_headers = False
        has_introduction_in_headers = False

        if "headers" in doc.metadata and isinstance(doc.metadata["headers"], list):
            headers = doc.metadata["headers"]
            has_abstract_in_headers = any(
                "abstract" in h.strip().lower() for h in headers
            )
            has_introduction_in_headers = any(
                "introduction" in h.strip().lower() for h in headers
            )

        return (
            has_abstract_in_text
            or has_introduction_in_text
            or has_abstract_in_headers
            or has_introduction_in_headers
        )

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute PII filtering on documents.

        Args:
            documents: List of documents to process

        Returns:
            Filtered list of documents (if apply_filter is True)
        """
        if not documents:
            self.logger.warning("No documents to process")
            return documents

        # Calculate PII percentage for all documents
        for document in documents:
            try:
                total_words = len(document.content.split())
                special_tokens_count = 0
                for special_token in self.special_tokens:
                    special_tokens_count += document.content.count(special_token)
                # Percentage of pii_tokens overall
                pii_percentage = special_tokens_count / total_words if total_words > 0 else 0
                document.add_pipeline_metadata("pii_tokens_percentage", pii_percentage)
            except Exception as e:
                self.logger.error(
                    f"Error processing {document.filename}, exception {e}"
                )
                document.add_pipeline_metadata("pii_tokens_percentage", 0)

        # Apply filtering if enabled
        if self.apply_filter:
            original_len = len(documents)
            filtered_documents = []

            for doc in documents:
                pii_percentage = doc.get_pipeline_metadata("pii_tokens_percentage", 0)
                has_abstract_intro = self._has_abstract_or_introduction(doc)
                meets_threshold = pii_percentage >= self.threshold

                # Logic:
                # - If document has abstract/intro: ALWAYS keep (regardless of PII)
                # - Otherwise: apply threshold-based filtering according to action
                if has_abstract_intro:
                    should_keep = True
                elif self.action == "discard":
                    # Discard documents with high PII (keep documents with low PII)
                    should_keep = not meets_threshold
                else:  # action == "keep"
                    # Keep only documents with high PII
                    should_keep = meets_threshold

                if should_keep:
                    filtered_documents.append(doc)
                else:
                    self.logger.debug(
                        f"Filtered out {doc.filename} (PII: {pii_percentage:.4f})"
                    )

            filtered_count = len(filtered_documents)
            removed_count = original_len - filtered_count
            percentage_kept = (
                (filtered_count / original_len) * 100 if original_len > 0 else 0
            )

            self.logger.info(
                f"PII filtering complete: {filtered_count}/{original_len} documents kept "
                f"({percentage_kept:.2f}%), {removed_count} documents removed"
            )

            return filtered_documents

        return documents
