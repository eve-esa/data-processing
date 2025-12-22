"""Length-based document filtering step."""

from typing import List
from eve.base_step import PipelineStep
from eve.model.document import Document


class LengthFilterStep(PipelineStep):
    """Filter documents based on content length (word count).

    This step filters documents based on their word count with configurable
    thresholds and policies.

    Config parameters:

        - length (int): The word count threshold for filtering
        - comparison (str): Either "less" or "greater" to compare against threshold
        - action (str): Either "keep" or "discard" - what to do with documents matching the condition

    Examples:
        # Keep documents with more than 1000 words
        config: {length: 1000, comparison: "greater", action: "keep"}

        # Discard documents with less than 100 words
        config: {length: 100, comparison: "less", action: "discard"}

        # Keep documents with less than 5000 words (filter out long docs)
        config: {length: 5000, comparison: "less", action: "keep"}
    """

    def __init__(self, config: dict):
        super().__init__(config, name="LengthFilter")

        # Validate required config parameters
        if "length" not in config:
            raise ValueError("LengthFilterStep requires 'length' parameter in config")

        self.length_threshold = config.get("length")
        self.comparison = config.get("comparison", "greater").lower()
        self.action = config.get("action", "keep").lower()

        # Validate comparison parameter
        if self.comparison not in ["less", "greater"]:
            raise ValueError(f"Invalid comparison '{self.comparison}'. Must be 'less' or 'greater'")

        # Validate action parameter
        if self.action not in ["keep", "discard"]:
            raise ValueError(f"Invalid action '{self.action}'. Must be 'keep' or 'discard'")

        self.logger.info(
            f"Initialized LengthFilter: {self.action} documents with "
            f"{self.comparison} than {self.length_threshold} words"
        )

    def _get_word_count(self, document: Document) -> int:
        """Get the word count of a document.

        Args:
            document: Document to count words for

        Returns:
            Number of words in the document
        """
        return len(document.content.split())

    def _meets_length_condition(self, document: Document) -> bool:
        """Check if document meets the length condition.

        Args:
            document: Document to check

        Returns:
            True if document word count meets the comparison condition
        """
        word_count = self._get_word_count(document)

        if self.comparison == "greater":
            return word_count > self.length_threshold
        else:  # "less"
            return word_count < self.length_threshold

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute length filtering on documents.

        Args:
            documents: List of documents to filter

        Returns:
            Filtered list of documents based on length criteria
        """
        if not documents:
            self.logger.warning("No documents to filter")
            return documents

        original_count = len(documents)

        # Add word count to pipeline metadata for all documents
        for document in documents:
            word_count = self._get_word_count(document)
            document.add_pipeline_metadata("word_count", word_count)

        # Apply filtering based on action
        filtered_documents = []

        for document in documents:
            meets_condition = self._meets_length_condition(document)

            # Keep document if:
            # - action is "keep" AND condition is met
            # - action is "discard" AND condition is NOT met
            should_keep = (self.action == "keep" and meets_condition) or \
                         (self.action == "discard" and not meets_condition)

            if should_keep:
                filtered_documents.append(document)
            else:
                word_count = self._get_word_count(document)
                self.logger.debug(
                    f"Filtered out {document.filename} ({word_count} words)"
                )

        filtered_count = len(filtered_documents)
        removed_count = original_count - filtered_count

        # Log statistics
        if original_count > 0:
            percentage_kept = (filtered_count / original_count) * 100
            self.logger.info(
                f"Length filtering complete: {filtered_count}/{original_count} documents kept "
                f"({percentage_kept:.2f}%), {removed_count} documents removed"
            )
        else:
            self.logger.info("No documents were processed")

        return filtered_documents
