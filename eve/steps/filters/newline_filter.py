"""Newline-based document filtering step."""

from typing import List
from eve.base_step import PipelineStep
from eve.model.document import Document


class NewLineFilterStep(PipelineStep):
    """Filter documents based on the number of newline chunks.

    This step filters documents based on their newline chunk count with configurable
    thresholds and policies. A newline chunk is defined as a sequence of text
    separated by newline characters.

    Config parameters:
        chunks (int): The newline chunk count threshold for filtering
        comparison (str): Either "less" or "greater" to compare against threshold
        action (str): Either "keep" or "discard" - what to do with documents matching the condition

    Examples:
        # Keep documents with more than 10 chunks
        config: {chunks: 10, comparison: "greater", action: "keep"}

        # Discard documents with less than 5 chunks
        config: {chunks: 5, comparison: "less", action: "discard"}

        # Keep documents with less than 100 chunks (filter out heavily chunked docs)
        config: {chunks: 100, comparison: "less", action: "keep"}
    """

    def __init__(self, config: dict):
        super().__init__(config, name="NewLineFilter")

        # Validate required config parameters
        if "chunks" not in config:
            raise ValueError("NewLineFilterStep requires 'chunks' parameter in config")

        self.chunk_threshold = config.get("chunks")
        self.comparison = config.get("comparison", "greater").lower()
        self.action = config.get("action", "keep").lower()

        # Validate comparison parameter
        if self.comparison not in ["less", "greater"]:
            raise ValueError(f"Invalid comparison '{self.comparison}'. Must be 'less' or 'greater'")

        # Validate action parameter
        if self.action not in ["keep", "discard"]:
            raise ValueError(f"Invalid action '{self.action}'. Must be 'keep' or 'discard'")

        self.logger.info(
            f"Initialized NewLineFilter: {self.action} documents with "
            f"{self.comparison} than {self.chunk_threshold} chunks"
        )

    def _get_chunk_count(self, document: Document) -> int:
        """Get the number of newline characters in a document.

        Counts the actual newline characters (\n) in the document content.

        Args:
            document: Document to count newlines for

        Returns:
            Number of newline characters in the document
        """
        # Count newline characters
        return document.content.count('\n')

    def _meets_chunk_condition(self, document: Document) -> bool:
        """Check if document meets the chunk count condition.

        Args:
            document: Document to check

        Returns:
            True if document chunk count meets the comparison condition
        """
        chunk_count = self._get_chunk_count(document)

        if self.comparison == "greater":
            return chunk_count > self.chunk_threshold
        else:  # "less"
            return chunk_count < self.chunk_threshold

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute newline chunk filtering on documents.

        Args:
            documents: List of documents to filter

        Returns:
            Filtered list of documents based on chunk count criteria
        """
        if not documents:
            self.logger.warning("No documents to filter")
            return documents

        original_count = len(documents)

        # Add chunk count to pipeline metadata for all documents
        for document in documents:
            chunk_count = self._get_chunk_count(document)
            document.add_pipeline_metadata("newline_count", chunk_count)

        # Apply filtering based on action
        filtered_documents = []

        for document in documents:
            meets_condition = self._meets_chunk_condition(document)

            # Keep document if:
            # - action is "keep" AND condition is met
            # - action is "discard" AND condition is NOT met
            should_keep = (self.action == "keep" and meets_condition) or \
                         (self.action == "discard" and not meets_condition)

            if should_keep:
                filtered_documents.append(document)
            else:
                chunk_count = self._get_chunk_count(document)
                self.logger.debug(
                    f"Filtered out {document.filename} ({chunk_count} chunks)"
                )

        filtered_count = len(filtered_documents)
        removed_count = original_count - filtered_count

        # Log statistics
        if original_count > 0:
            percentage_kept = (filtered_count / original_count) * 100
            self.logger.info(
                f"NewLine filtering complete: {filtered_count}/{original_count} documents kept "
                f"({percentage_kept:.2f}%), {removed_count} documents removed"
            )
        else:
            self.logger.info("No documents were processed")

        return filtered_documents
