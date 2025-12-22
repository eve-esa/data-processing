"""Reference and acknowledgement filtering step."""

import re
from typing import List
from eve.base_step import PipelineStep
from eve.model.document import Document


class ReferenceFilterStep(PipelineStep):
    """Filter documents containing references or acknowledgements.

    This step removes documents that contain reference or acknowledgement sections,
    checking both in document headers and in the document text content.

    The filter checks for:
    
    - Headers containing: "reference", "references", "acknowledgement", "acknowledgements"
    - Text content containing markdown headers with these keywords

    Config parameters:
    
        - action (str): Either "keep" or "discard" (default: "discard")
            - "discard": Remove documents with references/acknowledgements
            - "keep": Keep only documents with references/acknowledgements

    Examples:
        # Remove documents with references or acknowledgements (default behavior)
        config: {action: "discard"}

        # Keep only documents with references or acknowledgements
        config: {action: "keep"}
    """

    def __init__(self, config: dict):
        super().__init__(config, name="ReferenceFilter")

        self.action = config.get("action", "discard").lower()

        # Validate action parameter
        if self.action not in ["keep", "discard"]:
            raise ValueError(f"Invalid action '{self.action}'. Must be 'keep' or 'discard'")

        # Regex patterns for detecting references and acknowledgements in text
        # Allow for leading whitespace before the header
        self.reference_header_regex = re.compile(
            r"^\s*#{1,6}\s*references?\b.*$", re.IGNORECASE | re.MULTILINE
        )
        self.acknowledgement_header_regex = re.compile(
            r"^\s*#{1,6}\s*acknowledgements?\b.*$", re.IGNORECASE | re.MULTILINE
        )

        # Keywords to check in headers
        self.reference_keywords = ["reference", "references"]
        self.acknowledgement_keywords = ["acknowledgement", "acknowledgements"]

        self.logger.info(
            f"Initialized ReferenceFilter: {self.action} documents with references/acknowledgements"
        )

    def _has_reference_in_headers(self, document: Document) -> bool:
        """Check if document has reference in headers metadata.

        Args:
            document: Document to check

        Returns:
            True if any header contains reference keywords
        """
        if "headers" not in document.metadata:
            return False

        headers = document.metadata["headers"]
        if not isinstance(headers, list):
            return False

        return any(
            h.strip().lower() in self.reference_keywords
            for h in headers
        )

    def _has_acknowledgement_in_headers(self, document: Document) -> bool:
        """Check if document has acknowledgement in headers metadata.

        Args:
            document: Document to check

        Returns:
            True if any header contains acknowledgement keywords
        """
        if "headers" not in document.metadata:
            return False

        headers = document.metadata["headers"]
        if not isinstance(headers, list):
            return False

        return any(
            h.strip().lower() in self.acknowledgement_keywords
            for h in headers
        )

    def _has_reference_in_text(self, document: Document) -> bool:
        """Check if document has reference header in text content.

        Args:
            document: Document to check

        Returns:
            True if text contains reference markdown header
        """
        return bool(self.reference_header_regex.search(document.content))

    def _has_acknowledgement_in_text(self, document: Document) -> bool:
        """Check if document has acknowledgement header in text content.

        Args:
            document: Document to check

        Returns:
            True if text contains acknowledgement markdown header
        """
        return bool(self.acknowledgement_header_regex.search(document.content))

    def _contains_reference_or_acknowledgement(self, document: Document) -> bool:
        """Check if document contains references or acknowledgements.

        Checks both headers metadata and text content.

        Args:
            document: Document to check

        Returns:
            True if document contains references or acknowledgements
        """
        return (
            self._has_reference_in_headers(document) or
            self._has_acknowledgement_in_headers(document) or
            self._has_reference_in_text(document) or
            self._has_acknowledgement_in_text(document)
        )

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute reference/acknowledgement filtering on documents.

        Args:
            documents: List of documents to filter

        Returns:
            Filtered list of documents based on reference/acknowledgement criteria
        """
        if not documents:
            self.logger.warning("No documents to filter")
            return documents

        original_count = len(documents)
        filtered_documents = []

        for document in documents:
            contains_ref_ack = self._contains_reference_or_acknowledgement(document)

            # Keep document if:
            # - action is "discard" AND document does NOT contain ref/ack
            # - action is "keep" AND document DOES contain ref/ack
            should_keep = (self.action == "discard" and not contains_ref_ack) or \
                         (self.action == "keep" and contains_ref_ack)

            if should_keep:
                filtered_documents.append(document)
            else:
                self.logger.debug(
                    f"Filtered out {document.filename} (contains ref/ack: {contains_ref_ack})"
                )

        filtered_count = len(filtered_documents)
        removed_count = original_count - filtered_count

        # Log statistics
        if original_count > 0:
            percentage_kept = (filtered_count / original_count) * 100
            self.logger.info(
                f"Reference/Acknowledgement filtering complete: {filtered_count}/{original_count} documents kept "
                f"({percentage_kept:.2f}%), {removed_count} documents removed"
            )
        else:
            self.logger.info("No documents were processed")

        return filtered_documents
