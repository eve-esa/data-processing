"""Checkpoint management for pipeline resume functionality."""
import json
import hashlib
from pathlib import Path
from typing import Set, List
from eve.model.document import Document


class CheckpointManager:
    """Manages checkpoints for pipeline resume functionality."""

    def __init__(self, output_dir: Path, resume: bool = False):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory where checkpoint file will be stored
            resume: Whether to load existing checkpoint on initialization
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / ".pipeline_checkpoint.jsonl"
        self.processed_ids: Set[str] = set()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load existing checkpoint if resume is enabled
        if resume and self.checkpoint_file.exists():
            self._load_checkpoint()

    def  _get_document_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on file path and content hash.

        Args:
            doc: Document to generate ID for

        Returns:
            Unique document ID
        """
        content_hash = hashlib.md5(doc.content.encode()).hexdigest()[:8]
        return f"{doc.filename}_{content_hash}"

    def _load_checkpoint(self) -> None:
        """Load processed document IDs from checkpoint file."""
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.processed_ids.add(data["doc_id"])
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            self.processed_ids = set()

    def mark_processed(self, doc: Document) -> None:
        """Mark a document as processed and save to checkpoint.

        Args:
            doc: Document that has been processed
        """
        doc_id = self._get_document_id(doc)
        if doc_id not in self.processed_ids:
            self.processed_ids.add(doc_id)

            # Append to checkpoint file
            try:
                with open(self.checkpoint_file, "a", encoding="utf-8") as f:
                    checkpoint_data = {
                        "doc_id": doc_id,
                        "filename": doc.filename,
                        "content_length": len(doc.content)
                    }
                    f.write(json.dumps(checkpoint_data) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write checkpoint: {e}")

    def is_processed(self, doc: Document) -> bool:
        """Check if a document has already been processed.

        Args:
            doc: Document to check

        Returns:
            True if document has been processed, False otherwise
        """
        doc_id = self._get_document_id(doc)
        return doc_id in self.processed_ids

    def filter_unprocessed(self, documents: List[Document]) -> List[Document]:
        """Filter out documents that have already been processed.

        Args:
            documents: List of documents to filter

        Returns:
            List of unprocessed documents
        """
        unprocessed = [doc for doc in documents if not self.is_processed(doc)]

        skipped_count = len(documents) - len(unprocessed)
        if skipped_count > 0:
            print(f"Checkpoint: Skipping {skipped_count} already processed documents")

        return unprocessed

    def clear(self) -> None:
        """Clear the checkpoint file and reset processed IDs."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self.processed_ids = set()

    def get_stats(self) -> dict:
        """Get checkpoint statistics.

        Returns:
            Dictionary with checkpoint stats
        """
        return {
            "checkpoint_exists": self.checkpoint_file.exists(),
            "processed_count": len(self.processed_ids),
            "checkpoint_file": str(self.checkpoint_file)
        }
