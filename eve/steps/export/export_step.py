from pathlib import Path
import aiofiles
from typing import List

from eve.model.document import Document
from eve.base_step import PipelineStep
from eve.checkpoint import CheckpointManager
import json


class ExportStep(PipelineStep):

    def __init__(self, config: dict, name: str = "ExportStep"):
        """Initialize the export step.

        Args:
            config: Configuration containing:
            
                - output_dir: Output directory path
                - format: Output format (jsonl, md, etc.)
                - resume: Whether to enable resume functionality (default: False)
            name: Name for logging purposes
        """
        super().__init__(config, name)

        # Initialize checkpoint manager if resume is enabled
        self.resume = config.get("resume", False)
        output_dir = Path(config.get("output_dir", "./output"))

        if self.resume:
            self.checkpoint = CheckpointManager(output_dir, resume=True)
            stats = self.checkpoint.get_stats()
            self.logger.info(f"Resume mode enabled: {stats['processed_count']} documents already processed")
        else:
            self.checkpoint = None

    async def export_jsonl(self, documents: List[Document]) -> List[Document]:
        output_dir = Path(self.config.get("output_dir", "./output"))
        result = []

        if not output_dir.exists():
            self.logger.info(f"{output_dir} does not exist. creating...")
            output_dir.mkdir(parents=True, exist_ok=True)

        for document in documents:
            output_file = (
                output_dir
                / f"{Path(document.filename).stem}.{self.config.get('format', 'jsonl')}"
            )
            async with aiofiles.open(output_file, "a+", encoding="utf-8") as f:
                await f.write(json.dumps(document.__dict__()))
                await f.write("\n")

            # Mark as processed in checkpoint
            if self.checkpoint:
                self.checkpoint.mark_processed(document)

            result.append(document)
        return result

    async def export_md(self, documents: List[Document]) -> List[Document]:
        output_dir = Path(self.config.get("output_dir", "./output"))
        result = []
        if not output_dir.exists():
            self.logger.info(f"{output_dir} does not exist. creating...")
            output_dir.mkdir(parents=True, exist_ok=True)
        for document in documents:
            output_file = (
                output_dir
                / f"{Path(document.filename).stem}.{self.config.get('format', 'jsonl')}"
            )
            async with aiofiles.open(output_file, "a+", encoding="utf-8") as f:
                await f.write(json.dumps(document.__dict__()))
                await f.write("\n")
            self.logger.info(f"Saved file: {output_file}")

            # Mark as processed in checkpoint
            if self.checkpoint:
                self.checkpoint.mark_processed(document)

            result.append(document)
        return result

    async def dummy_export(self, documents: List[Document]) -> List[Document]:
        return documents

    async def execute(self, documents: List[Document]) -> List[Document]:
        # Note: Documents are already filtered in create_batches() when resume is enabled
        # No need to filter again here to avoid memory overhead

        format = self.config.get("format", "jsonl")
        if format == "jsonl":
            result = await self.export_jsonl(documents)
        elif format == "dummy":
            result = await self.dummy_export(documents)
        else:
            result = await self.export_md(documents)

        return result
