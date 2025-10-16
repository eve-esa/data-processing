from pathlib import Path
import aiofiles
from typing import List

from eve.model.document import Document
from eve.base_step import PipelineStep
import json


class ExportStep(PipelineStep):

    async def export_jsonl(self, documents: List[Document]) -> List[Document]:
        destination = Path(self.config.get("destination", "./output"))
        result = []

        if not destination.exists():
            self.logger.info(f"{destination} does not exist. creating...")
            destination.mkdir(parents=True, exist_ok=True)

        for document in documents:
            output_file = (
                destination
                / f"{Path(document.filename).stem}.{self.config.get('format', 'jsonl')}"
            )
            async with aiofiles.open(output_file, "a+", encoding="utf-8") as f:
                await f.write(json.dumps(document.__dict__()))
                await f.write("\n")
            result.append(document)
        return result

    async def export_md(self, documents: List[Document]) -> List[Document]:
        destination = Path(self.config.get("destination", "./output"))
        result = []
        if not destination.exists():
            self.logger.info(f"{destination} does not exist. creating...")
            destination.mkdir(parents=True, exist_ok=True)
        for document in documents:
            output_file = (
                destination
                / f"{Path(document.filename).stem}.{self.config.get('format', 'jsonl')}"
            )
            async with aiofiles.open(output_file, "a+", encoding="utf-8") as f:
                await f.write(json.dumps(document.__dict__()))
                await f.write("\n")
            self.logger.info(f"Saved file: {output_file}")
            result.append(document)
        return result

    async def dummy_export(self, documents: List[Document]) -> List[Document]:
        return documents

    async def execute(self, documents: List[Document]) -> List[Document]:
        format = self.config.get("format", "jsonl")
        if format == "jsonl":
            result = await self.export_jsonl(documents)
        elif format == "dummy":
            result = await self.dummy_export(documents)
        else:
            result = await self.export_md(documents)

        return result
