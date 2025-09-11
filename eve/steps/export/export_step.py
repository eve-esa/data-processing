from pathlib import Path
import aiofiles
from typing import List

from eve.model.document import Document
from eve.base_step import PipelineStep

class ExportStep(PipelineStep):
    async def execute(self, documents: List[Document]) -> None:
        destination = Path(self.config.get("destination", "./output"))

        if not destination.exists():
            self.logger.info(f"{destination} does not exist. creating...")
            destination.mkdir(parents = True, exist_ok = True)
        
        for document in documents:
            output_file = destination / f"{document.filename}.{self.config.get('format', 'md')}"
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(document.content)
            self.logger.info(f"Saved file: {output_file}")

        return None
