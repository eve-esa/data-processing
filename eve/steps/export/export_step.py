import aiofiles
from pathlib import Path
from typing import List, Union, Tuple

from eve.model.document import Document
from eve.base_step import PipelineStep

class ExportStep(PipelineStep):
    async def execute(self, input_data: Union[List[Document], List[Tuple[Path, str]]]) -> List[Document]:
        destination = Path(self.config.get("destination", "./output"))

        if not destination.exists():
            self.logger.info(f"{destination} does not exist. creating...")
            destination.mkdir(parents = True, exist_ok = True)
        
        documents = []
        if input_data and isinstance(input_data[0], tuple):
            documents = [Document.from_tuple(item) for item in input_data]
        else:
            documents = input_data

        for document in documents:
            output_file = destination / f"{document.filename}.{self.config.get('format', 'md')}"
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(document.content)
            self.logger.info(f"Saved file: {output_file}")

        return input_data
