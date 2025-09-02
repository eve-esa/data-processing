from pathlib import Path
import aiofiles

from eve.base_step import PipelineStep

class ExportStep(PipelineStep):
    async def execute(self, input_data: list) -> list:
        destination = Path(self.config.get("destination", "./output"))

        if not destination.exists():
            self.logger.info(f"{destination} does not exist. creating...")
            destination.mkdir(parents = True, exist_ok = True)

        for file_path, content in input_data:
            output_file = destination / f"{file_path.stem}.{self.config.get('format', 'md')}"
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(content)
            self.logger.info(f"Saved file: {output_file}")

        return input_data
