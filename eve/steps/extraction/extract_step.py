from pathlib import Path
from typing import List, Tuple

from eve.base_step import PipelineStep
from eve.steps.extraction.htmls import HtmlExtractor
from eve.steps.extraction.xmls import XmlExtractor
from eve.steps.extraction.pdfs import PdfExtractor

class ExtractionStep(PipelineStep):
    async def _html_extraction(self, file_path: Path) -> str:
        html_extractor = HtmlExtractor(file_path)
        text = await html_extractor.extract_text()
        return text
    
    async def _pdf_extraction(self, file_path: Path, url: str) -> str:
        pdf_extractor = PdfExtractor(file_path, url)
        text = await pdf_extractor.extract_text()
        return text
    
    async def _xml_extraction(self, file_path: Path) -> str:
        xml_extractor = XmlExtractor(file_path)
        text = await xml_extractor.extract_text()
        return text

    async def execute(self, input_data: List[Path]) -> List[Tuple[Path, str]]:
        format = self.config.get("format")  # write a wrapper to find out the extension
        url = self.config.get("url", None)
        self.logger.info(f"Extracting text from {format} files. File count: {len(input_data)}")

        result = []
        for file_path in input_data:
            try:
                if format == "html":
                    extracted_text = await self._html_extraction(file_path)
                elif format == "pdf":
                    extracted_text = await self._pdf_extraction(file_path, url)
                elif format == "xml":
                    extracted_text = await self._xml_extraction(file_path)
                else:
                    self.logger.error(f"Unsupported format: {format}")
                    raise ValueError(f"Unsupported format: {format}")
                if extracted_text:
                    result.append((file_path, extracted_text))
                else:
                    self.logger.warning(f"No text extracted from {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to extract text from {file_path}: {str(e)}")
                continue

        self.logger.info(f"Extracted text from {len(result)} files")
        return result