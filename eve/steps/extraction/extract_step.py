from pathlib import Path

from eve.base_step import PipelineStep
from eve.steps.extraction.htmls import HtmlExtractor
from eve.steps.extraction.xmls import XmlExtractor
from eve.steps.extraction.pdfs import PdfExtractor


class ExtractionStep(PipelineStep):
    def _html_extraction(self, input_data):
        html_extractor = HtmlExtractor(input_data)
        text = html_extractor.extract_text()
        return text
    
    def _pdf_extraction(self, input_data, url):
        pdf_extrator = PdfExtractor(input_data, url)
        text = pdf_extrator.extract_text()
        return text
    
    def _xml_extraction(self, input_data):
        xml_extractor = XmlExtractor(input_data)
        text = xml_extractor.extract_text()
        return text

    def execute(self, input_data) -> list:
        format = self.config.get("format") # else write a wrapper to find out extensions
        url = self.config.get("url", None)
        self.logger.info(f"extracting text from : {format} files. file count: {len(input_data)}")

        if format == "html":
            text = self._html_extraction(input_data)
        elif format == "pdf":
            text = self._pdf_extraction(input_data, url)
        elif format == "xml":
            text = self._xml_extraction(input_data)
        else:
            self.logger.error(f"unsupported format: {format}")
            raise ValueError(f"unsupported format: {format}")
        
        return text

