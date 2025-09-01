import requests
from pathlib import Path
from typing import Optional

from eve.logging import logger

class PdfExtractor:
    def __init__(self, input_data: list, endpoint: str):
        self.input_data = input_data
        self.endpoint = f"{endpoint}/predict"
        self.extractions = []

    def _call_nougat(self, file_path: Path) -> Optional[str]:
        """
        internal method to call the Nougat API for a single PDF file.
        """
        try:
            with open(file_path, "rb") as f:
                files = {
                    'file': (file_path.name, f, 'application/pdf')
                }
                headers = {
                    'accept': 'application/json'
                }
                response = requests.post(self.endpoint, headers = headers, files = files)

                if response.status_code == 200:
                    return response.text
                else:
                    return None

        except requests.RequestException as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")
            return None

    def extract_text(self) -> list:
        self.extractions = []

        for file_path in self.input_data:
            result = self._call_nougat(file_path)
            self.extractions.append(result)
        return self.extractions