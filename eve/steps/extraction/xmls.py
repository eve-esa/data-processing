import re
import xml.etree.ElementTree as ET

class XmlExtractor:
    def __init__(self, input_data: list):
        self.input_data = input_data
        self.outputs = []

    def _extract_text_from_tree(self, element):
        """recursively extract text from the XML tree."""
        texts = []
        if element.text:
            texts.append(element.text)
        for child in element:
            texts.extend(self._extract_text_from_tree(child))
        if element.tail:
            texts.append(element.tail)
        return texts
    
    def _clean_newlines(self, text: str) -> str:
        """clean excessive newlines and strip whitespace."""
        cleaned_text = re.sub(r'\n{3,}', '\n\n', text)
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def extract_text(self) -> list:
        
        for file_path in self.input_data:
            tree = ET.parse(file_path)
            root = tree.getroot()
            extracted_texts = self._extract_text_from_tree(root)
            full_text = ''.join(extracted_texts)
            cleaned_text = self._clean_newlines(full_text)
            self.outputs.append(cleaned_text)
        
        return self.outputs