from trafilatura import extract

class HtmlExtractor:
    def __init__(self, input_data: list):
        self.input_data = input_data
        self.extractions = []
    
    def extract_text(self) -> list:
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        text = None

        for encoding in encodings_to_try:
            for file in self.input_data:
                with open(file, 'r', encoding = encoding) as file:
                    data = file.read()
                    text = extract(data)
                    self.extractions.append(text)

        return self.extractions