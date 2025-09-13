import tempfile
from pathlib import Path

from eve.model.document import Document
from eve.utils import find_format

def create_temp_files():
    files_config = [
        ('test content', '.txt'),
        ('test content', '.txt'),
        ('unique content', '.txt')
    ]
    input_files = []
    for content, suffix in files_config:
        with tempfile.NamedTemporaryFile(mode = 'w', delete = False, suffix = suffix) as temp_file:
            temp_file.write(content)
            input_files.append(temp_file.name)
    return input_files

def create_documents(input_files):
    """Create Document objects from file paths."""
    documents = []
    for file_path in input_files:
        path_obj = Path(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
        doc = Document(
            file_path=path_obj,
            content=content,
            file_format=find_format(path_obj),
        )
        documents.append(doc)
    return documents