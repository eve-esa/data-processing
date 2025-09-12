import aiofiles
from typing import Optional, AsyncGenerator, List, Union, Tuple
from pathlib import Path


def find_format(file_path: Path):
    return file_path.suffix.lstrip('.').lower()

async def read_file(file_path: Path, mode: str, encodings=None) -> Optional[str]:
    if "b" in mode: # binary mode doesnt take encoding
        async with aiofiles.open(file_path, mode) as f:
            return await f.read()
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            async with aiofiles.open(file_path, mode, encoding=encoding) as f:
                return await f.read()
        except UnicodeDecodeError:
            continue
    return None


async def read_in_chunks(file_path: Path, mode: str, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
    """
    read a binary file in chunks.
    """
    async with aiofiles.open(file_path, mode) as f:
        while chunk := await f.read(chunk_size):
            yield chunk


def normalize_to_documents(input_data: Union[List['Document'], List[str], List[Path], List[Tuple[Path, str]]]) -> List['Document']:
    """
    Convert various input formats to a consistent list of Document objects.
    
    Args:
        input_data: List of file paths, Document objects, or tuples to normalize
        
    Returns:
        List of Document objects
    """
    from eve.model.document import Document
    
    if not input_data:
        return []
    
    documents = []
    first_item = input_data[0]
    
    if isinstance(first_item, str):
        # Convert string paths to Documents
        documents = [Document.from_path_and_content(Path(item), "") for item in input_data]
    elif isinstance(first_item, Path):
        # Convert Path objects to Documents
        documents = [Document.from_path_and_content(item, "") for item in input_data]
    elif isinstance(first_item, tuple):
        # Convert tuples to Documents  
        documents = [Document.from_tuple(item) for item in input_data]
    else:
        # Already Document objects
        documents = input_data
    
    return documents
