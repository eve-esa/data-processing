from pathlib import Path
import aiofiles
from typing import Optional, AsyncGenerator

async def read_file(file_path: Path, mode: str, encodings = None) -> Optional[str]:
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            async with aiofiles.open(file_path, mode, encoding = encoding) as f:
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
