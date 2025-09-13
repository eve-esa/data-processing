from unittest.mock import MagicMock, patch
import pytest

from eve.steps.dedup.dedup_step import DuplicationStep
from eve.model.document import Document
from create_files import create_documents, create_temp_files


@pytest.fixture
def temp_files():
    """Yield a list of Document objects created from temporary files."""
    input_files = create_temp_files()
    documents = create_documents(input_files)
    yield documents

@pytest.fixture
def duplication_step():
    return DuplicationStep(config={"method": "exact"})

@pytest.mark.asyncio
async def test_exact_deduplication(temp_files, duplication_step):
    """test exact deduplication."""
    duplication_step.config = {"method": "exact"}
    result = await duplication_step.execute(temp_files)
    assert len(result) == 2  # 3 files - 1 duplicate = 2 remaining
    assert all(isinstance(doc, Document) for doc in result)

@pytest.mark.asyncio
async def test_lsh_deduplication(temp_files, duplication_step):
    """test LSH deduplication."""
    duplication_step.config = {"method": "lsh"}
    with patch('eve.steps.dedup.dedup_step.LSH') as MockLSH:
        mock_lsh_instance = MagicMock()
        mock_lsh_instance.find_duplicates.return_value = [[temp_files[0], temp_files[1]]]
        MockLSH.return_value = mock_lsh_instance
        result = await duplication_step.execute(temp_files)
        assert len(result) == 2  # 3 files - 1 duplicate = 2 remaining 
        assert all(isinstance(doc, Document) for doc in result)

@pytest.mark.asyncio
async def test_invalid_method(temp_files, duplication_step):
    """test invalid deduplication method."""
    duplication_step.config = {"method": "invalid"}
    with pytest.raises(ValueError) as exc_info:
        await duplication_step.execute(temp_files)
    assert str(exc_info.value) == "Invalid deduplication method: invalid"