import tempfile
from unittest.mock import MagicMock, patch

import pytest

from eve.steps.dedup.dedup_step import DuplicationStep
from eve.model.document import Document


@pytest.fixture
def temp_files():
    """temporary files creation"""
    # need to try more file formats
    file1 = tempfile.NamedTemporaryFile(mode = 'w', delete = False, suffix = '.txt')
    file2 = tempfile.NamedTemporaryFile(mode = 'w', delete = False, suffix = '.txt')
    file3 = tempfile.NamedTemporaryFile(mode = 'w', delete = False, suffix = '.txt')
    file1.write("test content")
    file2.write("test content")  # Duplicate of file1
    file3.write("unique content")
    file1.close()
    file2.close()
    file3.close()
    yield [file1.name, file2.name, file3.name]

@pytest.fixture
def temp_dir():
    """temporary directory for test output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def duplication_step():
    step = DuplicationStep(config = {"method": "exact"})
    return step

@pytest.mark.asyncio
async def test_exact_deduplication(temp_files, duplication_step):
    """test exact deduplication."""
    duplication_step.config = {"method": "exact"}
    result = await duplication_step.execute(temp_files)
    assert len(result) == 2  # 3 files - 1 duplicate = 2 remaining
    assert all(isinstance(doc, Document) for doc in result)
    assert all(doc.get_metadata('deduplication_method') == 'exact' for doc in result)
    assert all(doc.get_metadata('is_duplicate') == False for doc in result)

@pytest.mark.asyncio
async def test_lsh_deduplication(temp_files, duplication_step):
    """test LSH deduplication."""
    duplication_step.config = {"method": "lsh"}
    with patch('eve.steps.dedup.dedup_step.LSH') as MockLSH:
        mock_lsh_instance = MagicMock()
        # Create Document objects for the mock return value
        # The LSH should return groups of Documents, not file paths
        # We need to return the actual Document objects that will be created during conversion
        def mock_find_duplicates():
            # Since we can't predict the exact Document objects, we'll return an empty list
            # meaning no duplicates found, so all 3 documents should remain
            return []
        
        mock_lsh_instance.find_duplicates.return_value = []
        MockLSH.return_value = mock_lsh_instance
        result = await duplication_step.execute(temp_files)
        assert len(result) == 3  # 3 files - 0 duplicates = 3 remaining 
        assert all(isinstance(doc, Document) for doc in result)
        assert all(doc.get_metadata('deduplication_method') == 'lsh' for doc in result)
        assert all(doc.get_metadata('is_duplicate') == False for doc in result) 

@pytest.mark.asyncio
async def test_invalid_method(temp_files, duplication_step):
    """test invalid deduplication method."""
    duplication_step.config = {"method": "invalid"}
    with pytest.raises(ValueError) as exc_info:
        await duplication_step.execute(temp_files)
    assert str(exc_info.value) == "Invalid deduplication method: invalid"