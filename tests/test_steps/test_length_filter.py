"""Tests for the length filter step."""

import pytest
from pathlib import Path

from eve.steps.filters.length_filter import LengthFilterStep
from eve.model.document import Document


class TestLengthFilterStep:
    """Test suite for the LengthFilterStep class."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents with different word counts."""
        return [
            Document(content="Short text", file_path=Path("short.txt"), file_format="txt"),  # 2 words
            Document(content=" ".join(["word"] * 50), file_path=Path("medium.txt"), file_format="txt"),  # 50 words
            Document(content=" ".join(["word"] * 500), file_path=Path("long.txt"), file_format="txt"),  # 500 words
            Document(content=" ".join(["word"] * 2000), file_path=Path("verylong.txt"), file_format="txt"),  # 2000 words
        ]

    def test_initialization_valid_config(self):
        """Test successful initialization with valid config."""
        config = {
            "length": 1000,
            "comparison": "greater",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        assert step.length_threshold == 1000
        assert step.comparison == "greater"
        assert step.action == "keep"

    def test_initialization_missing_length(self):
        """Test that initialization fails without length parameter."""
        config = {
            "comparison": "greater",
            "action": "keep"
        }

        with pytest.raises(ValueError, match="requires 'length' parameter"):
            LengthFilterStep(config)

    def test_initialization_invalid_comparison(self):
        """Test that initialization fails with invalid comparison."""
        config = {
            "length": 1000,
            "comparison": "equal",  # Invalid
            "action": "keep"
        }

        with pytest.raises(ValueError, match="Invalid comparison"):
            LengthFilterStep(config)

    def test_initialization_invalid_action(self):
        """Test that initialization fails with invalid action."""
        config = {
            "length": 1000,
            "comparison": "greater",
            "action": "remove"  # Invalid
        }

        with pytest.raises(ValueError, match="Invalid action"):
            LengthFilterStep(config)

    def test_initialization_default_values(self):
        """Test that default values are set correctly."""
        config = {"length": 1000}
        step = LengthFilterStep(config)

        assert step.comparison == "greater"
        assert step.action == "keep"

    @pytest.mark.asyncio
    async def test_keep_greater_than(self, sample_documents):
        """Test keeping documents greater than threshold."""
        config = {
            "length": 100,
            "comparison": "greater",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # Should keep only documents with > 100 words
        assert len(result) == 2
        assert result[0].filename == "long.txt"
        assert result[1].filename == "verylong.txt"

    @pytest.mark.asyncio
    async def test_keep_less_than(self, sample_documents):
        """Test keeping documents less than threshold."""
        config = {
            "length": 100,
            "comparison": "less",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # Should keep only documents with < 100 words
        assert len(result) == 2
        assert result[0].filename == "short.txt"
        assert result[1].filename == "medium.txt"

    @pytest.mark.asyncio
    async def test_discard_greater_than(self, sample_documents):
        """Test discarding documents greater than threshold."""
        config = {
            "length": 100,
            "comparison": "greater",
            "action": "discard"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # Should discard documents with > 100 words, keeping the rest
        assert len(result) == 2
        assert result[0].filename == "short.txt"
        assert result[1].filename == "medium.txt"

    @pytest.mark.asyncio
    async def test_discard_less_than(self, sample_documents):
        """Test discarding documents less than threshold."""
        config = {
            "length": 100,
            "comparison": "less",
            "action": "discard"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # Should discard documents with < 100 words, keeping the rest
        assert len(result) == 2
        assert result[0].filename == "long.txt"
        assert result[1].filename == "verylong.txt"

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty document list."""
        config = {
            "length": 1000,
            "comparison": "greater",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        result = await step.execute([])

        assert result == []

    @pytest.mark.asyncio
    async def test_all_filtered_out(self, sample_documents):
        """Test case where all documents are filtered out."""
        config = {
            "length": 5000,
            "comparison": "greater",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # No documents have > 5000 words
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_none_filtered_out(self, sample_documents):
        """Test case where no documents are filtered out."""
        config = {
            "length": 1,
            "comparison": "greater",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # All documents have > 1 character
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_metadata_added(self, sample_documents):
        """Test that word_count metadata is added to documents."""
        config = {
            "length": 100,
            "comparison": "greater",
            "action": "keep"
        }
        step = LengthFilterStep(config)

        result = await step.execute(sample_documents)

        # Check that all filtered documents have word_count metadata
        for doc in result:
            assert "word_count" in doc.metadata
            assert doc.metadata["word_count"] == len(doc.content.split())

    @pytest.mark.asyncio
    async def test_exact_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        docs = [
            Document(content=" ".join(["word"] * 100), file_path=Path("exact.txt"), file_format="txt"),  # exactly 100 words
            Document(content=" ".join(["word"] * 99), file_path=Path("under.txt"), file_format="txt"),   # 99 words
            Document(content=" ".join(["word"] * 101), file_path=Path("over.txt"), file_format="txt"),   # 101 words
        ]

        # Test greater than (should not include exactly 100)
        config_greater = {
            "length": 100,
            "comparison": "greater",
            "action": "keep"
        }
        step_greater = LengthFilterStep(config_greater)
        result_greater = await step_greater.execute(docs)

        assert len(result_greater) == 1
        assert result_greater[0].filename == "over.txt"

        # Test less than (should not include exactly 100)
        config_less = {
            "length": 100,
            "comparison": "less",
            "action": "keep"
        }
        step_less = LengthFilterStep(config_less)
        result_less = await step_less.execute(docs)

        assert len(result_less) == 1
        assert result_less[0].filename == "under.txt"

    @pytest.mark.asyncio
    async def test_case_insensitive_config(self):
        """Test that comparison and action parameters are case insensitive."""
        config = {
            "length": 1000,
            "comparison": "GREATER",
            "action": "KEEP"
        }
        step = LengthFilterStep(config)

        assert step.comparison == "greater"
        assert step.action == "keep"

    @pytest.mark.asyncio
    async def test_multiple_filters_chained(self):
        """Test chaining multiple length filters (range filtering)."""
        docs = [
            Document(content=" ".join(["word"] * 10), file_path=Path("tiny.txt"), file_format="txt"),     # 10 words
            Document(content=" ".join(["word"] * 50), file_path=Path("small.txt"), file_format="txt"),    # 50 words
            Document(content=" ".join(["word"] * 200), file_path=Path("medium.txt"), file_format="txt"),  # 200 words
            Document(content=" ".join(["word"] * 1000), file_path=Path("large.txt"), file_format="txt"),  # 1000 words
        ]

        # Filter 1: Discard documents less than 20 words
        config1 = {
            "length": 20,
            "comparison": "less",
            "action": "discard"
        }
        step1 = LengthFilterStep(config1)
        result1 = await step1.execute(docs)

        # Filter 2: Discard documents greater than 500 words
        config2 = {
            "length": 500,
            "comparison": "greater",
            "action": "discard"
        }
        step2 = LengthFilterStep(config2)
        result2 = await step2.execute(result1)

        # Should only have medium-sized documents (between 20 and 500 words)
        assert len(result2) == 2
        assert result2[0].filename == "small.txt"
        assert result2[1].filename == "medium.txt"