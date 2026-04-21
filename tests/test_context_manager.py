"""Tests for context window management."""
from app.core.context_manager import (
    count_tokens,
    check_context_budget,
    chunk_text,
    merge_json_results,
)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert 2 <= tokens <= 5

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        tokens = count_tokens(text)
        assert 80 <= tokens <= 120

    def test_json_text(self):
        text = '{"evidence_items": [{"content": "test claim", "context": "surrounding text"}]}'
        tokens = count_tokens(text)
        assert tokens > 10


class TestCheckContextBudget:
    def test_fits(self):
        result = check_context_budget(
            system_prompt="You are an analyst.",
            user_prompt="Analyze this short text.",
            safe_context_limit=32768,
            max_output_tokens=4096,
        )
        assert result.fits is True
        assert result.utilization < 0.5

    def test_overflow(self):
        # Create a prompt that's too large
        long_text = "word " * 10000  # ~10000 tokens
        result = check_context_budget(
            system_prompt="System.",
            user_prompt=long_text,
            safe_context_limit=8000,
            max_output_tokens=4096,
        )
        assert result.fits is False
        assert "overflow" in result.message.lower() or "exceeds" in result.message.lower()

    def test_tight_fit(self):
        # Should fit but be high utilization
        text = "word " * 2000
        result = check_context_budget(
            system_prompt="System.",
            user_prompt=text,
            safe_context_limit=8000,
            max_output_tokens=4096,
        )
        assert result.fits is True
        assert result.utilization > 0.5

    def test_reserves_output_space(self):
        result = check_context_budget(
            system_prompt="Short system.",
            user_prompt="Short prompt.",
            safe_context_limit=100,
            max_output_tokens=4096,
        )
        # Even with short input, 4096 output reserve doesn't fit in 100 token budget
        assert result.fits is False


class TestChunkText:
    def test_short_text_no_chunking(self):
        chunks = chunk_text("Short text here.", chunk_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text here."
        assert chunks[0].chunk_index == 0

    def test_long_text_chunks(self):
        text = "word " * 500  # ~500 tokens
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=20)
        assert len(chunks) > 1
        # Each chunk should be roughly 100 tokens
        for chunk in chunks:
            assert chunk.token_count <= 110  # small margin

    def test_chunk_overlap(self):
        text = "word " * 300
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=20)
        # With overlap, later chunks should start before previous ones end
        assert len(chunks) > 2

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_chunk_indices(self):
        text = "word " * 500
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=20)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestMergeJsonResults:
    def test_concatenate_arrays(self):
        results = [
            {"items": [1, 2, 3]},
            {"items": [4, 5, 6]},
        ]
        merged = merge_json_results(results, "concatenate_arrays")
        assert merged["items"] == [1, 2, 3, 4, 5, 6]

    def test_concatenate_preserves_non_arrays(self):
        results = [
            {"items": [1], "summary": "first"},
            {"items": [2], "summary": "second"},
        ]
        merged = merge_json_results(results, "concatenate_arrays")
        assert merged["items"] == [1, 2]
        assert merged["summary"] == "first"  # keeps first occurrence

    def test_keep_last(self):
        results = [
            {"answer": "old"},
            {"answer": "new"},
        ]
        merged = merge_json_results(results, "keep_last")
        assert merged["answer"] == "new"

    def test_empty_results(self):
        assert merge_json_results([], "concatenate_arrays") == {}
