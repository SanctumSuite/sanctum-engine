"""Context window management.

Handles token counting, budget enforcement, and chunked processing.
This is the module that prevents the context overflow failures from the old codebase.
"""
import logging
from dataclasses import dataclass

import tiktoken

logger = logging.getLogger(__name__)

# Use cl100k_base for estimation. Not model-exact but within ~10%.
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base encoding."""
    if not text:
        return 0
    return len(_encoder.encode(text))


@dataclass
class ContextBudgetResult:
    """Result of a context budget check."""
    fits: bool
    input_tokens: int
    reserved_output: int
    total_needed: int
    safe_limit: int
    utilization: float  # 0.0 to 1.0
    message: str


def check_context_budget(
    system_prompt: str,
    user_prompt: str,
    safe_context_limit: int,
    max_output_tokens: int = 4096,
) -> ContextBudgetResult:
    """Check if the input fits within the context budget.

    Returns a ContextBudgetResult with details about whether it fits
    and how much of the budget is used.
    """
    input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)
    reserved_output = max(max_output_tokens, 4096)
    total_needed = input_tokens + reserved_output

    fits = total_needed <= safe_context_limit
    utilization = total_needed / safe_context_limit if safe_context_limit > 0 else 1.0

    if fits:
        message = f"OK: {input_tokens} input + {reserved_output} output = {total_needed} / {safe_context_limit} ({utilization:.0%})"
    else:
        overage = total_needed - safe_context_limit
        message = (
            f"Context overflow: {input_tokens} input + {reserved_output} output = {total_needed}, "
            f"exceeds safe limit {safe_context_limit} by {overage} tokens. "
            f"Reduce input or use chunked processing."
        )

    if utilization > 0.6 and fits:
        logger.warning(f"Context utilization high: {utilization:.0%} ({total_needed}/{safe_context_limit})")

    return ContextBudgetResult(
        fits=fits,
        input_tokens=input_tokens,
        reserved_output=reserved_output,
        total_needed=total_needed,
        safe_limit=safe_context_limit,
        utilization=utilization,
        message=message,
    )


@dataclass
class TextChunk:
    """A chunk of text with its position in the original."""
    text: str
    chunk_index: int
    token_count: int
    char_start: int
    char_end: int


def chunk_text(
    text: str,
    chunk_tokens: int = 2000,
    overlap_tokens: int = 200,
) -> list[TextChunk]:
    """Split text into token-sized chunks with overlap.

    Uses token boundaries for accuracy. Each chunk is at most
    chunk_tokens long, with overlap_tokens of overlap with the next.
    """
    if not text:
        return []

    tokens = _encoder.encode(text)
    total_tokens = len(tokens)

    if total_tokens <= chunk_tokens:
        return [TextChunk(
            text=text,
            chunk_index=0,
            token_count=total_tokens,
            char_start=0,
            char_end=len(text),
        )]

    chunks = []
    step = chunk_tokens - overlap_tokens
    if step <= 0:
        step = chunk_tokens  # overlap can't exceed chunk size

    idx = 0
    chunk_index = 0
    while idx < total_tokens:
        end = min(idx + chunk_tokens, total_tokens)
        chunk_token_ids = tokens[idx:end]
        chunk_text_str = _encoder.decode(chunk_token_ids)

        # Calculate approximate char positions
        prefix_text = _encoder.decode(tokens[:idx])
        char_start = len(prefix_text)
        char_end = char_start + len(chunk_text_str)

        chunks.append(TextChunk(
            text=chunk_text_str,
            chunk_index=chunk_index,
            token_count=len(chunk_token_ids),
            char_start=char_start,
            char_end=char_end,
        ))

        idx += step
        chunk_index += 1

    logger.info(f"Split {total_tokens} tokens into {len(chunks)} chunks ({chunk_tokens} tokens, {overlap_tokens} overlap)")
    return chunks


def merge_json_results(results: list[dict], merge_strategy: str, array_key: str | None = None) -> dict:
    """Merge JSON results from chunked processing.

    Strategies:
    - concatenate_arrays: Find array fields and concatenate them across results.
    - keep_last: Return only the last chunk's result.
    """
    if not results:
        return {}

    if merge_strategy == "keep_last":
        return results[-1]

    if merge_strategy == "concatenate_arrays":
        merged = {}
        for result in results:
            for key, value in result.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(value)
                elif key not in merged:
                    # For non-array fields, keep the first occurrence
                    merged[key] = value
        return merged

    # Default: return first result
    logger.warning(f"Unknown merge strategy '{merge_strategy}', returning first result")
    return results[0]
