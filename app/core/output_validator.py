"""Output validation, JSON parsing, repair, and schema validation.

This module ensures the engine never returns garbage to the caller.
Either valid, validated output or a clear structured error.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of output validation."""
    valid: bool
    data: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    was_repaired: bool = False


def validate_json_output(
    raw_text: str,
    schema: dict[str, Any] | None = None,
) -> ValidationResult:
    """Parse, repair, and validate JSON from LLM output.

    Pipeline:
    1. Strip wrapping (code fences, thinking tags, preamble)
    2. Try direct parse
    3. Find JSON boundaries and try parse
    4. Attempt truncation repair
    5. Schema validation if schema provided
    """
    if not raw_text or not raw_text.strip():
        return ValidationResult(
            valid=False,
            error_code="EMPTY_RESPONSE",
            error_message="LLM returned empty response",
        )

    cleaned = _strip_wrapping(raw_text)

    # Try direct parse
    parsed = _try_parse(cleaned)
    if parsed is not None:
        return _check_schema(parsed, schema, was_repaired=False)

    # Find JSON object boundaries
    parsed = _find_and_parse_json_object(cleaned)
    if parsed is not None:
        return _check_schema(parsed, schema, was_repaired=False)

    # Find JSON array boundaries
    parsed = _find_and_parse_json_array(cleaned)
    if parsed is not None:
        return _check_schema(parsed, schema, was_repaired=False)

    # Attempt truncation repair
    parsed = _repair_truncated_json(cleaned)
    if parsed is not None:
        logger.warning(f"Repaired truncated JSON ({len(raw_text)} chars)")
        return _check_schema(parsed, schema, was_repaired=True)

    return ValidationResult(
        valid=False,
        error_code="JSON_PARSE_FAILED",
        error_message=f"Could not parse JSON from response ({len(raw_text)} chars): {raw_text[:200]}",
    )


def _strip_wrapping(text: str) -> str:
    """Remove code fences, thinking tags, and preamble text."""
    cleaned = text.strip()

    # Strip <think>...</think> tags
    if "<think>" in cleaned:
        idx = cleaned.rfind("</think>")
        if idx != -1:
            cleaned = cleaned[idx + 8:].strip()

    # Strip preamble text before JSON (if significant text before JSON start)
    for marker in ["```json", "```", '{"', '[{']:
        idx = cleaned.find(marker)
        if idx > 50:
            cleaned = cleaned[idx:].strip()
            break

    # Strip markdown code fences
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)

    return cleaned.strip()


def _try_parse(text: str) -> dict | list | None:
    """Try to parse text as JSON, with multiple fallback strategies."""
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Replace literal newlines inside JSON strings.
    # Walk through the text tracking whether we're inside a string value,
    # and replace \n with space when we are.
    try:
        fixed = _fix_newlines_in_strings(text)
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 3: Regex-based fallback for simpler cases
    try:
        fixed = re.sub(
            r'(?<=": ")(.*?)(?="[,\}])',
            lambda m: m.group(0).replace('\n', ' ').replace('\r', ''),
            text,
            flags=re.DOTALL,
        )
        return json.loads(fixed)
    except (json.JSONDecodeError, Exception):
        pass

    return None


def _fix_newlines_in_strings(text: str) -> str:
    """Replace literal newlines that appear inside JSON string values."""
    result = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue

        if char == '\\':
            result.append(char)
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string and char == '\n':
            result.append(' ')  # replace newline with space
        elif in_string and char == '\r':
            pass  # drop carriage returns
        else:
            result.append(char)

    return ''.join(result)


def _find_and_parse_json_object(text: str) -> dict | None:
    """Find outermost { } and try to parse."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        result = _try_parse(text[start:end + 1])
        if isinstance(result, dict):
            return result
    return None


def _find_and_parse_json_array(text: str) -> dict | None:
    """Find outermost [ ] and try to parse. Wraps arrays in a dict."""
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        result = _try_parse(text[start:end + 1])
        if isinstance(result, list):
            return {"items": result}
    return None


def _repair_truncated_json(text: str) -> dict | None:
    """Attempt to repair JSON truncated by token limit.

    Strategy: find the last complete JSON object in an array,
    then close the containing structures.
    """
    # Find the start of the JSON structure
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return None

    start = min(
        obj_start if obj_start != -1 else len(text),
        arr_start if arr_start != -1 else len(text),
    )

    # Work from the end backwards to find the last complete object
    fragment = text[start:]

    # Count how many brackets/braces are unclosed
    open_braces = 0
    open_brackets = 0
    last_complete_pos = -1

    for i, char in enumerate(fragment):
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
            if open_braces >= 0:
                last_complete_pos = i
        elif char == '[':
            open_brackets += 1
        elif char == ']':
            open_brackets -= 1

    if last_complete_pos == -1:
        return None

    # Truncate to last complete object position
    truncated = fragment[:last_complete_pos + 1]

    # Close unclosed structures
    # Recount after truncation
    open_braces = 0
    open_brackets = 0
    for char in truncated:
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
        elif char == '[':
            open_brackets += 1
        elif char == ']':
            open_brackets -= 1

    # Remove trailing comma if present
    truncated = truncated.rstrip().rstrip(',')

    # Close open structures
    suffix = ']' * open_brackets + '}' * open_braces
    candidate = truncated + suffix

    result = _try_parse(candidate)
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"items": result}

    return None


def _check_schema(
    data: dict | list,
    schema: dict[str, Any] | None,
    was_repaired: bool,
) -> ValidationResult:
    """Validate parsed data against JSON Schema if provided."""
    if schema is None:
        return ValidationResult(valid=True, data=data, was_repaired=was_repaired)

    try:
        jsonschema.validate(instance=data, schema=schema)
        return ValidationResult(valid=True, data=data, was_repaired=was_repaired)
    except jsonschema.ValidationError as e:
        return ValidationResult(
            valid=False,
            data=data,
            error_code="SCHEMA_VALIDATION_FAILED",
            error_message=f"Schema validation failed: {e.message}",
            was_repaired=was_repaired,
        )
