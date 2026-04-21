"""Tests for output validation, JSON parsing, and repair."""
from app.core.output_validator import validate_json_output


class TestCleanJson:
    def test_simple_object(self):
        result = validate_json_output('{"key": "value"}')
        assert result.valid
        assert result.data == {"key": "value"}
        assert not result.was_repaired

    def test_nested_object(self):
        result = validate_json_output('{"items": [{"a": 1}, {"b": 2}]}')
        assert result.valid
        assert len(result.data["items"]) == 2

    def test_array_direct(self):
        result = validate_json_output('[{"a": 1}, {"b": 2}]')
        assert result.valid
        # Direct parse returns the list as-is
        assert isinstance(result.data, list)
        assert len(result.data) == 2


class TestCodeFences:
    def test_json_code_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = validate_json_output(text)
        assert result.valid
        assert result.data == {"key": "value"}

    def test_plain_code_fence(self):
        text = '```\n{"key": "value"}\n```'
        result = validate_json_output(text)
        assert result.valid

    def test_code_fence_with_newlines(self):
        text = '```json\n{\n  "items": [\n    {"content": "test"}\n  ]\n}\n```'
        result = validate_json_output(text)
        assert result.valid
        assert result.data["items"][0]["content"] == "test"


class TestThinkingTags:
    def test_strip_thinking(self):
        text = '<think>Let me analyze this...</think>\n{"key": "value"}'
        result = validate_json_output(text)
        assert result.valid
        assert result.data == {"key": "value"}

    def test_thinking_with_json(self):
        text = '<think>Considering the evidence...</think>{"evidence_items": [{"content": "test"}]}'
        result = validate_json_output(text)
        assert result.valid
        assert len(result.data["evidence_items"]) == 1


class TestPreamble:
    def test_preamble_text(self):
        text = 'Here is the analysis result:\n\n{"key": "value"}'
        result = validate_json_output(text)
        assert result.valid
        assert result.data == {"key": "value"}

    def test_long_preamble(self):
        preamble = "I've analyzed the document and here are my findings. " * 5
        text = preamble + '{"key": "value"}'
        result = validate_json_output(text)
        assert result.valid


class TestTruncatedJson:
    def test_truncated_array(self):
        # Simulates LLM running out of tokens mid-array
        text = '{"evidence_items": [{"content": "claim 1", "context": "ctx 1"}, {"content": "claim 2", "context": "ctx 2"}, {"content": "claim 3'
        result = validate_json_output(text)
        assert result.valid
        assert result.was_repaired
        # Should have salvaged at least the first 2 complete items
        assert len(result.data["evidence_items"]) >= 2

    def test_truncated_after_complete_object(self):
        text = '{"items": [{"a": 1}, {"b": 2}, {"c":'
        result = validate_json_output(text)
        assert result.valid
        assert result.was_repaired
        assert len(result.data["items"]) >= 2

    def test_truncated_mid_string(self):
        text = '{"evidence_items": [{"content": "The president said that the'
        result = validate_json_output(text)
        # May or may not be repairable — depends on structure
        # At minimum it shouldn't crash
        assert isinstance(result, object)


class TestSchemaValidation:
    def test_valid_schema(self):
        schema = {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {"type": "array"},
            },
        }
        result = validate_json_output('{"items": [1, 2, 3]}', schema=schema)
        assert result.valid

    def test_invalid_schema_missing_field(self):
        schema = {
            "type": "object",
            "required": ["items"],
            "properties": {
                "items": {"type": "array"},
            },
        }
        result = validate_json_output('{"other": "value"}', schema=schema)
        assert not result.valid
        assert result.error_code == "SCHEMA_VALIDATION_FAILED"
        # data should still be present (parsed but invalid)
        assert result.data == {"other": "value"}

    def test_no_schema_anything_goes(self):
        result = validate_json_output('{"anything": true}')
        assert result.valid


class TestEmptyAndInvalid:
    def test_empty_string(self):
        result = validate_json_output("")
        assert not result.valid
        assert result.error_code == "EMPTY_RESPONSE"

    def test_whitespace_only(self):
        result = validate_json_output("   \n\t  ")
        assert not result.valid
        assert result.error_code == "EMPTY_RESPONSE"

    def test_pure_text(self):
        result = validate_json_output("I cannot generate JSON for this request.")
        assert not result.valid
        assert result.error_code == "JSON_PARSE_FAILED"

    def test_none(self):
        result = validate_json_output(None)
        assert not result.valid


class TestNewlinesInStrings:
    def test_literal_newlines(self):
        # LLMs often produce literal newlines in JSON string values
        text = '{"content": "line one\nline two", "context": "some context"}'
        result = validate_json_output(text)
        assert result.valid
        assert "line" in result.data["content"]
