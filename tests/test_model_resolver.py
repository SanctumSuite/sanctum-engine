"""Tests for model resolution and capability mapping."""
from app.core.model_resolver import (
    infer_capabilities,
    infer_safe_context_limit,
    get_model_config,
    resolve_model,
    resolve_model_with_fallback,
)


class TestInferCapabilities:
    def test_qwen(self):
        caps = infer_capabilities("qwen3.5:latest")
        assert "reasoning" in caps
        assert "json" in caps

    def test_ministral(self):
        caps = infer_capabilities("ministral-3:latest")
        assert "fast" in caps
        assert "json" in caps

    def test_embedding(self):
        caps = infer_capabilities("nomic-embed-text:latest")
        assert "embedding" in caps

    def test_vision(self):
        caps = infer_capabilities("deepseek-ocr:latest")
        assert "vision" in caps
        assert "ocr" in caps

    def test_translation(self):
        caps = infer_capabilities("translategemma:27b-it-fp16")
        assert "translation" in caps

    def test_unknown_model(self):
        caps = infer_capabilities("some-unknown-model:latest")
        # Should get sensible defaults
        assert "reasoning" in caps or "fast" in caps


class TestInferSafeContextLimit:
    def test_qwen_capped(self):
        limit = infer_safe_context_limit("qwen3.5:latest", 262144)
        assert limit == 32768

    def test_small_model(self):
        limit = infer_safe_context_limit("deepseek-ocr:latest", 8192)
        assert limit == min(6000, int(8192 * 0.7))

    def test_embedding_model(self):
        limit = infer_safe_context_limit("nomic-embed-text:latest", 2048)
        assert limit <= 1800

    def test_unknown_model_70_percent(self):
        limit = infer_safe_context_limit("unknown-model:latest", 16384)
        assert limit == min(int(16384 * 0.7), 32768)


class TestGetModelConfig:
    def test_qwen_nothink(self):
        config = get_model_config("qwen3.5:latest")
        assert config["append_to_prompt"] == "/nothink"

    def test_normal_model_no_config(self):
        config = get_model_config("ministral-3:latest")
        assert config == {}


class TestResolveModel:
    def test_by_capability(self, db_session):
        result = resolve_model(db_session, "reasoning")
        assert result is not None
        assert result.name in ["test-reasoning:latest", "qwen3.5:latest"]

    def test_by_specific_name(self, db_session):
        result = resolve_model(db_session, "test-fast:latest")
        assert result is not None
        assert result.name == "test-fast:latest"

    def test_embedding(self, db_session):
        result = resolve_model(db_session, "embedding")
        assert result is not None
        assert result.name == "test-embed:latest"

    def test_nonexistent(self, db_session):
        result = resolve_model(db_session, "nonexistent-capability")
        assert result is None


class TestResolveModelWithFallback:
    def test_fallback_to_json(self, db_session):
        # "vision" capability doesn't exist in test data
        result = resolve_model_with_fallback(db_session, "vision")
        assert result is not None
        # Should fall back to a model with json or reasoning capability
        assert result.name in ["test-reasoning:latest", "test-fast:latest", "qwen3.5:latest"]

    def test_qwen_config_preserved(self, db_session):
        result = resolve_model(db_session, "qwen3.5:latest")
        assert result is not None
        assert result.config.get("append_to_prompt") == "/nothink"
