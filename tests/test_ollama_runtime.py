"""Integration tests for Ollama runtime. Requires Ollama running locally."""
import pytest
import httpx

from app.runtimes.ollama import OllamaRuntime

# Skip all tests if Ollama is not running
pytestmark = pytest.mark.integration


def ollama_available():
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def runtime():
    return OllamaRuntime(host="http://localhost:11434")


@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
class TestOllamaRuntime:

    @pytest.mark.asyncio
    async def test_is_healthy(self, runtime):
        assert await runtime.is_healthy()

    @pytest.mark.asyncio
    async def test_list_models(self, runtime):
        models = await runtime.list_models()
        assert len(models) > 0
        # Check model structure
        m = models[0]
        assert m.name
        assert m.parameter_size

    @pytest.mark.asyncio
    async def test_model_info(self, runtime):
        models = await runtime.list_models()
        assert len(models) > 0
        info = await runtime.model_info(models[0].name)
        assert info is not None
        assert info.context_length > 0

    @pytest.mark.asyncio
    async def test_generate_simple(self, runtime):
        """Test a simple text generation."""
        response = await runtime.generate(
            model="ministral-3:latest",
            messages=[
                {"role": "system", "content": "Reply in exactly one word."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            temperature=0.1,
            max_tokens=50,
            num_ctx=4096,
        )
        assert response.content.strip()
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_generate_json(self, runtime):
        """Test JSON generation."""
        response = await runtime.generate(
            model="ministral-3:latest",
            messages=[
                {"role": "system", "content": "You MUST respond with valid JSON only."},
                {"role": "user", "content": 'Return a JSON object with key "answer" and value "hello".'},
            ],
            temperature=0.1,
            max_tokens=100,
            num_ctx=4096,
        )
        assert "{" in response.content
        assert "answer" in response.content

    @pytest.mark.asyncio
    async def test_embed(self, runtime):
        """Test embedding generation."""
        embeddings = await runtime.embed(
            model="nomic-embed-text:latest",
            texts=["Hello world", "Intelligence analysis"],
        )
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 100  # nomic-embed-text has 768 dimensions
        assert isinstance(embeddings[0][0], float)
