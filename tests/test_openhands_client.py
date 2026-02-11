import pytest

pytest.importorskip("openhands.sdk", reason="OpenHands SDK is required to import openhands_client module")

from hodor.llm.openhands_client import describe_model, get_api_key


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    """Ensure API key environment variables do not leak between tests."""
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "model,normalized,supports_reasoning,effort",
    [
        ("gpt-5", "openai/gpt-5", True, "medium"),
        ("openai/gpt-5-2025-08-07", "openai/gpt-5-2025-08-07", True, "medium"),
        ("gpt-5-codex", "openai/gpt-5-codex", True, "medium"),
        ("openai/gpt-5-codex-latest", "openai/gpt-5-codex-latest", True, "medium"),
        ("gpt-5.1-codex", "openai/gpt-5.1-codex", True, "medium"),
        ("gpt-5.1-codex-mini", "openai/gpt-5.1-codex-mini", True, "medium"),
        ("gpt-5-mini", "openai/gpt-5-mini", True, "medium"),
        ("openai/responses/gpt-5-mini", "openai/gpt-5-mini", True, "medium"),
        ("o3-mini", "openai/o3-mini", True, "medium"),
        ("o1-preview", "openai/o1-preview", True, "medium"),
        ("anthropic/claude-sonnet-4-5", "anthropic/claude-sonnet-4-5", False, "none"),
        ("bedrock/anthropic.claude-opus-4-6-v1", "bedrock/anthropic.claude-opus-4-6-v1", False, "none"),
        ("bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0", False, "none"),
    ],
)
def test_describe_model_normalization(model, normalized, supports_reasoning, effort):
    metadata = describe_model(model)
    assert metadata.normalized == normalized
    assert metadata.supports_reasoning == supports_reasoning
    assert metadata.default_reasoning_effort == effort


def test_describe_model_requires_value():
    with pytest.raises(ValueError):
        describe_model("")


def test_get_api_key_prefers_llm_override(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-universal")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    assert get_api_key("openai/gpt-4o") == "sk-universal"


def test_get_api_key_prefers_openai_for_openai_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    assert get_api_key("openai/gpt-4o") == "sk-openai"


def test_get_api_key_prefers_anthropic_for_anthropic_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    assert get_api_key("anthropic/claude-sonnet-4-5") == "sk-anthropic"


def test_get_api_key_fallback_order_without_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    assert get_api_key() == "sk-anthropic"


def test_get_api_key_returns_placeholder_for_bedrock():
    """Bedrock uses AWS credentials, not an API key."""
    assert get_api_key("bedrock/anthropic.claude-opus-4-6-v1") == "bedrock"


def test_get_api_key_raises_when_missing(monkeypatch):
    with pytest.raises(RuntimeError):
        get_api_key("openai/gpt-4o")


class _DummyLLM:
    def __init__(self, enable_encrypted_reasoning: bool):
        self.max_output_tokens = None
        self.extra_headers = None
        self.reasoning_effort = None
        self.reasoning_summary = None
        self.model = "openai/gpt-5"
        self.litellm_extra_body = None
        self.enable_encrypted_reasoning = enable_encrypted_reasoning


def test_responses_options_respects_encrypted_flag():
    responses_options = pytest.importorskip("openhands.sdk.llm.options.responses_options")

    opts_enabled = responses_options.select_responses_options(
        _DummyLLM(True), {}, include=None, store=None
    )
    assert "reasoning.encrypted_content" in opts_enabled.get("include", [])

    opts_disabled = responses_options.select_responses_options(
        _DummyLLM(False), {}, include=None, store=None
    )
    assert "reasoning.encrypted_content" not in opts_disabled.get("include", [])
