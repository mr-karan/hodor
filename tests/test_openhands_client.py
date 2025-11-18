import pytest

pytest.importorskip("openhands.sdk", reason="OpenHands SDK is required to import openhands_client module")

from hodor.llm.openhands_client import describe_model, get_api_key


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    """Ensure API key environment variables do not leak between tests."""
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
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


def test_get_api_key_raises_when_missing(monkeypatch):
    with pytest.raises(RuntimeError):
        get_api_key("openai/gpt-4o")
