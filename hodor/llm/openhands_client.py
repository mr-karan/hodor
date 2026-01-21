"""OpenHands SDK client adapter for Hodor.

This module provides a clean interface to OpenHands SDK, handling:
- LLM configuration and model selection
- API key management (with backward compatibility)
- Agent creation with appropriate tool presets
- Model name normalization for OpenAI Responses API
- Encrypted reasoning configuration (disabled by default for compatibility)
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import os
import shutil
from typing import Any

from openhands.sdk import LLM
from openhands.sdk.context import Skill
from openhands.tools.preset.default import get_default_agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Describes the normalized model string plus its capabilities."""

    raw: str
    normalized: str
    supports_reasoning: bool
    default_reasoning_effort: str


@dataclass(frozen=True)
class ModelRule:
    """Rules that customize parsing for specific model families."""

    identifiers: tuple[str, ...] = ()
    predicate: Callable[[str], bool] | None = None
    provider: str | None = None
    use_responses_endpoint: bool | None = None
    supports_reasoning: bool = False
    default_reasoning_effort: str = "none"

    def matches(self, model: str) -> bool:
        if self.predicate and self.predicate(model):
            return True
        model_lower = model.lower()
        return any(identifier in model_lower for identifier in self.identifiers)


def _looks_like_openai_identifier(identifier: str) -> bool:
    if not identifier:
        return False
    return identifier.startswith("gpt") or (
        identifier.startswith("o") and len(identifier) > 1 and identifier[1].isdigit()
    )


def _extract_provider_and_base(model: str) -> tuple[str | None, str | None]:
    """Return provider plus base model name (without provider/responses prefixes)."""

    segments = [segment for segment in model.split("/") if segment]
    if not segments:
        return None, None

    provider: str | None = None
    start_index = 0
    first_segment = segments[0].lower()
    if first_segment in {"openai", "anthropic"}:
        provider = first_segment
        start_index = 1

    normalized_segments = segments[start_index:]

    if normalized_segments and normalized_segments[0].lower() == "responses":
        normalized_segments = normalized_segments[1:]

    if not normalized_segments:
        base = None
    else:
        base = normalized_segments[0].lower()

    if provider is None and base and _looks_like_openai_identifier(base):
        provider = "openai"

    return provider, base


def _matches_openai_responses_model(model: str) -> bool:
    """Detect OpenAI models that should use the Responses API."""

    provider, base = _extract_provider_and_base(model)
    if provider != "openai" or not base:
        return False

    if "codex" in base:
        return True

    response_prefixes = ("gpt-5", "gpt5", "o3", "o4")
    return base.startswith(response_prefixes)


def _respect_encrypted_reasoning_flag(llm: Any, options: dict[str, Any]) -> None:
    """Ensure encrypted reasoning keys only pass when explicitly enabled."""

    if getattr(llm, "enable_encrypted_reasoning", True):
        return

    include_values = options.get("include")
    if not include_values:
        return

    filtered = [value for value in include_values if value != "reasoning.encrypted_content"]
    if filtered:
        options["include"] = filtered
    else:
        options.pop("include", None)


# Patch OpenHands Responses options at import time so the disable flag is honored
try:  # pragma: no cover - OpenHands may be unavailable in some environments
    from openhands.sdk.llm.options import responses_options as _responses_options
except Exception:  # pragma: no cover - dependency missing
    _responses_options = None
else:
    if not getattr(_responses_options, "_hodor_responses_patched", False):
        _original_select_responses_options = _responses_options.select_responses_options

        def _hodor_select_responses_options(llm, user_kwargs: dict[str, Any], *, include, store):
            options = _original_select_responses_options(llm, user_kwargs, include=include, store=store)
            _respect_encrypted_reasoning_flag(llm, options)
            return options

        _responses_options.select_responses_options = _hodor_select_responses_options
        _responses_options._hodor_responses_patched = True


# Ordered from most specific → least specific so substring matches work reliably.
MODEL_RULES: tuple[ModelRule, ...] = (
    # Latest OpenAI reasoning families use the Responses API automatically.
    ModelRule(
        predicate=_matches_openai_responses_model,
        provider="openai",
        use_responses_endpoint=True,
        supports_reasoning=True,
        default_reasoning_effort="medium",
    ),
    # Other OpenAI reasoning families.
    ModelRule(
        identifiers=("o1",),
        provider="openai",
        use_responses_endpoint=False,
        supports_reasoning=True,
        default_reasoning_effort="medium",
    ),
)


def describe_model(model: str) -> ModelMetadata:
    """Return normalized model name plus capability flags."""

    cleaned_model = model.strip()
    if not cleaned_model:
        raise ValueError("Model name must be provided")

    rule = _match_model_rule(cleaned_model)
    normalized = _normalize_model_path(cleaned_model, rule)
    supports_reasoning = rule.supports_reasoning if rule else False
    default_reasoning_effort = rule.default_reasoning_effort if rule else "none"

    return ModelMetadata(
        raw=cleaned_model,
        normalized=normalized,
        supports_reasoning=supports_reasoning,
        default_reasoning_effort=default_reasoning_effort,
    )


def _match_model_rule(model: str) -> ModelRule | None:
    for rule in MODEL_RULES:
        if rule.matches(model):
            return rule
    return None


def _normalize_model_path(model: str, rule: ModelRule | None) -> str:
    segments = [segment for segment in model.split("/") if segment]

    if rule and rule.provider:
        segments = _ensure_provider_segment(segments, rule.provider)

    if rule and rule.use_responses_endpoint is not None:
        segments = _ensure_responses_segment(segments, rule.use_responses_endpoint)

    return "/".join(segments)


def _ensure_provider_segment(segments: list[str], provider: str) -> list[str]:
    normalized = list(segments)
    if not normalized:
        return [provider]
    if normalized[0].lower() == provider.lower():
        normalized[0] = provider
        return normalized
    return [provider, *normalized]


def _ensure_responses_segment(segments: list[str], use_responses: bool) -> list[str]:
    """Remove 'responses/' segment if present - it should not be part of the model name.

    The OpenHands SDK determines Responses API usage via uses_responses_api() method,
    not via the model name. The model name should be clean (e.g., 'openai/gpt-5').
    """
    normalized = list(segments)
    # Always remove 'responses/' if present - it's not part of the actual model name
    has_responses_segment = len(normalized) > 1 and normalized[1].lower() == "responses"
    if has_responses_segment:
        normalized.pop(1)
    return normalized


def _detect_provider(model: str) -> str | None:
    """Detect LLM provider from model name.

    Args:
        model: Model name (e.g., "anthropic/claude-sonnet-4-5" or "openai/gpt-4")

    Returns:
        Provider name ("anthropic", "openai", etc.) or None if unknown
    """
    model_lower = model.lower()

    # Try to get provider from metadata first
    try:
        metadata = describe_model(model)
        rule = _match_model_rule(metadata.raw)
        if rule and rule.provider:
            return rule.provider
    except Exception:
        # If model metadata fails, continue with string matching
        pass

    # Fall back to simple string matching
    if "anthropic" in model_lower or "claude" in model_lower:
        return "anthropic"
    elif (
        "openai" in model_lower or "gpt" in model_lower or model_lower.startswith("o1") or model_lower.startswith("o3")
    ):
        return "openai"

    return None


def get_api_key(model: str | None = None) -> str:
    """Get LLM API key from environment variables with provider-aware selection.

    Selection logic:
    1. If LLM_API_KEY is set, always use it (highest priority, universal override)
    2. If model is provided, detect provider and use provider-specific key:
       - Anthropic/Claude models → ANTHROPIC_API_KEY
       - OpenAI/GPT models → OPENAI_API_KEY
    3. If no model or provider can't be detected, fall back to any available key
       (ANTHROPIC_API_KEY first, then OPENAI_API_KEY for backward compatibility)

    Args:
        model: LLM model name (e.g., "anthropic/claude-sonnet-4-5" or "openai/gpt-4").
               If None, falls back to provider-agnostic precedence.

    Returns:
        API key string

    Raises:
        RuntimeError: If no API key is found

    Examples:
        >>> # With both keys set and OpenAI model
        >>> os.environ["ANTHROPIC_API_KEY"] = "sk-ant-xxx"
        >>> os.environ["OPENAI_API_KEY"] = "sk-xxx"
        >>> get_api_key("openai/gpt-4")  # Returns sk-xxx (OpenAI key)

        >>> # With LLM_API_KEY set (overrides everything)
        >>> os.environ["LLM_API_KEY"] = "sk-universal"
        >>> get_api_key("openai/gpt-4")  # Returns sk-universal
    """
    # Priority 1: LLM_API_KEY (universal override)
    if api_key := os.getenv("LLM_API_KEY"):
        return api_key

    # Priority 2: Provider-specific key based on model
    if model:
        provider = _detect_provider(model)
        if provider == "anthropic":
            if api_key := os.getenv("ANTHROPIC_API_KEY"):
                return api_key
        elif provider == "openai":
            if api_key := os.getenv("OPENAI_API_KEY"):
                return api_key

    # Priority 3: Fallback to any available key (backward compatibility)
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        return api_key
    if api_key := os.getenv("OPENAI_API_KEY"):
        return api_key

    raise RuntimeError("No LLM API key found. Please set one of: LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")


def create_hodor_agent(
    model: str,
    api_key: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    base_url: str | None = None,
    verbose: bool = False,
    llm_overrides: dict[str, Any] | None = None,
    skills: list[Skill] | None = None,
) -> Any:
    # Get API key (provider-aware selection based on model)
    if api_key is None:
        api_key = get_api_key(model)

    metadata = describe_model(model)
    normalized_model = metadata.normalized

    # Build LLM config
    llm_config: dict[str, Any] = {
        "model": normalized_model,
        "api_key": api_key,
        "usage_id": "hodor_agent",  # Identifies this LLM instance for usage tracking
        "drop_params": True,  # Drop unsupported API parameters automatically
    }

    # Always disable encrypted reasoning to avoid API errors
    # OpenAI models that don't support encrypted reasoning will fail with
    # "Encrypted content is not supported with this model" error if enabled
    llm_config["enable_encrypted_reasoning"] = False

    # Add base URL if provided
    if base_url:
        llm_config["base_url"] = base_url

    # Handle temperature
    thinking_active = reasoning_effort is not None or metadata.supports_reasoning

    if temperature is not None:
        llm_config["temperature"] = temperature
    elif thinking_active:
        # Reasoning models require temperature 1.0
        llm_config["temperature"] = 1.0
    else:
        # Default to deterministic for non-reasoning models
        llm_config["temperature"] = 0.0

    # Handle reasoning effort
    # Only set reasoning_effort if the model supports it or user explicitly requested it.
    # Setting "none" causes LiteLLM to fail with "Unmapped reasoning effort: none" for
    # Anthropic models, so we omit the parameter entirely for non-reasoning models.
    if reasoning_effort:
        # User explicitly requested extended thinking
        llm_config["reasoning_effort"] = reasoning_effort
    elif metadata.supports_reasoning and metadata.default_reasoning_effort != "none":
        # Model supports reasoning - use its default effort level
        llm_config["reasoning_effort"] = metadata.default_reasoning_effort
    # For non-reasoning models or "none" effort, don't set the parameter at all

    # Apply any user overrides
    if llm_overrides:
        llm_config.update(llm_overrides)

    # Configure logging
    if verbose:
        logging.getLogger("openhands").setLevel(logging.DEBUG)
        logger.info(f"Creating OpenHands agent with model: {normalized_model}")
        logger.info(f"LLM config: {llm_config}")
    else:
        logging.getLogger("openhands").setLevel(logging.WARNING)

    # Create LLM instance
    llm = LLM(**llm_config)

    # Create agent with custom tools optimized for automated code reviews
    # Use subprocess terminal instead of tmux to avoid "command too long" errors
    # that occur when environment has large variables (DIRENV_DIFF, LS_COLORS, etc.)
    from openhands.sdk.agent.agent import Agent
    from openhands.sdk.context.agent_context import AgentContext
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.context import Skill
    from openhands.sdk.tool.spec import Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.glob import GlobTool
    from openhands.tools.grep import GrepTool
    from openhands.tools.planning_file_editor import PlanningFileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    # Set terminal dimensions dynamically based on actual terminal size
    # These environment variables are inherited by the subprocess terminal
    term_size = shutil.get_terminal_size(fallback=(200, 50))
    os.environ.setdefault("COLUMNS", str(term_size.columns))
    os.environ.setdefault("LINES", str(term_size.lines))

    tools = [
        Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),  # Bash commands
        Tool(name=GrepTool.name),  # Efficient code search via ripgrep
        Tool(name=GlobTool.name),  # Pattern-based file finding
        Tool(name=PlanningFileEditorTool.name),  # Read-optimized file editor for reviews
        Tool(name=FileEditorTool.name),  # Full file editor (if modifications needed)
        Tool(name=TaskTrackerTool.name),  # Task tracking
    ]

    if verbose:
        logger.info(
            f"Configured {len(tools)} tools: terminal, grep, glob, planning_file_editor, file_editor, task_tracker"
        )

    # Create condenser for context management
    condenser = LLMSummarizingCondenser(llm=llm.model_copy(update={"usage_id": "condenser"}), max_size=80, keep_first=4)

    context = None
    if skills:
        context = AgentContext(skills=skills)
        if verbose:
            skill_names = ", ".join([s.name for s in skills])
            logger.info(f"Injecting {len(skills)} skill(s) into agent context: {skill_names}")

    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={"cli_mode": True},
        condenser=condenser,
        agent_context=context,
    )

    return agent
