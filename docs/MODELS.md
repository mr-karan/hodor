# Models and providers

Hodor can use the same model providers exposed by the upstream `@mariozechner/pi-ai` package. In practice, you set the provider's API key and pass a model as `provider/model-id`.

## Quick examples

```bash
# Recommended default-style usage
export ANTHROPIC_API_KEY=sk-ant-...
hodor <PR_URL> --model anthropic/claude-sonnet-4-5-20250929

# OpenAI
export OPENAI_API_KEY=sk-...
hodor <PR_URL> --model openai/gpt-5

# OpenRouter / Kimi K2.6
export OPENROUTER_API_KEY=sk-or-...
hodor <PR_URL> --model openrouter/moonshotai/kimi-k2.6

# Best-effort pi-ai provider
export MISTRAL_API_KEY=...
hodor <PR_URL> --model mistral/mistral-large-latest
```

## Model name format

Use:

```bash
--model provider/model-id
```

The provider is the first path segment (`anthropic`, `openai`, `openrouter`, `mistral`, `google`, `xai`, etc.). Everything after the first `/` is the model ID, so OpenRouter model IDs can themselves contain slashes:

```bash
--model openrouter/moonshotai/kimi-k2.6
```

`bedrock/...` is accepted as a friendly alias for pi-ai's `amazon-bedrock` provider.

## Support tiers

- **Recommended/tested**: the default Anthropic model and models explicitly shown in Hodor examples. These are the models we expect users to start with.
- **Registry-backed/best-effort**: any provider/model exposed by the installed `pi-ai` registry. These can work without Hodor-specific code changes, but review quality, tool-calling behavior, reasoning support, and rate limits vary by provider/model.

This means “pi-ai supports it” does not guarantee it is a good PR reviewer. If a model fails to call tools reliably or produces weak reviews, choose a recommended model instead.

For the full provider/model matrix, see the upstream pi-ai supported providers documentation: <https://github.com/badlogic/pi-mono/tree/main/packages/ai#supported-providers>

## Authentication

Set the provider-specific API key expected by pi-ai before running Hodor:

| Provider | Example model | Environment variable |
|----------|---------------|----------------------|
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-5` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter/moonshotai/kimi-k2.6` | `OPENROUTER_API_KEY` |
| Google Gemini | `google/gemini-2.5-pro` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| xAI | `xai/grok-4` | `XAI_API_KEY` |
| Groq | `groq/...` | `GROQ_API_KEY` |
| AWS Bedrock | `bedrock/converse/...` | AWS credentials, not an LLM API key |

Examples:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export MISTRAL_API_KEY=...
export GEMINI_API_KEY=...
export XAI_API_KEY=...
```

You can also set `LLM_API_KEY` to override authentication for the selected provider only. This is useful in CI when you want one generic secret name:

```bash
export LLM_API_KEY=...
hodor <PR_URL> --model openrouter/moonshotai/kimi-k2.6
```

AWS Bedrock uses AWS credentials instead of an LLM API key:

```bash
export AWS_PROFILE=main
hodor <PR_URL> --model bedrock/converse/anthropic.claude-sonnet-4-5-v2
```

## Registry misses

Hodor normally requires the selected model to exist in the installed pi-ai registry. The exception is OpenRouter: because OpenRouter model slugs change frequently, Hodor falls back to a conservative OpenAI-compatible model definition for unknown `openrouter/...` slugs.

If a non-OpenRouter model is missing, update Hodor's `@mariozechner/pi-ai` dependency or choose a model that exists in the installed registry.

## Choosing a model

For routine PR review, start with the default Anthropic model. Use OpenRouter/Kimi or other providers when you specifically want to test cost, latency, availability, or model behavior. For automated CI, prefer a model you have tested against your codebase before enabling `--post`.
