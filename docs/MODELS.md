# Models and providers

Pass models as:

```bash
--model provider/model-id
```

Examples:

```bash
hodor <PR_URL> --model anthropic/claude-sonnet-4-5-20250929
hodor <PR_URL> --model openai/gpt-5
hodor <PR_URL> --model openrouter/moonshotai/kimi-k2.6
hodor <PR_URL> --model bedrock/converse/anthropic.claude-sonnet-4-5-v2
```

The provider is the first path segment. Everything after the first slash is the model ID, so OpenRouter IDs can contain more slashes.

`bedrock/...` is accepted as an alias for the pi-ai Bedrock provider.

## API keys

Set the key for the provider you select:

| Provider | Example model | Environment variable |
| --- | --- | --- |
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-5` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter/moonshotai/kimi-k2.6` | `OPENROUTER_API_KEY` |
| Google Gemini | `google/gemini-2.5-pro` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| xAI | `xai/grok-4` | `XAI_API_KEY` |
| Groq | `groq/...` | `GROQ_API_KEY` |
| AWS Bedrock | `bedrock/converse/...` | AWS credentials |

`LLM_API_KEY` can be used as a generic fallback for non-Bedrock providers.

## Recommended use

- Start with the default Anthropic model.
- Use `--reasoning-effort low|medium|high` when the selected provider supports reasoning controls.
- Test a model on your repository before enabling `--post` in CI.
- If a model does not call tools reliably, use a different model.

## Registry misses

Most providers are resolved through the installed `pi-ai` registry. If a model is missing, update Hodor or choose a model present in the registry.

OpenRouter is the exception: Hodor can create a conservative OpenAI-compatible model definition for unknown `openrouter/...` slugs because OpenRouter model names change frequently.

Upstream provider list: <https://github.com/badlogic/pi-mono/tree/main/packages/ai#supported-providers>
