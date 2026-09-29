# Models and providers

Hodor runs its review agent on the [Pi](https://github.com/earendil-works/pi) SDK. Pi's `pi-ai` registry supplies the model catalog: context windows, pricing, reasoning support, and prompt caching. Hodor adds three things on top: a Bedrock inference-profile syntax, an OpenRouter fallback, and an optional file for custom endpoints.

## Model strings

Pass a model as `provider/model-id`:

```bash
hodor <PR_URL> --model anthropic/claude-opus-5-5
hodor <PR_URL> --model openai/gpt-6-sol
hodor <PR_URL> --model openrouter/moonshotai/kimi-k2.6
hodor <PR_URL> --model bedrock/converse/global.anthropic.claude-opus-5-5
```

- The provider is the first path segment. The rest is the model ID, so OpenRouter IDs can contain slashes.
- `bedrock/` is an alias for Pi's `amazon-bedrock` provider. The `converse/` segment is optional.
- The default is `anthropic/claude-opus-5-5`. Its adaptive reasoning default is `xhigh` (see [Reasoning effort](#reasoning-effort)). Pass `--reasoning-effort high` or pick `anthropic/claude-sonnet-5` for cheaper reviews.
- A bare name such as `claude-opus-5-5` or `gpt-6-sol` is mapped to Anthropic or OpenAI.

## API keys

| Provider | Example model | Environment variable |
| --- | --- | --- |
| Anthropic | `anthropic/claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-6-sol` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter/moonshotai/kimi-k2.6` | `OPENROUTER_API_KEY` |
| Google Gemini | `google/gemini-2.5-pro` | `GEMINI_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| xAI | `xai/grok-4.7` | `XAI_API_KEY` |
| Groq | `groq/...` | `GROQ_API_KEY` |
| AWS Bedrock | `bedrock/converse/...` | AWS credentials (profile, env, or IAM role), or `AWS_BEARER_TOKEN_BEDROCK` |

When `LLM_API_KEY` is set, Hodor uses it instead of the provider-specific variable. For Bedrock, Pi treats `LLM_API_KEY` or `AWS_BEARER_TOKEN_BEDROCK` as a Bedrock API key (bearer token) and uses the AWS credential chain only when neither is set.

Anthropic, OpenAI, Bedrock, and OpenRouter are the tested providers. Other Pi providers are best-effort: if a model does not call tools reliably, pick another one. Try a model on your repository before you enable `--post` in CI.

## Reasoning effort

`--reasoning-effort` accepts `minimal`, `low`, `medium`, `high`, or `xhigh`. `--ultrathink` selects the maximum.

Without the flag, Hodor picks a level per review:

- Opus 4.7 and later default to `xhigh`.
- Incremental reviews and small diffs (10 files or fewer, 500 changed lines or fewer) step down to `high`.
- High-risk diffs (auth, migrations, crypto, IAM paths) and `--full` reviews keep the full level.

An explicit `--reasoning-effort` always wins.

## AWS Bedrock

Bedrock uses the standard AWS credential chain unless a Bedrock API key is set (see [API keys](#api-keys)). Set `AWS_REGION` (or `AWS_DEFAULT_REGION`) for the region you call from.

### Cross-region inference profiles

Use a system profile ID from the registry, for example `global.anthropic.claude-opus-5-5` or `us.anthropic.claude-sonnet-5`. Hodor reads the model's capabilities and pricing from the registry entry.

### Application inference profiles (cost attribution)

AWS bills cost allocation tags against a resource. For on-demand Bedrock, the only taggable resource is an application inference profile. To see Hodor's spend in Cost Explorer, create a tagged profile and pass its ARN.

A profile ARN names neither the vendor nor the model. Append `@<base-model-id>` so Hodor can look up the model's capabilities:

```bash
hodor <PR_URL> \
  --model 'bedrock/converse/arn:aws:bedrock:ap-south-1:123456789012:application-inference-profile/abc123@global.anthropic.claude-opus-5-5'
```

- The base model must exist in the installed registry. Use the ID that matches how the profile routes (for example `global.` for a profile copied from a global system profile), so reported cost matches AWS pricing.
- Without the suffix, Hodor refuses the ARN. Pi decides prompt caching, reasoning, and per-token cost by matching the model ID, so a bare ARN would silently lose all three.
- The region comes from the ARN.

### `--bedrock-tags` is not billing

`--bedrock-tags '{"team":"platform"}'` sets Bedrock `requestMetadata`. AWS uses it only to filter model invocation logs, and Cost Explorer ignores it. Use a tagged application inference profile for cost attribution.

## Registry misses

If a model is not in the installed registry, upgrade Hodor or pick a listed model.

OpenRouter is the exception, because its slugs change often. For an unknown `openrouter/...` slug, Hodor builds a conservative OpenAI-compatible descriptor that targets `https://openrouter.ai/api/v1`. Hodor turns this fallback off when a models file configures the `openrouter` provider (see below).

## Custom endpoints (`HODOR_MODELS_JSON`)

To use a self-hosted or proxied endpoint (LiteLLM, vLLM, an internal gateway), describe it in a Pi-format `models.json` and point `HODOR_MODELS_JSON` at the file:

```bash
export HODOR_MODELS_JSON=/etc/hodor/models.json
export LLM_API_KEY=...            # or the provider's own key variable
hodor <PR_URL> --model mycorp/my-model
```

```json
{
  "providers": {
    "mycorp": {
      "baseUrl": "https://llm.example.com/v1",
      "api": "openai-completions",
      "models": [
        {
          "id": "my-model",
          "reasoning": true,
          "contextWindow": 1048576,
          "maxTokens": 131072,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "compat": { "supportsStore": false, "supportsDeveloperRole": false },
          "thinkingLevelMap": { "medium": "high", "high": "high" }
        }
      ]
    }
  }
}
```

- The provider key is the `--model` prefix, and its case must match exactly (`MyCorp/my-model` for a `MyCorp` key).
- `compat` and `thinkingLevelMap` describe endpoint quirks, for example an endpoint that rejects the `store` field or Pi's default `medium` effort.
- A key that names a built-in provider (for example `openai`) overrides that provider, including its `baseUrl`.
- The full schema is Pi's `models.json`. See Pi's [model configuration docs](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md).
- When the variable is unset, Hodor reads no model configuration from disk.

### Hodor fails closed

A misconfigured file must not send reviews to a public endpoint. Hodor stops before it clones anything when:

- the path is missing, not a regular file, or unreadable;
- the file is not valid JSON or has no `providers` object;
- Pi reports a schema or provider error for the file;
- the `--model` prefix is neither a Pi provider nor a key in the file;
- the file configures `openrouter` and the model is not listed in it (no public fallback).

Error messages name the file path. They do not echo file contents.

### Treat the file as trusted code

Pi resolves `apiKey` and header values when it sends each request:

- `$NAME` or `${NAME}` reads an environment variable.
- A leading `!` runs the rest of the value as a shell command.

The file also chooses where the review and the API key are sent. Anyone who can edit it can read the review payload, receive the key, run commands as the Hodor process, and make requests to any host the runner can reach.

- Keep the file outside the repository under review. Never point `HODOR_MODELS_JSON` at a path in a PR or MR checkout.
- In CI, store the file with the runner or in a protected variable that only maintainers can change.
- Keep secrets out of literal values. Prefer `LLM_API_KEY` or a `$NAME` reference.
- Shared CI templates that lock the model should `unset HODOR_MODELS_JSON` so a project cannot redirect the reviewer.
