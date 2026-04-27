# OpenRouter models

Hodor can run OpenRouter models exposed by the underlying `@mariozechner/pi-ai` model registry. If a newer OpenRouter slug is not in the installed registry yet, Hodor falls back to an OpenRouter-compatible model definition. Model names use the `openrouter/<provider>/<model>` form.

## Kimi K2.6 demo

```bash
export OPENROUTER_API_KEY=sk-or-your-key

hodor https://github.com/owner/repo/pull/123 \
  --model openrouter/moonshotai/kimi-k2.6 \
  --reasoning-effort medium \
  --verbose
```

For GitLab MRs:

```bash
export OPENROUTER_API_KEY=sk-or-your-key

hodor https://gitlab.example.com/org/project/-/merge_requests/42 \
  --model openrouter/moonshotai/kimi-k2.6 \
  --post
```

## CI configuration

### GitHub Actions

```yaml
env:
  HODOR_MODEL: openrouter/moonshotai/kimi-k2.6
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

steps:
  - name: Run Hodor
    run: |
      hodor "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" \
        --model "$HODOR_MODEL" \
        --post
```

### GitLab CI

```yaml
variables:
  HODOR_MODEL: "openrouter/moonshotai/kimi-k2.6"

hodor-review:
  script:
    - MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
    - hodor "$MR_URL" --model "$HODOR_MODEL" --post
```

Set `OPENROUTER_API_KEY` as a masked CI secret/variable.

## How it works

Hodor passes OpenRouter models to pi-ai as `provider: "openrouter"`. pi-ai uses the OpenAI-compatible completions API at `https://openrouter.ai/api/v1` and reads `OPENROUTER_API_KEY` for authentication.
