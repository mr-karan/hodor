# OpenRouter

Use OpenRouter models with:

```bash
export OPENROUTER_API_KEY=sk-or-...

hodor <PR_URL> --model openrouter/moonshotai/kimi-k2.6
```

The model string format is `openrouter/<provider>/<model>`. Everything after `openrouter/` is passed as the OpenRouter model slug.

## GitHub Actions

```yaml
env:
  HODOR_MODEL: openrouter/moonshotai/kimi-k2.6
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

steps:
  - name: Run Hodor
    env:
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    run: |
      hodor "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" \
        --model "$HODOR_MODEL" \
        --post
```

## GitLab CI

```yaml
variables:
  HODOR_MODEL: openrouter/moonshotai/kimi-k2.6

hodor-review:
  script:
    - MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
    - hodor "$MR_URL" --model "$HODOR_MODEL" --post
```

Set `OPENROUTER_API_KEY` as a masked secret or CI variable.

## Notes

- Unknown OpenRouter slugs are allowed. Hodor falls back to an OpenAI-compatible definition.
- Review quality and tool use vary by model. Test before posting automatically.
- Use `--reasoning-effort medium` or `high` only if the chosen model supports it.
