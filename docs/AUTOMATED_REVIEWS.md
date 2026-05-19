# Automated reviews

This page has copy-pasteable CI examples for running Hodor on pull requests and merge requests.

## GitHub Actions

Minimal workflow:

```yaml
name: Hodor Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write
  issues: write

jobs:
  review:
    runs-on: ubuntu-latest
    container: ghcr.io/mr-karan/hodor:latest
    steps:
      - name: Run Hodor
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          bun run /app/dist/cli.js \
            "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" \
            --post
```

Notes:

- `actions/checkout` is not required. Hodor clones the repository when the Actions workspace is not a valid checkout.
- If you do use `actions/checkout` with a container job, make sure the container user can write to the GitHub runner temp directories. Otherwise checkout can fail before Hodor starts.
- Set `GH_TOKEN` as well as `GITHUB_TOKEN` for GitHub CLI compatibility.

Manual dispatch workflow:

```yaml
name: Manual Hodor Review

on:
  workflow_dispatch:
    inputs:
      pr_number:
        description: PR number to review
        required: true
        type: string

permissions:
  contents: read
  pull-requests: write
  issues: write

jobs:
  review:
    runs-on: ubuntu-latest
    container: ghcr.io/mr-karan/hodor:latest
    steps:
      - name: Run Hodor
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          bun run /app/dist/cli.js \
            "https://github.com/${{ github.repository }}/pull/${{ inputs.pr_number }}" \
            --post
```

## GitLab CI

```yaml
workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

hodor-review:
  stage: test
  image:
    name: ghcr.io/mr-karan/hodor:latest
    entrypoint: [""]
  variables:
    HODOR_MODEL: anthropic/claude-sonnet-4-5-20250929
  before_script:
    - glab auth login --hostname "$CI_SERVER_HOST" --token "$GITLAB_TOKEN"
  script:
    - MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
    - bun run /app/dist/cli.js "$MR_URL" --model "$HODOR_MODEL" --post --commit-status --code-quality gl-code-quality-report.json
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
    when: always
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  allow_failure: true
  timeout: 15m
```

Required variables:

- `ANTHROPIC_API_KEY`, or the API key for the model provider you use.
- `GITLAB_TOKEN` with API access if `--post`, `--commit-status`, or GitLab metadata fetches are needed.

## Posting modes

GitLab supports three review styles:

| Style | Summary note | Inline comments | Use when |
| --- | --- | --- | --- |
| `hybrid` | Yes | Yes | Default. Good for most MRs. |
| `inline` | No | Yes | You only want diff comments. |
| `summary` | Yes | No | Inline comments are not wanted or not supported. |

Example:

```bash
hodor "$MR_URL" --post --review-style inline
```

GitHub posting currently uses a summary PR comment.

## Metrics

Push metrics at the end of a run:

```bash
hodor "$MR_OR_PR_URL" --prometheus-push "$METRICS_PUSH_URL"
```

`METRICS_PUSH_URL` can be a Prometheus Pushgateway base URL or a VictoriaMetrics `/api/v1/import/prometheus` endpoint. Push failures are logged as warnings and do not fail the review.

Dashboard JSON lives in [`docs/grafana/`](./grafana/).

## Skills

Put repo-specific review instructions under `.agents/skills` in the repository being reviewed. See [SKILLS.md](./SKILLS.md).

## Common failures

### `No API key found`

Set the provider key in CI, for example `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`.

### `fatal: not a git repository`

Upgrade to `v0.6.0` or newer. Hodor now validates CI workspaces before using them and falls back to cloning when the workspace is empty.

### GitHub `actions/checkout` fails with `EACCES` in a container

Do not run checkout for the Hodor job unless you need it. Hodor can clone the repository itself.

### Review is too slow

- Avoid running on draft PRs.
- Skip docs-only changes in your workflow rules.
- Use `--reasoning-effort low` for routine changes.
- Keep large generated files out of diffs.
