<a href="https://zerodha.tech"><img src="https://zerodha.tech/static/images/github-badge.svg" align="right" /></a>

# Hodor

> Agentic code reviewer for GitHub PRs, GitLab MRs, and local diffs. Powered by the [pi-coding-agent](https://github.com/badlogic/pi-mono) SDK.

Hodor runs as a stateful agent with tools (`bash`, `grep`, `read`, `git diff`) to autonomously analyze code changes, find bugs, and post structured reviews.

## Install

```bash
# Just run it (zero install, always latest)
npx @mrkaran/hodor <PR_URL>

# Or install globally
npm install -g @mrkaran/hodor
```

Docker images are also available at `ghcr.io/mr-karan/hodor:latest` for CI environments.

## Setup

```bash
# Set an API key for your LLM provider
export ANTHROPIC_API_KEY=sk-...   # Anthropic (default)
export OPENAI_API_KEY=sk-...      # OpenAI
export AWS_PROFILE=default        # AWS Bedrock (no API key needed)

# For posting reviews as comments
gh auth login                     # GitHub
glab auth login                   # GitLab
```

## Usage

```bash
# Review a GitHub PR
npx @mrkaran/hodor https://github.com/owner/repo/pull/123

# Review a GitLab MR (including self-hosted)
npx @mrkaran/hodor https://gitlab.example.com/org/project/-/merge_requests/42

# Post the review as a PR/MR comment
npx @mrkaran/hodor <PR_URL> --post

# Review local changes (no PR URL needed)
npx @mrkaran/hodor --local                         # diff against origin/main
npx @mrkaran/hodor --local --diff-against HEAD~1   # diff against specific ref

# Use a different model
npx @mrkaran/hodor <PR_URL> --model openai/gpt-5
npx @mrkaran/hodor <PR_URL> --model bedrock/converse/anthropic.claude-sonnet-4-5-v2

# Extended reasoning for complex PRs
npx @mrkaran/hodor <PR_URL> --reasoning-effort high

# Custom review instructions
npx @mrkaran/hodor <PR_URL> --prompt "Focus on SQL injection and auth bypasses"

# Verbose mode (watch the agent think)
npx @mrkaran/hodor <PR_URL> -v
```

> If you installed globally with `npm install -g`, replace `npx @mrkaran/hodor` with `hodor`.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-5-20250929` | LLM model (Anthropic, OpenAI, or Bedrock) |
| `--reasoning-effort` | – | Extended thinking: `low`, `medium`, `high` |
| `--ultrathink` | Off | Maximum reasoning effort |
| `--local` | Off | Review local git changes (no PR URL required) |
| `--diff-against` | `origin/main` | Git ref to diff against in `--local` mode |
| `--post` | Off | Post review as a comment on the PR/MR |
| `--prompt` | – | Append custom instructions to the review prompt |
| `--prompt-file` | – | Use a custom prompt file |
| `--workspace` | Temp dir | Workspace directory (reuse for faster multi-PR reviews) |
| `--bedrock-tags` | – | JSON cost allocation tags for AWS Bedrock |
| `--prometheus-push` | – | Push review metrics to a Prometheus Pushgateway |
| `-v, --verbose` | Off | Stream agent reasoning and tool calls |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `LLM_API_KEY` | Generic fallback (when provider-specific key is not set) |
| `GITHUB_TOKEN` / `GITLAB_TOKEN` | Post comments to PRs/MRs (with `--post`) |
| `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` | AWS Bedrock auth (no API key needed) |

## CI/CD

### GitHub Actions

```yaml
name: Hodor Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    container: ghcr.io/mr-karan/hodor:latest
    steps:
      - name: Run Hodor
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          hodor "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" --post
```

### GitLab CI

```yaml
include:
  - project: 'commons/gitlab-templates'
    ref: master
    file: '/hodor/.gitlab-ci-template.yml'

hodor-review:
  extends: .hodor-review
```

See [AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md) for advanced CI workflows.

## Token Optimization

Hodor automatically optimizes token usage:

- **Diff embedding**: For PRs under 200KB, the diff is embedded directly in the prompt, cutting agent turns from ~60 to ~5.
- **Incremental reviews**: On re-runs, only reviews changes since the last hodor comment (detected via SHA markers in posted comments).
- **Compaction**: SDK auto-summarizes older conversation turns when context grows too large.

## Skills

Hodor discovers repository-specific review guidelines from `.pi/skills/` or `.hodor/skills/`:

```bash
mkdir -p .hodor/skills/review-guidelines
```

```markdown
# .hodor/skills/review-guidelines/SKILL.md
---
name: review-guidelines
description: Security and performance review checklist.
---

- All API endpoints must have authentication checks.
- Database queries must use parameterized statements.
- API responses should be < 200ms p95.
```

Skills are loaded automatically during reviews. See [SKILLS.md](./docs/SKILLS.md) for details.

## Development

```bash
bun install          # Install dependencies
bun run build        # Build
bun run test         # Run tests
bun run dev -- <url> # Run from source
```

## License

MIT
