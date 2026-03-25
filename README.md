<a href="https://zerodha.tech"><img src="https://zerodha.tech/static/images/github-badge.svg" align="right" /></a>

# Hodor

> Agentic code reviewer for GitHub PRs, GitLab MRs, and local diffs. Powered by the [pi-coding-agent](https://github.com/badlogic/pi-mono) SDK.

Hodor runs as a stateful agent with tools (`bash`, `grep`, `read`, `git diff`) to autonomously analyze code changes, find bugs, and post structured reviews.

## Install

```bash
# npx (zero install, always latest)
npx @mrkaran/hodor <PR_URL>

# Global install
npm install -g @mrkaran/hodor

# Docker
docker pull ghcr.io/mr-karan/hodor:latest

# From source
git clone https://github.com/mr-karan/hodor && cd hodor
bun install && bun run build
```

## Usage

```bash
# Review a GitHub PR
hodor https://github.com/owner/repo/pull/123

# Review a GitLab MR (including self-hosted)
hodor https://gitlab.example.com/org/project/-/merge_requests/42

# Post the review as a comment
hodor <PR_URL> --post

# Review local changes (no PR URL needed)
hodor --local                              # diff against origin/main
hodor --local --diff-against HEAD~1        # diff against specific ref
hodor --local --diff-against feature-branch

# Use a different model
hodor <PR_URL> --model openai/gpt-5
hodor <PR_URL> --model bedrock/converse/anthropic.claude-sonnet-4-5-v2

# Extended reasoning for complex PRs
hodor <PR_URL> --reasoning-effort high
hodor <PR_URL> --ultrathink

# Custom review instructions
hodor <PR_URL> --prompt "Focus on SQL injection and auth bypasses"
hodor <PR_URL> --prompt-file .hodor/security-review.md

# Verbose mode (watch the agent think)
hodor <PR_URL> -v
```

**Docker:**
```bash
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  ghcr.io/mr-karan/hodor:latest \
  https://github.com/owner/repo/pull/123 --post
```

## Configuration

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-5` | LLM model (Anthropic, OpenAI, or Bedrock) |
| `--reasoning-effort` | – | Extended thinking: `low`, `medium`, `high` |
| `--ultrathink` | Off | Maximum reasoning effort |
| `--local` | Off | Review local changes (no PR URL required) |
| `--diff-against` | `origin/main` | Git ref to diff against in local mode |
| `--post` | Off | Post review comment to GitHub/GitLab |
| `--prompt` | – | Append custom instructions to the review prompt |
| `--prompt-file` | – | Replace the review prompt entirely |
| `--workspace` | Temp dir | Workspace directory (re-use for faster multi-PR reviews) |
| `--bedrock-tags` | – | JSON cost allocation tags for AWS Bedrock |
| `--prometheus-push` | – | Push metrics to a Prometheus Pushgateway URL |
| `-v, --verbose` | Off | Stream agent reasoning and tool calls |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `LLM_API_KEY` | Generic fallback (used when provider-specific key is not set) |
| `GITHUB_TOKEN` / `GITLAB_TOKEN` | Post comments to PRs/MRs (only with `--post`) |
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

See [AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md) for advanced workflows.

## Token Optimization

Hodor automatically optimizes token usage:

- **Diff embedding**: For PRs under 200KB, the diff is embedded directly in the prompt, cutting turns from ~60 to ~5.
- **Incremental reviews**: On re-runs, only reviews changes since the last hodor comment (detected via SHA markers).
- **Compaction**: SDK auto-summarizes older turns when context grows too large.

## Skills

Hodor discovers repository-specific review guidelines from `.pi/skills/` or `.hodor/skills/`. Create a skill file to enforce conventions:

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
