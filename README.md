<a href="https://zerodha.tech"><img src="https://zerodha.tech/static/images/github-badge.svg" align="right" /></a>

# Hodor

> Agentic code reviewer for GitHub PRs, GitLab MRs, Gitea/Forgejo PRs, and local diffs. Powered by the [pi-coding-agent](https://github.com/badlogic/pi-mono) SDK.

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
export ANTHROPIC_API_KEY=sk-...     # Anthropic (default)
export OPENAI_API_KEY=sk-...        # OpenAI
export OPENROUTER_API_KEY=sk-or-... # OpenRouter (e.g., Kimi K2.6)
export AWS_PROFILE=default          # AWS Bedrock (no API key needed)

# For posting reviews as comments
gh auth login                     # GitHub
glab auth login                   # GitLab

# For Gitea/Forgejo private repos or posting comments
export GITEA_TOKEN=your-token     # or FORGEJO_TOKEN
```

## Usage

```bash
# Review a GitHub PR
npx @mrkaran/hodor https://github.com/owner/repo/pull/123

# Review a GitLab MR (including self-hosted)
npx @mrkaran/hodor https://gitlab.example.com/org/project/-/merge_requests/42

# Review a Gitea or Forgejo PR
npx @mrkaran/hodor https://git.example.com/owner/repo/pulls/123

# Post the review as a PR/MR comment
npx @mrkaran/hodor <PR_URL> --post

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

## Local Mode

Review local git changes without a PR URL. Useful for pre-push reviews, Bitbucket PRs, or any git repo.

```bash
# Review uncommitted changes against origin/main (default)
npx @mrkaran/hodor --local

# Review against a specific branch or ref
npx @mrkaran/hodor --local --diff-against develop
npx @mrkaran/hodor --local --diff-against HEAD~3

# Review a feature branch against main
git checkout feature-branch
npx @mrkaran/hodor --local --diff-against origin/main

# Use a specific workspace directory
npx @mrkaran/hodor --local --workspace /path/to/repo

# Combine with other flags
npx @mrkaran/hodor --local --diff-against origin/main --model openai/gpt-5 -v
```

Local mode:
- Includes **uncommitted changes** (staged + unstaged), not just commits
- Auto-resolves to the **git repo root** (works from subdirectories)
- Skips PR metadata fetching and workspace cloning
- `--post` is disabled (no remote to post to)

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-5-20250929` | LLM model as `provider/model-id`. Recommended: Anthropic, OpenAI, Bedrock, OpenRouter. Other pi-ai providers (e.g., Mistral, Gemini, xAI, Groq) are best-effort. See [docs/MODELS.md](./docs/MODELS.md). |
| `--reasoning-effort` | – | Extended thinking: `low`, `medium`, `high` |
| `--ultrathink` | Off | Maximum reasoning effort |
| `--local` | Off | Review local git changes (no PR URL required) |
| `--diff-against` | `origin/main` | Git ref to diff against in `--local` mode |
| `--post` | Off | Post review as a comment on the PR/MR |
| `--review-style` | `hybrid` | GitLab posting style: `summary`, `inline`, or `hybrid` |
| `--code-quality` | – | Write a CodeClimate JSON artifact for GitLab code quality reports |
| `--commit-status` | Off | Post a pass/fail commit status to the GitLab MR head SHA |
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
| `OPENROUTER_API_KEY` | OpenRouter API key (for `openrouter/...` models, e.g. `openrouter/moonshotai/kimi-k2.6`) |
| Provider-specific keys | For best-effort pi-ai providers, use the env var pi-ai expects (e.g. `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `GROQ_API_KEY`) |
| `LLM_API_KEY` | Generic fallback (when provider-specific key is not set) |
| `GITHUB_TOKEN` / `GITLAB_TOKEN` | Post comments to GitHub PRs / GitLab MRs (with `--post`) |
| `GITEA_TOKEN` / `FORGEJO_TOKEN` | Read private repos and post comments on Gitea/Forgejo PRs |
| `GITEA_HOST` / `FORGEJO_HOST` | Hostname for Gitea/Forgejo when not inferable from a full PR URL |
| `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` | AWS Bedrock auth (no API key needed) |

See [docs/MODELS.md](./docs/MODELS.md) for the full model/provider matrix and [docs/OPENROUTER.md](./docs/OPENROUTER.md) for an end-to-end Kimi K2.6 example.

## Gitea / Forgejo

Hodor supports Gitea and Forgejo pull request URLs in this format:

```bash
npx @mrkaran/hodor https://git.example.com/owner/repo/pulls/123
```

For public repositories, metadata fetching may work without a token. Set `GITEA_TOKEN` or `FORGEJO_TOKEN` for private repositories, higher API limits, and `--post`:

```bash
export GITEA_TOKEN=your-token
npx @mrkaran/hodor https://git.example.com/owner/repo/pulls/123 --post
```

Fork PRs are checked out from the PR source repository when Gitea exposes the source clone URL. If the source branch or fork has been deleted, checkout will fail with a workspace error.

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
# .gitlab-ci.yml
workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"

hodor-review:
  stage: test
  image:
    name: ghcr.io/mr-karan/hodor:latest
    entrypoint: [""]
  variables:
    HODOR_MODEL: "anthropic/claude-sonnet-4-5-20250929"
  before_script:
    - glab auth login --hostname $CI_SERVER_HOST --token $GITLAB_TOKEN
  script:
    - MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
    - bun run /app/dist/cli.js "$MR_URL" --model "$HODOR_MODEL" --post --code-quality gl-code-quality-report.json --commit-status
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
    when: always
  allow_failure: true
  timeout: 15m
```

This posts inline comments on the diff, a summary note, a pass/fail commit status, and a code quality report visible in the MR widget.

See [AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md) for advanced workflows.

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

---

## Architecture

Hodor is written in TypeScript and runs on [Bun](https://bun.sh). Key components:

| Module | Purpose |
|--------|---------|
| `src/cli.ts` | Commander.js CLI entry point |
| `src/agent.ts` | Core review orchestration, URL parsing, comment posting |
| `src/workspace.ts` | CI detection, repo cloning, branch checkout |
| `src/prompt.ts` | Prompt template building and interpolation |
| `src/model.ts` | Model string parsing, API key resolution |
| `src/gitlab.ts` | GitLab API via `glab` CLI (comments, inline notes, draft notes, commit status) |
| `src/github.ts` | GitHub API via `gh` CLI |
| `src/render.ts` | JSON review output → markdown rendering |
| `src/codequality.ts` | CodeClimate JSON artifact for GitLab code quality widget |
| `src/metrics.ts` | Token usage and cost formatting |
| `templates/` | Review prompt template (JSON schema) |

The agent runtime is provided by [`@mariozechner/pi-coding-agent`](https://github.com/badlogic/pi-mono) with [`@mariozechner/pi-ai`](https://github.com/badlogic/pi-mono) for LLM access. The agent session gets read-only tools (bash, read, grep, find, ls) and a review prompt, then autonomously analyzes the PR.

---

## Learn More

### Hodor Documentation
- **[SKILLS.md](./docs/SKILLS.md)** - Creating repository-specific review guidelines
- **[AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md)** - Advanced CI/CD workflows

### Contributing
Found a bug? Want to add a feature? Open an issue at https://github.com/mr-karan/hodor/issues.

---
## License

MIT
