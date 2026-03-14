<a href="https://zerodha.tech"><img src="https://zerodha.tech/static/images/github-badge.svg" align="right" /></a>

# Hodor

> An agentic code reviewer for GitHub and GitLab pull requests, powered by the [pi-coding-agent](https://github.com/badlogic/pi-mono) SDK.

Hodor performs automated, in-depth code reviews by running as a stateful agent with a reasoning-action loop. It can analyze code, run commands, and provide context-aware feedback.

**Features:**
- **Cross-platform**: Works with GitHub and GitLab (cloud and self-hosted).
- **Sandboxed**: Each review runs in an isolated, temporary workspace.
- **Context-aware**: Uses repository-specific "Skills" to enforce conventions.
- **CI-Native**: Optimizes execution when running in GitHub Actions or GitLab CI.
- **Observability**: Provides detailed logs, token usage, and cost estimates.

---

## How It Works

Unlike simple LLM-prompting tools, Hodor uses the pi-coding-agent SDK to operate as an agent that can reason and act.

### Autonomous Decision Making
- **Planning**: The agent analyzes the PR and creates an execution plan.
- **Tool Selection**: It chooses appropriate tools (`grep`, file read, `git diff`) based on the context.
- **Iterative Refinement**: It observes results, adapts its strategy, and retries on failures. The agent decides what to inspect and in what order, rather than following a hardcoded workflow.

### Tool Orchestration
The agent has access to:
- **Bash**: Execute shell commands (`git`, `grep`, test runners).
- **File Operations**: Read, search, and analyze source code.
- **Grep / Find / Ls**: Fast file discovery and pattern matching.

The agent decides which tools to use and when, not just following a script.

### Comparison

| Traditional Static Analysis | Hodor (Agentic Review) |
|-----------------------------|------------------------|
| Single LLM call with full diff | Multi-step reasoning with tool feedback |
| Fixed prompts, no adaptation | Dynamic strategy based on observations |
| Shallow analysis (no code execution) | Can run tests, check builds, and verify behavior |
| Manual tool integration | Autonomous tool selection and orchestration |
| No memory between steps | Stateful execution with event history |

**Result**: Hodor can identify issues that require multi-step analysis, such as race conditions, integration problems, and security vulnerabilities, going beyond simple style checks.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/mr-karan/hodor
cd hodor
bun install
bun run build
```

### 2. Configure

```bash
gh auth login              # GitHub (for posting reviews)
glab auth login            # GitLab (optional, for GitLab MRs)
export ANTHROPIC_API_KEY=sk-your-key   # or OPENAI_API_KEY

# Or reuse an existing pi login (OAuth-backed subscriptions like OpenAI Codex)
pi                         # then run /login once if you have not already

# Or use AWS Bedrock (no API key needed, uses AWS credentials)
export AWS_PROFILE=main    # or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
```

### 3. Run a review

```bash
# Run a review and print the output to the console
bun run dist/cli.js https://github.com/owner/repo/pull/123

# Auto-post the review as a comment
bun run dist/cli.js https://github.com/owner/repo/pull/123 --post

# See the agent's real-time actions with verbose mode
bun run dist/cli.js https://github.com/owner/repo/pull/123 --verbose
```

**Docker Alternative:**
```bash
docker pull ghcr.io/mr-karan/hodor:latest
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  ghcr.io/mr-karan/hodor:latest \
  https://github.com/owner/repo/pull/123
```

---

## Skills: Repository-Specific Context

Hodor uses the upstream pi-coding-agent skills format (`agentskills.io`). Skills are discovered from `.pi/skills` or `.hodor/skills` and loaded on demand, such as:
- Coding conventions (naming, patterns, anti-patterns)
- Security requirements (auth checks, input validation)
- Performance expectations (latency budgets, memory limits)
- Testing policies (coverage thresholds, required fixtures)

### How to Use Skills

**1. Create a skill directory:**
```bash
mkdir -p .pi/skills/review-guidelines
```

**2. Add a skill file (`.pi/skills/review-guidelines/SKILL.md`):**
```markdown
---
name: review-guidelines
description: Project-specific review checklist for security, performance, and tests.
---

## Security
- All API endpoints must have authentication checks.
- User input MUST be validated and sanitized.
- Never log sensitive data (passwords, tokens, PII).

## Performance
- Database queries must have indexes.
- API responses should be < 200ms p95.
- Avoid N+1 queries in loops.
```

**3. Run review with skills:**
The agent will automatically discover skills from `.pi/skills/` or `.hodor/skills/` in the reviewed repository and read matching skills on demand.
```bash
bun run dist/cli.js <PR_URL> --workspace . --verbose
```
Use `--verbose` to see discovered skills and diagnostics.

See [SKILLS.md](./docs/SKILLS.md) for detailed examples and patterns.

---

## CLI Usage

```bash
# Basic console review
bun run dist/cli.js https://github.com/owner/repo/pull/123

# Auto-post to the PR (requires gh/glab auth)
bun run dist/cli.js https://github.com/owner/repo/pull/123 --post

# GitLab MR (including self-hosted)
bun run dist/cli.js https://gitlab.example.com/org/project/-/merge_requests/42 --post

# Use a different model and enable extended reasoning
bun run dist/cli.js <PR_URL> \
  --model anthropic/claude-sonnet-4-5 \
  --reasoning-effort medium \
  --verbose

# Use AWS Bedrock
bun run dist/cli.js <PR_URL> --model bedrock/converse/anthropic.claude-sonnet-4-5-v2

# Use your ChatGPT Plus/Pro Codex subscription via pi auth storage
bun run dist/cli.js <PR_URL> --model openai-codex/gpt-5.4

# Enable maximum reasoning effort (maps to xhigh where supported)
bun run dist/cli.js <PR_URL> --ultrathink

# Append custom instructions to the base prompt
bun run dist/cli.js <PR_URL> --prompt "Focus on authorization bugs and SQL injection vectors."

# Replace the base prompt entirely
bun run dist/cli.js <PR_URL> --prompt-file .hodor/custom-review.md

# Reuse a workspace for multiple PRs in the same repo for faster runs
bun run dist/cli.js PR1_URL --workspace /tmp/workspace
bun run dist/cli.js PR2_URL --workspace /tmp/workspace  # Reuses clone
```

See `bun run dist/cli.js --help` for all flags. Use `--verbose` to watch the agent's reasoning process in real-time.

---

## Automation

### GitHub Actions

```yaml
# .github/workflows/hodor.yml
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
          bun run /app/dist/cli.js "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" --post
```

### GitLab CI

```yaml
# .gitlab-ci.yml
include:
  - project: 'commons/gitlab-templates'
    ref: master
    file: '/hodor/.gitlab-ci-template.yml'

hodor-review:
  extends: .hodor-review
```

See [AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md) for advanced workflows.

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-5-20250929` | LLM model to use. Supports Anthropic, OpenAI, OpenAI Codex via pi auth, and AWS Bedrock. |
| `--reasoning-effort` | None | Enable extended thinking (`low`, `medium`, `high`, `xhigh`). |
| `--ultrathink` | Off | Maximum reasoning effort with extended thinking budget (`xhigh` where supported). |
| `--prompt` | – | Append custom instructions to the base prompt. |
| `--prompt-file` | – | Replace base prompt with a custom markdown file. |
| `--workspace` | Temp dir | Directory for repo checkout. Re-use for faster multi-PR reviews. |
| `--post` | Off | Auto-post review comment to GitHub/GitLab. |
| `--verbose` | Off | Stream agent events in real-time. |

**Environment Variables**

| Variable | Purpose | Required |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | For Anthropic models |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI models |
| `LLM_API_KEY` | Generic fallback API key (used when provider-specific key is not set) | Optional |
| `GITHUB_TOKEN` / `GITLAB_TOKEN` | Post comments to PRs/MRs | Only with `--post` |
| `GITLAB_PRIVATE_TOKEN` | Alternative GitLab token name (checked after `GITLAB_TOKEN`) | Optional |
| `GITLAB_HOST` | Self-hosted GitLab instance (auto-detected from MR URL) | Optional |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS Bedrock authentication | For `bedrock/` models |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | AWS region for Bedrock (e.g., `ap-south-1`) | For `bedrock/` models |
| `AWS_PROFILE` | AWS profile name (alternative to access keys) | For `bedrock/` models |

**Note**: Hodor automatically selects the provider-specific key for the requested model (`ANTHROPIC_API_KEY` for Claude, `OPENAI_API_KEY` for GPT). It also reuses pi auth storage from `~/.pi/agent/auth.json` (or `$PI_CODING_AGENT_DIR/auth.json`) for OAuth-backed providers such as `openai-codex/*`. For `bedrock/` models, no API key is needed; authentication uses AWS credentials (environment variables, profiles, or IAM roles).

**CI Detection**

Hodor auto-detects CI environments and optimizes its execution:
- **GitLab CI**: Uses `$CI_PROJECT_DIR` as the workspace, `$CI_MERGE_REQUEST_TARGET_BRANCH_NAME` for the target branch, and `$CI_MERGE_REQUEST_DIFF_BASE_SHA` for deterministic diffs.
- **GitHub Actions**: Uses `$GITHUB_WORKSPACE` and `$GITHUB_BASE_REF` for target branch detection.

---

## Observability

Every run prints token usage, cache hits, runtime, and an estimated cost:

```
**Review Metrics** | 3 turns, 8 tool calls | 2m 5s
Tokens: in `1.0K`, cached `900`, out `80` | Cost: `$1.23`
```

With the `--verbose` flag, you can see the agent's reasoning process in real-time, including tool calls, bash commands, and file reads.

---

## Development

```bash
bun install          # Install dependencies
bun run build        # Build with tsup
bun run test         # Run tests
bun run test:watch   # Watch mode
bun run typecheck    # Type-check
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
| `src/gitlab.ts` | GitLab API via `glab` CLI |
| `src/github.ts` | GitHub API via `gh` CLI |
| `src/render.ts` | JSON review output → markdown rendering |
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

MIT – see [LICENSE](./LICENSE).
