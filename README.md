<a href="https://zerodha.tech"><img src="https://zerodha.tech/static/images/github-badge.svg" align="right" /></a>

# Hodor

> An agentic code reviewer for GitHub and GitLab pull requests, powered by the [OpenHands Agent SDK](https://docs.openhands.dev/sdk/arch/overview).

Hodor performs automated, in-depth code reviews by running as a stateful agent with a reasoning-action loop. It can analyze code, run commands, and provide context-aware feedback.

**Features:**
- **Cross-platform**: Works with GitHub and GitLab (cloud and self-hosted).
- **Sandboxed**: Each review runs in an isolated, temporary workspace.
- **Context-aware**: Uses repository-specific "Skills" to enforce conventions.
- **CI-Native**: Optimizes execution when running in GitHub Actions or GitLab CI.
- **Observability**: Provides detailed logs, token usage, and cost estimates.

---

## How It Works

Unlike simple LLM-prompting tools, Hodor uses the OpenHands SDK to operate as an agent that can reason and act.

### Autonomous Decision Making
- **Planning**: The agent analyzes the PR and creates an execution plan.
- **Tool Selection**: It chooses appropriate tools (`grep`, file read, `git diff`) based on the context.
- **Iterative Refinement**: It observes results, adapts its strategy, and retries on failures. The agent decides what to inspect and in what order, rather than following a hardcoded workflow.

### Tool Orchestration
Powered by [OpenHands tools](https://docs.openhands.dev/sdk/arch/tool-system), the agent has access to:
- **Terminal**: Execute shell commands (`git`, `grep`, test runners).
- **File Operations**: Read, search, and analyze source code.
- **Planning Tools**: Break down complex reviews into subtasks.
- **Task Tracker**: Maintain a checklist of findings.

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
pip install uv just
git clone https://github.com/mr-karan/hodor
cd hodor
just sync
```

### 2. Configure

```bash
gh auth login              # GitHub (for posting reviews)
glab auth login            # GitLab (optional, for GitLab MRs)
export LLM_API_KEY=sk-your-llm-key   # or ANTHROPIC_API_KEY/OPENAI_API_KEY
```

### 3. Run a review

```bash
# Run a review and print the output to the console
uv run hodor https://github.com/owner/repo/pull/123

# Auto-post the review as a comment
uv run hodor https://github.com/owner/repo/pull/123 --post

# See the agent's real-time actions with verbose mode
uv run hodor https://github.com/owner/repo/pull/123 --verbose
```

**Docker Alternative:**
```bash
docker pull ghcr.io/mr-karan/hodor:latest
docker run --rm \
  -e LLM_API_KEY=$LLM_API_KEY \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  ghcr.io/mr-karan/hodor:latest \
  https://github.com/owner/repo/pull/123
```

---

## Skills: Repository-Specific Context

Hodor supports the [OpenHands Skills system](https://docs.openhands.dev/sdk/guides/skill) for applying custom review guidelines. Skills inject repository-specific context into the agent's system prompt, such as:
- Coding conventions (naming, patterns, anti-patterns)
- Security requirements (auth checks, input validation)
- Performance expectations (latency budgets, memory limits)
- Testing policies (coverage thresholds, required fixtures)

### How to Use Skills

**1. Create a skills directory:**
```bash
mkdir -p .hodor/skills
```

**2. Add a skill file (`.hodor/skills/conventions.md`):**
```markdown
# Code Review Guidelines for MyProject

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
The agent will automatically discover and load skills from the `.hodor/skills/` directory within the specified workspace.
```bash
hodor <PR_URL> --workspace . --verbose
```
Use `--verbose` to see which skills were loaded.

See [SKILLS.md](./docs/SKILLS.md) for detailed examples and patterns.

---

## CLI Usage

```bash
# Basic console review
hodor https://github.com/owner/repo/pull/123

# Auto-post to the PR (requires gh/glab auth and token env vars)
hodor https://github.com/owner/repo/pull/123 --post

# GitLab MR (including self-hosted)
hodor https://gitlab.example.com/org/project/-/merge_requests/42 --post

# Use repository skills for a context-aware review
hodor ... --workspace . --verbose
# Agent loads skills from .hodor/skills/ automatically

# Use a different model and enable extended reasoning for complex PRs
hodor ... \
  --model anthropic/claude-sonnet-4-5 \
  --reasoning-effort medium \
  --verbose

# Enable maximum reasoning effort with extended thinking (for very complex PRs)
hodor ... --ultrathink --max-iterations 1000

# Allow unlimited agent iterations for thorough reviews
hodor ... --max-iterations -1 --verbose

# Append custom instructions to the base prompt
hodor ... --prompt "Focus on authorization bugs and SQL injection vectors."

# Replace the base prompt entirely
hodor ... --prompt-file .hodor/custom-review.md

# Reuse a workspace for multiple PRs in the same repo for faster runs
hodor PR1_URL --workspace /tmp/workspace
hodor PR2_URL --workspace /tmp/workspace  # Reuses clone
```

See `hodor --help` for all flags. Use `--verbose` to watch the agent's reasoning process in real-time.

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
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
        run: |
          hodor "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" --post
```

### GitLab CI

```yaml
# .gitlab-ci.yml
hodor-review:
  image: ghcr.io/mr-karan/hodor:latest
  stage: test
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  variables:
    LLM_API_KEY: $LLM_API_KEY
    GITLAB_TOKEN: $GITLAB_TOKEN
  script:
    - hodor "${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}" --post
  allow_failure: true
```

See [AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md) for advanced workflows.

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-5-20250929` | LLM model to use. Supports any [LiteLLM model](https://docs.litellm.ai/docs/providers). |
| `--temperature` | Auto (0.0 for non-reasoning) | Override sampling temperature for LLM reasoning. |
| `--reasoning-effort` | `none` | Enable extended thinking for complex PRs (`low`, `medium`, `high`). |
| `--max-iterations` | `500` | Maximum number of agent iterations/steps. Use `-1` for unlimited execution. |
| `--ultrathink` | Off | Enable maximum reasoning effort with extended thinking budget (shortcut for `--reasoning-effort high` with 500K token budget). |
| `--prompt` | – | Append custom instructions to the base prompt. |
| `--prompt-file` | – | Replace base prompt with a custom markdown file. |
| `--workspace` | Temp dir | Directory for repo checkout. Re-use for faster multi-PR reviews. |
| `--post` | Off | Auto-post review comment to GitHub/GitLab. |
| `--verbose` | Off | Stream agent events in real-time. |

**Environment Variables**

| Variable | Purpose | Required |
|----------|---------|----------|
| `LLM_API_KEY` | LLM provider authentication (recommended) | Yes (see note) |
| `ANTHROPIC_API_KEY` | Claude API key (backward compatibility) | Alternative to above |
| `OPENAI_API_KEY` | OpenAI API key (backward compatibility) | Alternative to above |
| `GITHUB_TOKEN` / `GITLAB_TOKEN` | Post comments to PRs/MRs | Only with `--post` |
| `GITLAB_HOST` | Self-hosted GitLab instance (auto-detected) | Optional |
| `LLM_BASE_URL` | Custom OpenAI-compatible gateway | Optional |

**Note**: Hodor checks for API keys in the order: `LLM_API_KEY` → `ANTHROPIC_API_KEY` → `OPENAI_API_KEY`.

**CI Detection**

Hodor auto-detects CI environments and optimizes its execution:
- **GitLab CI**: Uses `$CI_PROJECT_DIR` as the workspace, `$CI_MERGE_REQUEST_TARGET_BRANCH_NAME` for the target branch, and `$CI_MERGE_REQUEST_DIFF_BASE_SHA` for deterministic diffs.
- **GitHub Actions**: Uses `$GITHUB_WORKSPACE` and `$GITHUB_BASE_REF` for target branch detection.

---

## Observability

Every run prints token usage, cache hits, runtime, and an estimated cost:

```
============================================================
Token Usage Metrics:
  - Input tokens:       18,240
  - Output tokens:       3,102
  - Cache hits:         12,480 (68.5%)
  - Total tokens:       21,342

Cost Estimate:      $0.42
Review Time:        2m 11s
============================================================
```

With the `--verbose` flag, you can see the agent's reasoning process:
```
Executing: gh pr diff 123 --no-pager
  ✓ Exit code: 0
Agent planning: Breaking down review into 3 subtasks
Executing: grep -r "TODO\|FIXME" src/
  ✓ Exit code: 0
Reading file: src/auth.py
Agent analyzing: Checking authentication flow
```

This helps you understand what the agent is doing, which tools it chooses, and how it adapts.

---

## Development

```bash
just sync       # Install dependencies
just check      # Format, lint, and type-check
just test-cov   # Run tests with coverage
just review URL # Review a PR
```

See [AGENTS.md](./AGENTS.md) for architecture details and contribution guidelines.

---

## Learn More

### Hodor Documentation
- **[AGENTS.md](./AGENTS.md)** - Development guidelines, OpenHands architecture, workspace setup, CI integration
- **[SKILLS.md](./docs/SKILLS.md)** - Creating repository-specific review guidelines and trigger-based skills
- **[AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md)** - Advanced CI/CD workflows, label triggers, multi-model configs

### OpenHands SDK Resources
- **[Agent Architecture](https://docs.openhands.dev/sdk/arch/agent)** - How the reasoning-action loop works
- **[Skills System](https://docs.openhands.dev/sdk/guides/skill)** - Creating and applying context to agents
- **[Tool System](https://docs.openhands.dev/sdk/arch/tool-system)** - Built-in tools and custom tool development
- **[Workspace Management](https://docs.openhands.dev/sdk/arch/workspace)** - Sandboxed execution environments
- **[PR Review Example](https://docs.openhands.dev/sdk/guides/github-workflows/pr-review)** - Official OpenHands PR review workflow

### Contributing
Found a bug? Want to add a feature? See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

---

## License

MIT – see [LICENSE](./LICENSE).