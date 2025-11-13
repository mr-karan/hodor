# Hodor 🚪

> **Truly agentic PR reviews** powered by the [OpenHands Agent SDK](https://docs.openhands.dev/sdk/arch/overview)
> Not just LLM prompting—a full reasoning-action loop with autonomous tool orchestration

**Cross-platform (GitHub + GitLab)** • **Sandboxed workspaces** • **Multi-step planning** • **Context-aware analysis** • **2-3 minute reviews**

---

## What Makes Hodor Agentic?

Unlike simple LLM-prompting tools, Hodor runs as a **stateful agent** with a [reasoning-action loop](https://docs.openhands.dev/sdk/arch/agent):

### 🧠 Autonomous Decision Making
- **Planning Phase**: Agent analyzes the PR and creates an execution plan
- **Tool Selection**: Chooses appropriate tools (grep, file read, git diff) based on context
- **Iterative Refinement**: Observes results, adapts strategy, retries on failures
- **No Hardcoded Workflows**: Agent decides what to inspect and in what order

### 🔧 Tool Orchestration
Powered by [OpenHands tools](https://docs.openhands.dev/sdk/arch/tool-system), the agent has access to:
- **Terminal**: Execute bash commands (`git`, `grep`, test runners)
- **File Operations**: Read, search, and analyze source code
- **Planning Tools**: Break down complex reviews into subtasks
- **Task Tracker**: Maintain review checklist and findings

The agent **decides** which tools to use and when—not following a script.

### 🎯 Context-Aware Reviews
- **Skills System**: Apply repository-specific guidelines and conventions ([learn more](/sdk/guides/skill))
- **Dynamic Focus**: Agent determines which files need deep analysis vs surface checks
- **Three-Dot Diff**: Reviews only PR changes, ignoring stale branch artifacts
- **Security Analysis**: Built-in risk assessment before executing commands

### ⚡ Why This Matters

| Traditional Approach | Hodor (Agentic) |
|---------------------|-----------------|
| Single LLM call with full diff | Multi-step reasoning with tool feedback |
| Fixed prompts, no adaptation | Dynamic strategy based on observations |
| Shallow analysis (no code execution) | Can run tests, check builds, verify behavior |
| Manual tool integration | Autonomous tool selection and orchestration |
| No memory between steps | Stateful conversation with event history |

**Result**: Hodor catches bugs that require multi-step analysis—race conditions, integration issues, security vulnerabilities—not just style violations.

---

## Quick Start

### 1. Install + sync

```bash
pip install uv just
git clone https://github.com/mr-karan/hodor
cd hodor
just sync
```

### 2. Authenticate + configure

```bash
gh auth login              # GitHub (for posting reviews)
glab auth login            # GitLab (optional, for GitLab MRs)
export LLM_API_KEY=sk-your-llm-key   # or ANTHROPIC_API_KEY/OPENAI_API_KEY
```

### 3. Run a review

```bash
uv run hodor https://github.com/owner/repo/pull/123

# Auto-post the review as a comment
uv run hodor https://github.com/owner/repo/pull/123 --post

# Watch the agent work with verbose mode
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

## Feature Highlights

### Core Capabilities
- **Truly Agentic**: Multi-step reasoning with autonomous tool selection and iterative refinement
- **Multi-platform**: GitHub and GitLab (including self-hosted) with URL autodetection
- **CI-Native**: Auto-detects GitLab CI and GitHub Actions, skips redundant cloning
- **Sandboxed Execution**: Every review runs in isolated workspace with automatic cleanup

### Customization & Control
- **Skills System**: Apply repository-specific guidelines via `.hodor/skills/` directory ([see Skills](#skills-repository-specific-context))
- **Prompt Overrides**: Use `--prompt` (inline) or `--prompt-file` for custom instructions
- **Reasoning Depth**: Control extended thinking with `--reasoning-effort` (none/low/medium/high)
- **Workspace Reuse**: Use `--workspace` to cache repo state across multiple reviews

### Observability
- **Streaming Progress**: `-v/--verbose` shows real-time agent actions (tool calls, exit codes, observations)
- **Token Metrics**: Always-on tracking of input/output tokens, cache hits, costs
- **Event History**: Full conversation log for debugging and replay

---

## Skills: Repository-Specific Context

Hodor supports the [OpenHands Skills system](https://docs.openhands.dev/sdk/guides/skill) for applying custom review guidelines:

### What Are Skills?

Skills inject **repository-specific context** into the agent's system prompt:
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
- All API endpoints must have authentication checks
- User input MUST be validated and sanitized
- Never log sensitive data (passwords, tokens, PII)

## Performance
- Database queries must have indexes
- API responses should be < 200ms p95
- Avoid N+1 queries in loops

## Testing
- All new functions need unit tests
- Integration tests for API changes
- Mock external services, don't call real APIs
```

**3. Run review with skills:**
```bash
hodor <PR_URL> --workspace . --verbose
```

The agent will automatically discover and load skills from `.hodor/skills/` when reviewing the PR. Use `--verbose` to see which skills were loaded.

### Organizing Skills by Topic

Organize guidelines into separate files by domain:

```bash
# .hodor/skills/security.md
When reviewing security-related code:
- Check for timing-safe comparison of credentials
- Verify session expiration handling
- Ensure password hashing uses bcrypt/argon2

# .hodor/skills/performance.md
Performance requirements:
- API responses < 200ms p95
- No N+1 queries in loops
- Database queries must have appropriate indexes
```

All files in `.hodor/skills/` are loaded automatically on every review.

See [SKILLS.md](./docs/SKILLS.md) for detailed examples and patterns.

---

## CLI Usage Cheatsheet

```bash
# Basic console review (agent runs autonomously)
hodor https://github.com/owner/repo/pull/123

# Auto-post to the PR (requires gh/glab auth and token env vars)
hodor https://github.com/owner/repo/pull/123 --post

# GitLab MR (including self-hosted)
hodor https://gitlab.example.com/org/project/-/merge_requests/42 --post

# With repository skills for context-aware review
hodor ... --workspace . --verbose
# Agent loads skills from .hodor/skills/ automatically

# Custom model + extended reasoning for complex PRs
hodor ... \
  --model anthropic/claude-sonnet-4-5 \
  --reasoning-effort medium \
  --verbose

# Override prompt with custom instructions
hodor ... \
  --prompt "Focus on authorization bugs and SQL injection vectors." \
  --workspace /tmp/hodor \
  --verbose

# Reuse workspace for multiple PRs in same repo (faster)
hodor PR1_URL --workspace /tmp/workspace
hodor PR2_URL --workspace /tmp/workspace  # Reuses clone
```

See `hodor --help` for all flags.

**Pro Tip**: Use `--verbose` to watch the agent's reasoning process in real-time—see tool selections, command outputs, and iterative refinements.

---

## Automation Recipes

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

See [AUTOMATED_REVIEWS.md](./docs/AUTOMATED_REVIEWS.md) for advanced workflows (labels, reviewer triggers, multi-model configs).

---

## Configuration Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-sonnet-4-5` | Any OpenHands-compatible model (`openai/*`, `anthropic/*`, custom `LLM_BASE_URL`). |
| `--temperature` | Auto (0.0 for non-reasoning) | Override sampling temperature for LLM reasoning. |
| `--reasoning-effort` | `none` | Enable extended thinking for complex PRs (`low`, `medium`, `high`). Agent gets more time to plan and reason. |
| `--prompt` / `--prompt-file` | – | Append custom instructions to the agent's system prompt. |
| `--workspace` | Temp dir | Directory for repo checkout. Reuse for faster multi-PR reviews. Skills loaded from `.hodor/skills/` if present. |
| `--post` | Off | Auto-post review comment to GitHub/GitLab. Requires `gh`/`glab` auth. |
| `--verbose` | Off | Stream agent events in real-time: tool calls, bash output, observations, reasoning steps. |

**Environment Variables**

| Variable | Purpose | Required |
|----------|---------|----------|
| `LLM_API_KEY` | LLM provider authentication (recommended) | ✅ Yes (see note) |
| `ANTHROPIC_API_KEY` | Claude API key (backward compatibility) | Alternative to above |
| `OPENAI_API_KEY` | OpenAI API key (backward compatibility) | Alternative to above |
| `GITHUB_TOKEN` / `GITLAB_TOKEN` | Post comments to PRs/MRs | Only with `--post` |
| `GITLAB_HOST` | Self-hosted GitLab instance (extracted from URL automatically) | Optional |
| `LLM_BASE_URL` | Custom OpenAI-compatible gateway | Optional |

**Note**: Hodor checks API keys in order: `LLM_API_KEY` → `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` (uses first found).

**CI Detection**

Hodor auto-detects CI environments and skips cloning:
- **GitLab CI**: Uses `$CI_PROJECT_DIR` when `$GITLAB_CI=true`
- **GitHub Actions**: Uses `$GITHUB_WORKSPACE` when `$GITHUB_ACTIONS=true`

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

## Metrics & Observability

Every run prints token usage, cache hits, runtime, and estimated cost:

```
============================================================
📊 Token Usage Metrics:
  • Input tokens:       18,240
  • Output tokens:       3,102
  • Cache hits:         12,480 (68.5%)
  • Total tokens:       21,342

💰 Cost Estimate:      $0.42
⏱️  Review Time:        2m 11s
============================================================
```

**With `--verbose` flag**, see the agent's reasoning process:
```
🔧 Executing: gh pr diff 123 --no-pager
   ✓ Exit code: 0
💬 Agent planning: Breaking down review into 3 subtasks
🔧 Executing: grep -r "TODO\|FIXME" src/
   ✓ Exit code: 0
✏️  Reading file: src/auth.py
💬 Agent analyzing: Checking authentication flow
```

This visibility helps you understand:
- **What the agent is thinking** (planning, analyzing)
- **Which tools it chooses** (and why)
- **How it adapts** (retries, error handling)
- **Token efficiency** (cache hits, prompt optimization)

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
- **[PR Review Example](https://docs.openhands.dev/sdk/examples/github-workflows/pr-review)** - Official OpenHands PR review workflow

### Contributing
Found a bug? Want to add a feature? See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup and guidelines.

---

## License

MIT – see [LICENSE](./LICENSE).
