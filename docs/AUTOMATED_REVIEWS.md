# Automated Code Reviews with Hodor

Hodor is optimized for **automated, READ-ONLY code reviews** in CI/CD pipelines. This document explains how to set up and use Hodor for continuous code quality monitoring.

## Overview

Hodor provides AI-powered code reviews that:
- Run automatically on every PR/MR (or on-demand)
- Use efficient search tools (grep, find, bash, file read)
- Post review comments directly to GitHub/GitLab
- Focus on bugs, security, and performance (not style)
- Support repo-specific guidelines via skills system
- Work with self-hosted GitLab instances

## Key Features for CI/CD

### 1. Enhanced Tooling for Fast Reviews

Hodor uses specialized tools for efficient code analysis:

| Tool | Purpose | Example Use |
|------|---------|-------------|
| **grep** | Pattern search across codebase | Find all `TODO`, `FIXME`, null checks, error patterns |
| **find** | File discovery | Find all `*.test.js`, `**/*.py`, config files |
| **read** | File viewer | View code with line numbers |
| **bash** | Git and CLI commands | Run `git diff`, `git log`, language-specific linters |

These tools are **much faster** than having the LLM script search operations, reducing review time and token usage.

### 2. READ-ONLY by Design

Hodor is designed for **automated reviews without human intervention**:

- Workspace is a fresh clone (agent can't push changes)
- Environment is isolated (temporary directory, cleaned up after)
- Safe for untrusted PRs (no risk of malicious code execution)

### 3. Repository-Specific Guidelines

Use the **skills system** to encode your team's standards:

```bash
# In your repository root
.pi/skills/<name>/SKILL.md      # Upstream skills format (recommended)
.pi/skills/*.md                 # Flat markdown skills (supported)
.hodor/skills/...               # Also supported (same layout)
```

See [SKILLS.md](./SKILLS.md) for detailed documentation.

## Setup for GitHub Actions

### Quick Start

1. **Add Secrets** to your GitHub repository:
   - `ANTHROPIC_API_KEY`: Your Claude API key (or `OPENAI_API_KEY` for OpenAI)
   - `GITHUB_TOKEN` is automatically provided by GitHub Actions

2. **Create the workflow file**:

```yaml
# .github/workflows/hodor-review.yml
name: AI Code Review

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

## Setup for GitLab CI

### Quick Start (Recommended)

Use the shared template from `commons/gitlab-templates`:

```yaml
# .gitlab-ci.yml
include:
  - project: 'commons/gitlab-templates'
    ref: master
    file: '/hodor/.gitlab-ci-template.yml'

hodor-review:
  extends: .hodor-review
```

This handles everything: Docker image, model mapping, API keys, and IMDS credential fetching for Bedrock.

### Manual Setup

If you can't use the shared template:

1. **Set CI/CD Variables** in GitLab Settings > CI/CD > Variables:
   - `ANTHROPIC_API_KEY`: Your Claude API key
   - `GITLAB_TOKEN`: GitLab access token with `api` scope

2. **Add to your `.gitlab-ci.yml`**:

```yaml
hodor-review:
  image:
    name: ghcr.io/mr-karan/hodor:latest
    entrypoint: [""]
  stage: test
  before_script:
    - glab auth login --hostname $CI_SERVER_HOST --token $GITLAB_TOKEN
  script:
    - MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
    - hodor "$MR_URL" --model anthropic/claude-sonnet-4-5 --post --code-quality gl-code-quality-report.json --commit-status
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
    when: always
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  allow_failure: true
  timeout: 15m
```

## GitLab Advanced Integration

When posting to GitLab MRs, Hodor uses several API features for a rich review experience.

### Inline Diff Comments

By default (`--review-style hybrid`), Hodor posts each finding as an inline comment pinned to the exact line in the MR diff view. Reviewers see annotations right where the issue is, not buried in a wall-of-text comment.

If a finding's line can't be mapped to the diff (e.g., the line wasn't changed), Hodor skips that inline comment gracefully and includes it in the summary instead.

### Suggestion Blocks

When the agent can suggest a specific fix, it includes it as a GitLab suggestion block. This renders an "Apply suggestion" button in the MR — the author clicks it, and GitLab commits the fix automatically. No copy-paste needed.

### Review Styles

Control how Hodor posts reviews with `--review-style`:

| Style | Summary Note | Inline Comments | Best For |
|-------|-------------|-----------------|----------|
| `hybrid` (default) | Compact table with counts and verdict | Per-finding on diff lines | Most teams |
| `inline` | None | Per-finding on diff lines | Minimal noise |
| `summary` | Full markdown (legacy behavior) | None | Compatibility |

### Draft Notes (Batch Publishing)

Hodor creates all inline comments as draft notes first, then publishes them atomically. This means the MR author receives **one notification** for the entire review, not one per finding.

### Commit Status

With `--commit-status`, Hodor posts a pass/fail status check on the MR head SHA:
- **Success**: No P0 or P1 (critical) findings
- **Failed**: One or more P0/P1 findings

This integrates with GitLab's merge checks — you can configure your project to block merging when Hodor flags critical issues.

### Code Quality Artifact

With `--code-quality gl-code-quality-report.json`, Hodor writes a CodeClimate-format JSON file. When declared as a CI artifact, GitLab renders the findings inline in the MR diff widget natively — no API calls required.

```yaml
artifacts:
  reports:
    codequality: gl-code-quality-report.json
  when: always
```

### Comment Cleanup

On re-review (e.g., after a force-push), Hodor automatically deletes its previous comments before posting new ones. This keeps the MR timeline clean — no stale review noise.

## Configuration Options

### Model Selection

Choose the right model for your needs:

```yaml
# Fast and cost-effective (default)
HODOR_MODEL: anthropic/claude-sonnet-4-5

# Most thorough reviews
HODOR_MODEL: anthropic/claude-opus-4-6

# AWS Bedrock (no API key needed, uses IAM role)
HODOR_MODEL: bedrock/converse/anthropic.claude-sonnet-4-5-v2

# OpenAI
HODOR_MODEL: openai/gpt-5
```

### Custom Prompts

Override the default review prompt:

```bash
# Inline prompt
bun run /app/dist/cli.js $MR_URL --prompt "Focus on security issues only..."

# From file (create your own custom prompt file)
bun run /app/dist/cli.js $MR_URL --prompt-file custom-prompt.txt
```

## Cost Optimization

### Token Usage

| Review Type | Typical Tokens | Approx. Cost |
|-------------|---------------|--------------|
| Small PR (<5 files) | 5K-15K | $0.03-$0.09 |
| Medium PR (5-15 files) | 15K-40K | $0.09-$0.24 |
| Large PR (15+ files) | 40K-100K | $0.24-$0.60 |

### Optimization Tips

1. **Focus Reviews**: Review only changed files (default behavior), use custom prompts for specific concerns, skip trivial changes with CI rules
2. **Smart Triggers**: Run on critical PRs only (use labels), skip draft PRs, run once per PR (not on every commit)

Example GitLab CI rule:
```yaml
rules:
  - if: $CI_MERGE_REQUEST_LABELS !~ /skip-review/
    when: on_success
  - if: $CI_MERGE_REQUEST_DRAFT == "true"
    when: never
```

## Security Considerations

### Safe for Untrusted Code

- **Isolated Environment**: Each review runs in a fresh, temporary workspace
- **No Write Access**: Agent can read code but can't push changes
- **Automatic Cleanup**: Workspace is deleted after review

### Protecting Secrets

1. **Use CI/CD variables**: Store sensitive data in GitHub Secrets / GitLab CI Variables
2. **Limit token scope**: GITLAB_TOKEN only needs `api` scope
3. **Rotate tokens**: Periodically rotate API keys

## Troubleshooting

### "No API key found"

**Cause**: Missing CI/CD variable
**Solution**: Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in CI/CD settings

### "Failed to clone repository"

**Cause**: Missing or invalid GITLAB_TOKEN
**Solution**: Generate token with `api` scope in GitLab Settings > Access Tokens

### "Review finds no issues on obviously buggy code"

**Solution**: Try:
1. Use `--verbose` to see what agent is checking
2. Add repo-specific guidelines in `.pi/skills/` or `.hodor/skills/`
3. Increase `--reasoning-effort high`
4. Use `--ultrathink` for maximum depth

### Job Timeout

The default timeout is 30m. For very large MRs:
```yaml
hodor-review:
  extends: .hodor-review
  timeout: 60m
```

## Best Practices

### DO:
- Run automated reviews on all non-trivial PRs
- Use `.pi/skills/` or `.hodor/skills/` for project-specific guidelines
- Post reviews automatically (use `--post` flag)
- Enable verbose logging initially (debug issues)
- Treat reviews as suggestions (not blockers)

### DON'T:
- Don't block PRs on review results (use as advisory)
- Don't review trivial changes (docs, typos, formatting)
- Don't use expensive models for small PRs
- Don't commit secrets in skill files

## Support

- **Documentation**: See [README.md](../README.md) and [SKILLS.md](./SKILLS.md)
- **Issues**: Report bugs at https://github.com/mr-karan/hodor/issues
