# Repository Guidelines

## Project Structure & Module Organization
- TypeScript project built with [tsup](https://tsup.egoist.dev/), runs on [Bun](https://bun.sh)
- `src/cli.ts` is the Commander.js CLI entry point
- `src/agent.ts` manages the review loop using the pi-coding-agent SDK
- `src/workspace.ts` handles repo cloning, CI detection, and PR branch checkout for GitHub and GitLab
- `src/model.ts` handles model string parsing and API key resolution
- `src/prompt.ts` builds review prompts from templates in `templates/`
- `src/gitlab.ts` wraps GitLab API via `glab` CLI
- `src/github.ts` wraps GitHub API via `gh` CLI
- `src/metrics.ts` formats token usage and cost metrics
- `src/types.ts` contains shared interfaces (Platform, ParsedPrUrl, MrMetadata, etc.)
- `src/utils/exec.ts` wraps child_process for subprocess execution
- `src/utils/logger.ts` provides structured logging (chalk-colored stderr)
- Tests live in `tests/` using vitest

## Pi-Coding-Agent SDK Architecture
Hodor uses [`@earendil-works/pi-coding-agent`](https://github.com/earendil-works/pi) for the agent runtime:
- **Session Creation**: `createAgentSession()` configures the model, tools, and settings
- **Tools**: Read-only tools — `createReadTool`, `createBashTool`, `createGrepTool`, `createFindTool`, `createLsTool`
- **Model Resolution**: `ModelRegistry` from `@earendil-works/pi-coding-agent` for known providers; custom model objects for Bedrock ARNs
- **Event Monitoring**: `session.subscribe()` for real-time progress logging in verbose mode
- **Error Handling**: Check `session.state.error` after completion (SDK swallows errors into state)

### Workspace Setup
The workspace module (`workspace.ts`) handles three scenarios:
1. **CI Environment**: Auto-detects GitLab CI (`$GITLAB_CI=true`) or GitHub Actions (`$GITHUB_ACTIONS=true`) and skips cloning
2. **Local Review**: Clones repo via `gh` or `glab` CLI tools and checks out PR branch
3. **Reusable Workspace**: If `--workspace` specified, reuses existing clone and fetches latest changes

**Self-hosted GitLab Support**: Extracts host from URL (e.g., `gitlab.example.com`) and uses it for cloning.

### PR Review Process
The agent follows a strict process to avoid reviewing the entire codebase:
1. **Get Diff**: Run `git diff origin/target...HEAD` to see ONLY changed files
2. **Review Changed Files**: ONLY analyze files that appear in the diff
3. **Analyze Each Change**: Look for bugs in new/modified code, ignore pre-existing code unless PR breaks it

**Critical Rules**:
- Use three-dot diff (`origin/main...HEAD`) to show only PR changes
- Disable git pager with `git --no-pager` to avoid interactive prompts
- For GitLab CI, calculate the current merge base from the target branch; use `CI_MERGE_REQUEST_DIFF_BASE_SHA` only as a fallback

## Build, Test, and Development Commands
```bash
bun install          # Install dependencies
bun run build        # Build with tsup
bun run test         # Run tests with vitest
bun run test:watch   # Watch mode
bun run typecheck    # TypeScript type checking
bun run dev -- <url> # Run from source
```

Docker: `docker build -t hodor .` for local testing.

## Coding Style & Naming Conventions
TypeScript with strict mode enabled (`noUnusedLocals`, `noUnusedParameters`). ESM modules only. camelCase for functions/variables, PascalCase for types/interfaces, UPPER_SNAKE_CASE for constants. Target Node 22 / Bun 1.x.

## Testing Guidelines
Vitest is configured via `vitest.config.ts`. Tests in `tests/` directory, files named `*.test.ts`. Mock external dependencies (exec, SDK) at module level with `vi.mock()`.

## Commit & Pull Request Guidelines
Follow the existing history: short, imperative subjects under 72 chars plus optional body. Reference issues (`Fixes #123`) when applicable.

## Security & Configuration
Store API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`, `GITLAB_TOKEN`) in environment variables. Never commit credentials. The `.env` file is gitignored.

### CI/CD Environment Variables
When running in GitLab CI or GitHub Actions, Hodor automatically detects:
- **GitLab CI**: `$GITLAB_CI`, `$CI_PROJECT_DIR`, `$CI_PROJECT_PATH`, `$CI_MERGE_REQUEST_IID`, `$CI_MERGE_REQUEST_DIFF_BASE_SHA`
- **GitHub Actions**: `$GITHUB_ACTIONS`, `$GITHUB_WORKSPACE`, `$GITHUB_REPOSITORY`, `$GITHUB_BASE_REF`

If detected, workspace setup skips cloning and uses the existing checkout.
