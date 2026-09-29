# Repository Guidelines

Hodor is an AI code reviewer for GitHub PRs, GitLab MRs, Gitea/Forgejo PRs, and local diffs. It is TypeScript, built with tsup, runs on Bun (Node 22 compatible), and uses the Pi coding-agent SDK (`@earendil-works/pi-coding-agent`, `@earendil-works/pi-ai`) as its agent runtime. The primary deployment is GitLab CI through a shared template.

## Project map

The full module table is in `README.md` under Architecture. The paths you touch most:

- `src/agent.ts`: review orchestration. Preflight, workspace, Pi session, the `submit_review` tool, recovery, metrics.
- `src/model.ts`, `src/models-json.ts`: model strings, Bedrock ARN models, adaptive reasoning, `HODOR_MODELS_JSON`.
- `src/review-diff.ts`: full, incremental, and snapshot diff bases.
- `src/publisher.ts`, `src/gitlab.ts`: GitLab inline notes, rolling summary, discussion reconciliation.
- `src/resolve-location.ts`: snippet-based line resolution.
- `templates/`: default review profile and review task prompt.
- `tests/`: vitest, `*.test.ts`.

## Pi SDK integration

- **Session:** `createAgentSession()` with tools `git_diff`, `read`, `grep`, `find`, `ls`, and `submit_review` (`terminate: true`). All but `submit_review` come from `src/review-tools.ts`: `git_diff` serves the precomputed diff, and the others replace Pi's built-ins with versions confined to `git ls-files`. There is no `bash`. `--tiny-diff-fast-path` exposes only `submit_review`.
- **Models:** `createModelRuntime()` in `src/models-json.ts` wraps `ModelRuntime.create()`. Registry models come from `modelRuntime.getModel()`. Bedrock application inference profile ARNs are built by `buildBedrockArnModel()` from an `@<base-model-id>` registry entry.
- **Settings:** `SettingsManager.inMemory()` enables compaction, sets `cacheWarming: "off"` (Pi defaults to streaming warming, which costs money in one-shot CI), and bounds retries (`retry.maxAgentDelayMs`).
- **Bedrock request fields:** `wrapBedrockStream()` wraps `session.agent.streamFunction` to add `requestMetadata` and OpenAI-on-Bedrock reasoning. The instance property is `streamFunction`; `streamFn` is only the constructor option.
- **Errors:** the SDK stores failed turns in `session.state.errorMessage`. Check it after `session.prompt()`.
- **Metrics:** read usage from `session.getSessionStats()`. `session.messages` is the projected context and omits compacted and retried turns.
- **Events:** `session.subscribe()` drives verbose progress, including `auto_retry_*` and `compaction_*` events.

When you upgrade Pi, read the release notes for every version in between, then check each API Hodor calls against the new `.d.ts` files. Do not reach SDK internals through `as unknown as` casts: a rename then fails silently instead of at compile time. That is how the Bedrock hook stayed dead from 0.7.0 to 0.7.7.

## Review process invariants

- Hodor embeds diffs under 200KB in the prompt. Otherwise the agent runs `git --no-pager diff` with a three-dot range (`origin/<target>...HEAD`).
- Only changed code is in scope. Pre-existing issues count only when the change breaks them.
- Re-runs are incremental from the last `<!-- hodor:sha:... -->` marker. After a force-push or rebase, the review compares the last reviewed snapshot with the current HEAD. `--full` disables both.
- In GitLab CI, compute the merge base from the target branch. Use `CI_MERGE_REQUEST_DIFF_BASE_SHA` only as a fallback.
- CI runs (`$GITLAB_CI`, `$GITHUB_ACTIONS`) reuse the existing checkout. Local runs clone with `gh` or `glab`, and `--workspace` reuses a clone.

## Build, test, and development

```bash
bun install
bun run typecheck
bun run test                             # vitest
bun run build                            # tsup to dist/
bun run dev -- <url>                     # run from source
bun run eval:list                        # validate eval cases
bun run eval -- --model <provider/model> # paid: runs real model reviews
```

`just check` runs typecheck and tests. `docker buildx build --load -t hodor:local .` builds the CI image.

## Coding style

- Strict TypeScript (`noUnusedLocals`, `noUnusedParameters`), ESM only.
- camelCase for functions and variables, PascalCase for types, UPPER_SNAKE_CASE for constants.
- No `any`, no non-null `!`, no `as` casts except when narrowing untyped external data behind a guard.
- Validate at boundaries (CLI input, env vars, files, platform APIs). Error messages name the input but never echo secrets or file contents.

## Testing

- Tests live in `tests/`, one `*.test.ts` per module.
- Prefer real objects over mocks. Use real temp files and real Pi objects (`Agent`, `ModelRuntime`) where it is cheap; see `tests/bedrock-stream.test.ts` and `tests/models-json.test.ts`.
- Mock only hard boundaries: `src/utils/exec.ts` (the `gh`/`glab`/`git` subprocesses) and the LLM stream. Use `vi.mock()` at module level.
- A test that fakes the SDK cannot catch SDK drift. For SDK hooks, add at least one test against the real SDK type.

## Commits and pull requests

- Conventional commits, matching history: `feat(gitlab): ...`, `fix(bedrock): ...`, `chore(deps): ...`, `chore(release): X.Y.Z`. Subject under 72 characters; the body explains why.
- Reference issues with `Fixes #123`.
- External PRs are merged with a merge commit.

## Releases

1. Bump `version` in `package.json` and commit `chore(release): X.Y.Z` on `main`.
2. Push `main`, then push tag `vX.Y.Z`. The tag runs `docker-release.yml` (`ghcr.io/mr-karan/hodor:X.Y.Z`) and `npm-publish.yml` (`@mrkaran/hodor`).
3. After both succeed, create the GitHub Release: `gh release create vX.Y.Z --title vX.Y.Z --generate-notes --latest`. No workflow does this.
4. Bump the pinned image in the shared GitLab template (`commons/gitlab-templates`, `hodor/`) and run its `test-template.mjs`.

## Security

- Keep API keys and tokens in environment variables. Never commit them. `.env` is gitignored.
- The agent has no shell, and its file tools only reach tracked files in the checkout (`src/review-tools.ts`). Do not add a tool that runs commands, reads untracked paths, or inherits `process.env`; a prompt-injected diff controls what the model asks for.
- `HODOR_MODELS_JSON` is trusted configuration: Pi runs `!command` values and interpolates `$ENV` in it. Never load it from the checkout under review.
- Hodor markers (`hodor:sha`, cache markers, prior-review text, finding discussions) are machine state only when the bot account wrote the note. `partitionNotesByProvenance()` in `src/provenance.ts` is the only place that decides this, by numeric user id. New code that reads a marker must take `TrustedHodorNote` values from it.
