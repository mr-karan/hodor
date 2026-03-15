# GitLab Advanced Integration Roadmap

> Codebase: TypeScript (v0.3.4) on Bun/Node 22, pi-coding-agent SDK
> Status: Planning

## Problem

Hodor's GitLab integration is functional but surface-level. The entire review — every finding, summary, verdict — posts as **one monolithic MR comment** via `glab api ... --method POST --field body=...`. No inline annotations on specific diff lines. No suggestion blocks. No threading. No approval integration.

GitLab's API has features purpose-built for code review bots that Hodor doesn't use.

## Current Architecture (v0.3.x — TypeScript)

```
CLI (cli.ts --post)
  → reviewPr() (agent.ts)
    → pi-coding-agent SDK creates agent session
    → Agent analyzes diff, calls submit_review tool
    → Returns ReviewOutput (typed: findings[], overall_correctness, overall_explanation)
  → renderMarkdown(review) (render.ts) → flat markdown string
  → postReviewComment({ prUrl, reviewText }) (agent.ts)
    → postGitlabMrComment() (gitlab.ts) → glab api POST .../notes --field body=...
```

### Key files

| File | Role |
|------|------|
| `src/gitlab.ts` | GitLab API via `glab` CLI — fetch MR info, post comment, summarize notes |
| `src/agent.ts` | Review loop — `reviewPr()` + `postReviewComment()` + `submit_review` tool |
| `src/cli.ts` | Commander.js CLI — `--post`, `--model`, `--verbose`, etc. |
| `src/review.ts` | TypeBox schema for `submit_review` tool + validation |
| `src/render.ts` | `renderMarkdown(review)` — structured ReviewOutput → markdown |
| `src/types.ts` | Shared interfaces: `ReviewOutput`, `ReviewFinding`, `Platform`, etc. |
| `src/prompt.ts` | Template interpolation for review prompt |
| `src/workspace.ts` | CI detection, repo cloning, branch checkout |
| `templates/tool-review.md` | Agent prompt template — diff workflow + `submit_review` schema |

### Key differences from old Python codebase

1. **TypeScript + Bun** (not Python + OpenHands)
2. **pi-coding-agent SDK** (not openhands.sdk) — `createAgentSession()`, `session.prompt()`, `session.subscribe()`
3. **Tool-based review submission** — agent calls `submit_review` tool (not text output parsed as JSON)
4. **`glab api` for all GitLab API calls** (not python-gitlab library)
5. **Structured `ReviewOutput` already flows through the pipeline** — no flattening-to-string problem
6. **`render.ts` already does path stripping** in `formatLocation()` — handles `/builds/`, `/workspace/`, `/tmp/hodor-review-*/`

### Critical insight: `glab api` vs python-gitlab

The old Python code used `python-gitlab` (native GitLab API client). The new code uses `glab api` (CLI wrapper for REST calls). This means:
- **No SDK objects** like `mr.discussions.create()` — we shell out to `glab api` with `--method POST --field key=value`
- **JSON payloads** need to be passed as `--input -` (stdin) or `--raw-field` for complex nested objects (like `position`)
- All new GitLab API features (#1-#7) will be implemented as `glab api` REST calls

## Roadmap (9 Items)

### #1 — Inline Diff Comments (Discussions API)

**Impact**: Highest | **Effort**: Medium

Each finding becomes a pinpointed DiffNote on the exact diff line.

**API endpoint**: `POST /projects/:id/merge_requests/:iid/discussions`

**Implementation plan** (in `src/gitlab.ts`):
- New function: `getGitlabMrDiffRefs(owner, repo, mrNumber, host?)` — fetches MR, extracts `diff_refs` (base_sha, head_sha, start_sha)
- New function: `postGitlabInlineComment(owner, repo, mrNumber, body, filePath, line, diffRefs, host?)` — creates a DiffNote discussion via `glab api` with JSON body containing `position` object
- The `position` object requires: `base_sha`, `head_sha`, `start_sha`, `position_type: "text"`, `old_path`, `new_path`, `new_line`
- Invalid positions (line not in diff) → graceful warning + skip, not crash

**Data already available**: `ReviewFinding.code_location` has `absolute_file_path` and `line_range`. `render.ts:formatLocation()` already strips workspace prefixes.

**glab api pattern** for nested JSON:
```bash
echo '{"body":"...","position":{"base_sha":"...","head_sha":"...","start_sha":"...","position_type":"text","new_path":"file.ts","old_path":"file.ts","new_line":42}}' | \
  glab api projects/:encoded/merge_requests/:iid/discussions --method POST --input -
```

---

### #2 — Draft Notes + Bulk Publish

**Impact**: High | **Effort**: Low | **Prerequisites**: #1

Batch all inline comments, publish atomically (1 email notification instead of N).

**API endpoints**:
- `POST /projects/:id/merge_requests/:iid/draft_notes` — create each draft
- `POST /projects/:id/merge_requests/:iid/draft_notes/bulk_publish` — publish all at once

**Implementation**: Same `glab api` pattern as #1. Draft notes accept the same `position` object.

---

### #3 — Suggestion Blocks

**Impact**: High | **Effort**: Low | **Prerequisites**: #1

GitLab's native one-click-apply code suggestions:

````markdown
```suggestion:-0+0
fixed_code_here
```
````

**Implementation plan**:
- Add optional `suggestion?: string` field to `ReviewFinding` in `types.ts`
- Update `REVIEW_FINDING_SCHEMA` in `review.ts` to accept optional `suggestion` field
- Update `templates/tool-review.md` prompt to tell the LLM to populate `suggestion` when it can offer a fix
- In the inline comment formatter, wrap `finding.suggestion` in GitLab syntax

---

### #4 — Summary Note + Inline Findings (Hybrid Posting)

**Impact**: High | **Effort**: Medium | **Prerequisites**: #1, #2, #3

The composition layer.

**New CLI flag**: `--review-style summary|inline|hybrid` (default: `hybrid`)

| Style | Summary Note | Inline Comments |
|-------|-------------|-----------------|
| `summary` | Current behavior | None |
| `inline` | None | Per-finding DiffNotes |
| `hybrid` | Compact table + verdict | Per-finding DiffNotes |

**Implementation plan** (in `src/agent.ts`):
- New function: `postReviewStructured(opts)` — handles the full GitLab inline posting flow:
  1. Cleanup old Hodor comments (#9)
  2. Fetch fresh diff_refs (#1)
  3. For each finding: relativize path → create draft note with position (#2)
  4. Post summary note if hybrid/summary
  5. Bulk publish draft notes (#2)
  6. Post commit status if enabled (#6)
- `postReviewComment()` remains as backward-compat path for GitHub + `summary` style
- New function in `render.ts`: `renderSummaryMarkdown(review)` — compact table format for hybrid mode

**Comment identification**: All Hodor comments include `<!-- hodor-review -->` marker.

**Graceful degradation**:
- If diff_refs unavailable → fall back to summary
- If inline comment fails → log warning, continue with others
- Each step independently try/caught

---

### #5 — Code Quality Artifact

**Impact**: Medium | **Effort**: Low

Emit `gl-code-quality-report.json` in CodeClimate format. GitLab renders findings inline in MR diff natively.

**New CLI flag**: `--code-quality <path>`

**Implementation plan**:
- New file: `src/codequality.ts` — `formatCodeQualityReport(review, workspacePrefix?)` → CodeClimate JSON string
- Priority mapping: P0→critical, P1→major, P2→minor, P3→info
- Fingerprint: MD5 of `title:path:line`
- `.gitlab-ci.yml` update: `artifacts: reports: codequality: gl-code-quality-report.json`

---

### #6 — Commit Status (Pass/Fail Widget)

**Impact**: Medium | **Effort**: Low

Post a pipeline-style status check on MR head SHA. Shows pass/fail in MR widget.

**New CLI flag**: `--commit-status / --no-commit-status`

**API endpoint**: `POST /projects/:id/statuses/:sha`

**Implementation plan** (in `src/gitlab.ts`):
- New function: `postGitlabCommitStatus(owner, repo, sha, state, host?, opts?)` via `glab api`
- State logic: no P0/P1 findings → `success`, any P0/P1 → `failed`

---

### #7 — Discussion Resolution on Re-review

**Impact**: Medium | **Effort**: Medium | **Prerequisites**: #1, #4

When Hodor re-reviews after a push, resolve discussions where the issue was fixed.

**Implementation plan** (in `src/gitlab.ts`):
- `listHodorDiscussions(owner, repo, mrNumber, host?)` — list discussions with `<!-- hodor-review -->` marker
- `resolveGitlabDiscussions(owner, repo, mrNumber, discussionIds, host?)` — resolve by ID via `glab api PUT`
- Integration: before posting new review, compare old discussions against new findings

---

### #8 — @hodor Mention Trigger (GitLab Duo) — DEFERRED

Not feasible as Hodor code today. GitLab Duo Agent Platform does not support custom third-party agents.

**Alternatives**: CI pipeline trigger (already works), ChatOps `/hodor review`, webhook-based trigger.

**Revisit when**: GitLab opens Duo Agent Platform to third parties.

---

### #9 — Previous Comment Cleanup

**Impact**: Medium | **Effort**: Low

Before posting new review, find and delete old Hodor comments.

**Implementation plan** (in `src/gitlab.ts`):
- `cleanupHodorComments(owner, repo, mrNumber, host?, marker?)` — list notes, filter by marker, delete matching
- Marker: `<!-- hodor-review -->` (invisible in rendered markdown)
- Called at start of `postReviewStructured()` before posting anything new

---

## Dependency Graph

```
#1 (Inline Comments) ──→ #4 (Hybrid Posting) ──→ integration complete
#2 (Draft Notes)     ──┘         │
#3 (Suggestions)     ────────────┘
                                  │
#9 (Cleanup)         ─────────────┤
#7 (Resolution)      ─────────────┘

#5 (Code Quality)    ← independent
#6 (Commit Status)   ← independent
#8 (GitLab Duo)      ← DEFERRED
```

## Implementation Order

| Step | Items | Files Modified |
|------|-------|----------------|
| 1 | #1 Inline comments, #9 Cleanup | `src/gitlab.ts` |
| 2 | #2 Draft notes | `src/gitlab.ts` |
| 3 | #3 Suggestions | `src/types.ts`, `src/review.ts`, `templates/tool-review.md` |
| 4 | #6 Commit status | `src/gitlab.ts` |
| 5 | #5 Code quality | New `src/codequality.ts` |
| 6 | #4 Hybrid posting | `src/agent.ts`, `src/render.ts`, `src/cli.ts` |
| 7 | #7 Discussion resolution | `src/gitlab.ts`, `src/agent.ts` |
| 8 | Tests | `tests/gitlab.test.ts`, `tests/render.test.ts`, etc. |

## Key Design Decisions

1. **`glab api` for everything**: No new npm dependencies — all GitLab API calls via `glab api` with JSON stdin
2. **Hybrid as default**: `--review-style hybrid` when using `--post` on GitLab
3. **Graceful degradation**: If diff_refs unavailable or inline fails → fall back to summary
4. **Marker-based identification**: `<!-- hodor-review -->` in all comments (not bot username check)
5. **Dedicated suggestion field**: In submit_review schema, not parsed from body
6. **Path stripping**: Reuse existing `render.ts:formatLocation()` logic
7. **Fresh diff_refs**: Re-fetch MR before posting (don't reuse stale metadata)
8. **Independent error handling**: Each posting step try/caught — partial failure doesn't crash
9. **GitHub unchanged**: New features are GitLab-only for now
10. **#8 deferred**: GitLab Duo Agent Platform not available to third parties
