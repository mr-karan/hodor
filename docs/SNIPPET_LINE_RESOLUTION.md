# Spec: Snippet-Based Line Resolution

**Status:** Draft / proposed
**Owner:** @mrkaran
**Created:** 2026-06-15
**Related code:** `src/review.ts`, `src/agent.ts`, `src/gitlab.ts`, `templates/tool-review.md`, `src/types.ts`

---

## 1. Problem

Today the review agent emits the location of each finding as line numbers directly:

```ts
code_location: {
  absolute_file_path: string;
  line_range: { start: number; end: number };
}
```

These numbers are **trusted verbatim**. `src/review.ts:validateReviewOutput` only checks
`start <= end` and that the path is absolute — it never verifies that the lines actually
correspond to the code the model is talking about. The model is asked to count lines in a
raw `git diff`, which it does unreliably, especially in large hunks or files it read only
partially.

The cost of a wrong number is concrete, not cosmetic:

- **GitLab inline comments are silently dropped.** `src/gitlab.ts` anchors every inline
  draft note / discussion to `position.new_line = finding.code_location.line_range.start`
  (`createGitlabDraftNote`). When the line doesn't sit on a
  valid diff position, the GitLab API rejects it; `isPositionErrorMessage` catches the
  error and we **warn + skip** (`src/gitlab.ts:391`). A real bug the model found never
  reaches the author.
- **GitLab `suggestion:` blocks apply to the wrong lines.** `src/agent.ts:~435` builds a
  ```` ```suggestion:-0+${span}` ```` block where `span = end - start`. If the range is
  off, the one-click "apply suggestion" rewrites the wrong lines — worse than a misplaced
  comment, it's a wrong auto-fix.
- **GitHub / Gitea** post a single summary comment via `issues/{n}/comments`, so wrong
  numbers only render as misleading `file:line` text today — but the same numbers gate any
  future move to real inline review comments on those platforms.

In short: line numbers are the one part of the review payload the model is worst at
producing and we do nothing to correct them.

## 2. Insight (borrowed from alibaba/open-code-review)

OCR (Go) never lets the LLM emit line numbers. The model quotes the **verbatim code** it's
commenting on (`existing_code`), and a deterministic resolver maps that snippet back to a
line range by matching it against the diff hunks, then the full new-file content. Only if
that fails does it make a second LLM call to regenerate a tighter snippet. Reference:
`internal/diff/resolver.go` and `internal/diff/relocation.go` in that repo.

We can do the same thing, and **more simply**, because Hodor already checks out the PR
branch into a real workspace — the post-change file is sitting on disk at
`finding.code_location.absolute_file_path`. We don't need to thread `NewFileContent`
through the pipeline like OCR does; we just read the file.

## 3. Goals / Non-goals

**Goals**
- Make inline comment placement robust to model line-number errors.
- Stop silently dropping GitLab inline comments due to bad positions.
- Make `suggestion` spans land on the right lines.
- Be purely additive: zero regression when the snippet is absent or unmatchable.

**Non-goals (v1)**
- LLM relocation fallback (the OCR `relocation.go` second LLM call). Deferred to §10.
- Changing GitHub/Gitea from summary to inline comments (separate effort; this spec just
  makes their line numbers trustworthy as a prerequisite).
- Reworking the agent's diff-acquisition flow. The agentic `git diff` + `read` loop stays.

## 4. Design

### 4.1 Schema change (`src/review.ts`, `src/types.ts`)

Add an **optional** `existing_code` field to each finding:

```ts
existing_code: Type.Optional(Type.String({ minLength: 1 }))
```

Verbatim copy of the exact source lines the finding refers to (no diff `+`/`-` markers, no
fences), matching `line_range`. Optional → backward compatible: old payloads and findings
where the model omits it still validate and post exactly as today.

### 4.2 Prompt change (`templates/tool-review.md`)

In the `submit_review` payload spec, instruct the model to include `existing_code`: paste
the exact current source lines the finding covers, copied from the file/diff, unmodified.
Emphasize it must be a verbatim contiguous quote so it can be located mechanically. Keep
`line_range` required (it remains the fallback and the disambiguation hint).

### 4.3 Resolver module (`src/resolve-location.ts`, new)

Pure, no I/O beyond reading the on-disk file. Mirrors OCR's normalize + consecutive-match
logic, adapted to read the workspace file.

```
resolveLineRange(finding, workspacePath, diffText?) -> { start, end, resolved: boolean }
```

Algorithm:

1. If `existing_code` is absent → return the model's `line_range` (`resolved: false`).
2. Normalize the snippet into non-empty lines: trim whitespace, strip a leading `+`/`-`
   if the model accidentally included diff markers, drop blank lines. (Same rules as OCR's
   `normalizeLine` / `splitAndNormalize`.)
3. Read the file at `absolute_file_path` (already validated absolute + inside workspace).
   Split into lines, normalize each the same way.
4. Find every index where the normalized snippet matches a **consecutive** run of file
   lines.
5. Disambiguate:
   - 0 matches → fall back to model's `line_range` (`resolved: false`), log a warn.
   - 1 match → use it (`resolved: true`).
   - >1 match → pick the match that **overlaps the diff** (using `diffText` hunk ranges if
     available), else the one nearest the model's emitted `start`. This is a strict
     improvement over OCR, which just takes the first.
6. Return 1-indexed `{ start, end }`.

Note we deliberately match against the **working-tree file** (the new side) rather than
re-parsing the diff for primary matching: it's simpler, and inline comments anchor to
`new_line` anyway. `diffText` is only needed for the multi-match tiebreak and can be
fetched once with a single `git diff` call if not already in hand (embedded-diff mode
already has it).

### 4.4 Integration point (`src/agent.ts`)

Before the posting loop (`for (const finding of review.findings)` ~line 423), resolve each
finding's location once and use the resolved range for both the inline anchor
(`createGitlabDraftNote` `line:`) and the `suggestion:-0+span`
computation. The summary render path (`src/render.ts`) uses the same corrected numbers.

### 4.5 Fallback behavior

When a snippet can't be matched: **keep the model's `line_range` exactly as today** and log
at warn level (include path + title so it's greppable). The resolver is pure upside — it
only ever changes a location when it has a confident match. No finding is dropped or
demoted on account of resolution.

### 4.6 Rollout

Default-on and additive. No flag. `existing_code` optional in the schema; resolver runs
whenever it's present; behavior is identical to today when it's absent or unmatched. Ship
the prompt change in the same PR so the model starts providing snippets immediately.

## 5. Edge cases

- **Snippet spans lines outside the diff** (model quoted surrounding context). The
  multi-match tiebreak prefers the diff-overlapping occurrence; if the single match is
  entirely outside the diff we still use it (the template already requires overlap, and a
  precise out-of-diff location is better than a wrong in-diff guess) but log it.
- **Whitespace-only / reformatted snippet.** Normalization (trim + strip markers) absorbs
  indentation and accidental `+`/`-`. Lines that normalize to empty are skipped on both
  sides.
- **File moved/renamed.** `absolute_file_path` is the new path in the checked-out branch,
  so the file exists; no special handling.
- **Deleted lines (finding about removed code).** A snippet of deleted code won't match the
  new file → falls back to model lines. Acceptable for v1; the old-side hunk match (OCR
  tier 2) is a future refinement.
- **Duplicate code blocks** (same lines appear N times). Tiebreak by diff overlap, then
  proximity to model's `start`. Documented as best-effort.

## 6. Testing (`tests/resolve-location.test.ts`, new)

- Exact match → corrected range.
- Match with indentation/whitespace differences → normalized match.
- Snippet with stray `+`/`-` markers → stripped and matched.
- No match → returns model's range, `resolved: false`.
- Multiple matches, one overlapping diff → picks the overlapping one.
- Multiple matches, no diff info → picks nearest to model `start`.
- Absent `existing_code` → passthrough.
- Off-by-one fixture: model says lines 40–42, real code is at 45–47 → resolver corrects to
  45–47. (This is the headline case the whole spec exists for.)

Plus a `review.test.ts` case asserting the optional field validates and round-trips.

## 7. What it buys us, per platform

| Platform | Today | After |
|---|---|---|
| GitLab inline | Wrong line → API rejects → comment silently dropped | Snippet resolves to real line → comment lands |
| GitLab suggestion | `suggestion` span can rewrite wrong lines | Span derived from resolved range |
| GitHub / Gitea summary | `file:line` text can be wrong | Trustworthy numbers; unblocks future inline support |

## 8. Implementation checklist

1. `src/types.ts` — add optional `existing_code` to `ReviewFinding`.
2. `src/review.ts` — add to `REVIEW_FINDING_SCHEMA` (optional, `minLength: 1`).
3. `templates/tool-review.md` — document `existing_code` in the payload spec + add a
   one-line instruction to quote verbatim.
4. `src/resolve-location.ts` — new resolver module (normalize, match, disambiguate).
5. `src/agent.ts` — call resolver once per finding before the posting loop; use resolved
   range for inline anchor + suggestion span.
6. `tests/resolve-location.test.ts` — unit tests per §6.
7. README/docs note that findings now self-verify their location.

## 9. Risks

- **Model omits or mangles the snippet.** Mitigated by the keep-model-lines fallback; worst
  case is current behavior.
- **Added prompt tokens / output tokens** for the snippet. Small; snippets are short (the
  template already caps code chunks at ≤3 lines and ranges at 5–10 lines).
- **Multi-match wrong pick** in files with repeated blocks. Best-effort tiebreak; logged.

## 10. Future work

- **Tier-2 old-side hunk matching** for findings about deleted code (OCR `resolveFromHunk`
  old side).
- **LLM relocation fallback** (OCR `relocation.go`): when deterministic matching fails,
  one extra LLM call to regenerate a tighter snippet, then retry. Gated so it only fires on
  the residual unmatched set — bounded cost.
- **GitHub/Gitea real inline review comments**, now that line numbers are trustworthy.

## 11. Open questions

- Should `existing_code` eventually become **required** once we confirm models reliably
  produce it? (Start optional; revisit after measuring match rate via the warn logs.)
- Do we want a metric (`src/metrics.ts`) for resolution hit-rate, to quantify how often the
  model's raw line numbers were wrong? Cheap to add and would justify/measure the feature.
