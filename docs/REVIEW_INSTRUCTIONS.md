# Review instructions

Hodor checks changed code for defects and concrete violations of your team's code conventions. Commit project conventions in `AGENTS.md`. Use `--instructions <path>` for extra review guidance and `--focus <text>` for a request for this run.

All instruction inputs retain Hodor's bundled review criteria and fixed review protocol. There is no profile replacement mode.

## Project conventions with no CI flags

Commit `AGENTS.md` at the repository root:

```markdown
# Project conventions

- Handlers must obtain the tenant ID from the authenticated session.
- Database migrations must remain compatible with the previous release.
- Payment amounts must use integer minor units.
- Public API errors must use the shared error response type.
```

Hodor loads the root guidance plus guidance in ancestor directories of changed files. Put an `AGENTS.md` in a subsystem directory to add rules for that subsystem. At each directory, Hodor prefers `AGENTS.md` and uses `CLAUDE.md` only when `AGENTS.md` is absent. Deeper rules win conflicts with broader rules within their directory scope.

For hosted reviews, Hodor reads guidance from an immutable snapshot of the available `origin/<target>` branch, fetching the target branch if the ref is missing. It never uses the previous reviewed MR commit as the instruction source. For local reviews, it reads the commit selected by `--diff-against`. Uncommitted instruction edits do not apply automatically.

This means a rule added or changed in an MR does not govern that MR's own review. Once the rule is accepted into the target branch, it applies to subsequent reviews. Changes to `AGENTS.md` and `CLAUDE.md` remain in the review diff even though ordinary Markdown is excluded from the embedded diff.

Hodor considers both paths of a rename and the original paths of deleted files when selecting guidance. Guidance applies only to changed paths under its directory. It is included in the system prompt, including on the tiny-diff fast path.

Only tracked snapshot files are eligible. Symlinks must resolve to tracked regular files inside the same snapshot. Cycles, broken links, and links outside the repository fail the review. Hodor does not expand `@imports`. It does not inherit instructions from your home directory or parent directories outside the checkout. Empty repository guidance files are ignored.

## What Hodor checks

Write concrete code rules and domain invariants. Examples include authorization requirements, migration compatibility, idempotency, error contracts, and naming conventions that your team explicitly enforces.

Hodor reports an explicit convention violation only when the changed lines demonstrate it. The finding cites the instruction file and rule. Severity follows impact; naming and style violations default to P3 unless they break a concrete contract. Hodor does not invent style rules or report pre-existing violations.

Agent workflow directives are ignored. Instructions to build, run tests, lint, format, install, commit, or deploy do not apply to Hodor's read-only review. Hodor has no shell.

## Extra instruction files

Use `--instructions <path>` to add guidance from a readable UTF-8 file:

```bash
hodor https://github.com/acme/payments/pull/184 \
  --instructions /etc/hodor/security.md \
  --post
```

Example content:

```markdown
# Security checks

- Trace changed authorization checks through handlers and database queries.
- Check whether outbound requests can reach attacker-selected internal hosts.
- Check changed logging and error paths for credential exposure.
```

Repeat the flag to load several files. Later files win conflicts with earlier files:

```bash
hodor "$MR_URL" \
  --instructions /etc/hodor/company.md \
  --instructions /etc/hodor/payments.md
```

Files add to the baseline criteria. Explicit instructions may also narrow reporting, for example, "Report only security findings." They cannot change the read-only protocol, structured output, or changed-delta scope.

Paths resolve from the directory where Hodor starts, before it prepares or clones the review workspace. A file committed in the repository is not discovered by this flag unless you pass its path and it is already available to the process. Use an absolute path inside the job container in CI.

## One-off focus

Use `--focus <text>` for a request for this run:

```bash
hodor "$MR_URL" \
  --focus "Check compatibility with workers running the previous release."

hodor "$MR_URL" --focus "Report only security findings."

hodor --local --diff-against origin/main \
  --instructions /etc/hodor/security.md \
  --focus "Check the migration rollback path."
```

Focus wins conflicts with explicit instruction files. It retains the baseline criteria, but can restrict which kinds of findings Hodor reports.

## Authority and trust

| Source | Allowed effect |
|---|---|
| Bundled review criteria | Baseline defect checks and finding standards |
| Accepted repository guidance | Add scoped code rules and domain context |
| `--instructions <path>` | Add checks or explicitly narrow reporting; later files win conflicts |
| `--focus <text>` | Set this run's focus; wins conflicts with instruction files |
| Fixed Hodor protocol | Always controls read-only tools, changed-delta scope, priorities, and submission |

Repository guidance cannot suppress baseline defect classes. PR metadata, comments, diffs, HEAD repository files, and repository skills are lower-trust context. They cannot override accepted guidance, explicit reviewer policy, or Hodor's protocol.

Treat files passed with `--instructions` and focus text as trusted reviewer configuration. In shared CI, mount centrally managed files or use a trusted job workspace. Passing an MR-controlled file explicitly promotes it to reviewer policy. Do not put secrets in any instruction source.

Repository skills remain an advanced option for specialized guidance loaded when relevant. They are discovered from `.agents/skills/` in the HEAD checkout and remain lower-trust context, not accepted reviewer policy. They cannot suppress checks. See [SKILLS.md](./SKILLS.md).

## GitLab CI

Automatic project guidance needs no extra flags. Keep your existing authentication and model setup. To add shared guidance and a one-off focus to an existing review job:

```yaml
hodor-review:
  stage: test
  image:
    name: ghcr.io/mr-karan/hodor:0.12.0
    entrypoint: [""]
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  script:
    - |
      MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
      bun run /app/dist/cli.js "$MR_URL" \
        --instructions /etc/hodor/company-review.md \
        --focus "Check compatibility with the previous release." \
        --post
```

The example assumes the runner makes the trusted file available at `/etc/hodor/company-review.md`, and the job has its GitLab and model credentials. Use an image release that includes these flags; update the pinned image when deploying the breaking change.

## GitHub Actions

Add the same options to your existing job:

```yaml
- name: Review pull request
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    bun run /app/dist/cli.js \
      "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" \
      --focus "Check changes to the OAuth callback flow." \
      --post
```

Repository rules are read from the target snapshot. A checkout is needed if you pass an explicit instruction file located in the job workspace.

## Limits and troubleshooting

Each explicit instruction file and each repository guidance file is limited to 128 KiB. Explicit files must be nonempty readable regular files encoded as UTF-8. Focus text must be nonempty and at most 128 KiB.

The combined repository guidance, explicit instruction contents, and focus text are limited to 256 KiB. Over-budget input fails with a clear error; Hodor never silently truncates or drops rules.

Run with `--verbose` to see the guidance snapshot, loaded instruction paths, and skill discovery logs. If the target snapshot cannot be resolved, Hodor fails instead of substituting the MR's HEAD. Fetch the comparison ref for local reviews or make target-branch access available in CI.

Changing explicit instruction contents, focus text, or the accepted target snapshot changes the review cache identity.

## Breaking migration

The following flags are removed and rejected with migration advice:

| Removed | Replacement |
|---|---|
| `--review-instructions <path>` | `--instructions <path>` |
| `--additional-instructions <text>` | `--focus <text>` |

The replacement mode is removed. Review existing files before migrating: they now add to the baseline instead of replacing it. To retain a narrow reporting scope, explicitly state it in the file or focus text. Older `--prompt` and `--prompt-file` flags remain unsupported.

The library API changes too:

```typescript
await reviewPr({
  prUrl: mrUrl,
  instructions: ["Check tenant isolation.", "Report only security findings."],
  focus: "Pay attention to the import endpoint.",
});
```

`instructions` contains instruction text, not file paths. Use `loadReviewInstructionsFile(path)` to load files. The old `reviewInstructions` and `additionalInstructions` options are removed. `buildReviewSystemPrompt()` accepts `instructions`, `focus`, and scoped `repositoryGuidance`, and always includes the bundled criteria.
