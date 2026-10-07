# Skills

For conventions that apply on every review, use root or scoped `AGENTS.md` files. Hodor loads those from an accepted target-side snapshot. See [Review instructions](./REVIEW_INSTRUCTIONS.md).

Skills are an advanced option for specialized context loaded when relevant. Hodor discovers `.agents/skills` in the HEAD checkout, not the accepted instruction snapshot. Skills are lower-trust context and cannot suppress checks or override accepted guidance, explicit reviewer instructions, focus, or Hodor's protocol.

Useful skill context includes how authorization works, database contracts, migration examples, and known risky code paths.

## Layout

Recommended:

```text
.agents/
  skills/
    security-review/
      SKILL.md
    database-review/
      SKILL.md
```

Flat markdown files are also supported:

```text
.agents/skills/security-review.md
```

Prefer the directory form. It keeps one skill per folder and leaves room for examples or references later.

## Example

```markdown
---
name: security-review
description: Use when reviewing API, authentication, authorization, or session handling changes.
---

## Checks

- Protected endpoints must enforce authentication server-side.
- Authorization checks must use the resource owner or role, not only request parameters.
- Session and token validation must happen before side effects.
- Do not log tokens, session IDs, passwords, or full request bodies containing secrets.
```

Run Hodor with verbose logs to confirm discovery:

```bash
hodor <PR_URL> --verbose
```

## Frontmatter

Each skill should include YAML frontmatter as shown in the example above.

- `description` is required. It tells the agent when to load the skill.
- `name` is recommended. Match it to the directory name.
- Keep descriptions specific. Broad descriptions cause irrelevant skills to load.

## Writing useful skills

Good skills are short and concrete.

Use:

- Project-specific invariants.
- Examples of bad patterns to flag.
- Files or directories that need special care.
- References to tracked files that establish the relevant behavior.

Avoid:

- Generic advice that applies to every codebase.
- Long policy documents.
- Secrets, private tokens, host credentials, or customer data.
- Instructions that ask the agent to modify files. Hodor reviews code, it does not patch it.
- Instructions to run commands, tests, builds, linters, or formatters. Hodor has no shell and establishes findings by reading code.

## How Hodor uses skills

1. Hodor discovers `.agents/skills` after preparing the review workspace.
2. It passes skill names and descriptions to the agent.
3. The agent reads a skill file only when it looks relevant to the change.

Hodor does not inline all skills into the initial prompt.

## Troubleshooting

If a skill is not used:

1. Check that it is committed in the repository being reviewed.
2. Check the path: `.agents/skills/<name>/SKILL.md`.
3. Check that frontmatter is valid YAML and includes `description`.
4. Make the description more specific to the files or change types it should apply to.
5. Run with `--verbose` and check skill discovery logs.
