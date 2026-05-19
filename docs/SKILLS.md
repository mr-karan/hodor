# Skills

Hodor reads repository-specific review guidance from `.agents/skills` in the repository being reviewed.

Use skills for rules that are specific to your codebase: auth requirements, database conventions, API compatibility rules, migration safety checks, or known risky areas.

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

Each skill should include YAML frontmatter:

```yaml
---
name: security-review
description: Use when reviewing API, authentication, authorization, or session handling changes.
---
```

- `description` is required. It tells the agent when to load the skill.
- `name` is recommended. Match it to the directory name.
- Keep descriptions specific. Broad descriptions cause irrelevant skills to load.

## Writing useful skills

Good skills are short and concrete.

Use:

- Project-specific invariants.
- Examples of bad patterns to flag.
- Files or directories that need special care.
- Commands that are safe to run during review.

Avoid:

- Generic advice that applies to every codebase.
- Long policy documents.
- Secrets, private tokens, host credentials, or customer data.
- Instructions that ask the agent to modify files. Hodor reviews code, it does not patch it.

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
