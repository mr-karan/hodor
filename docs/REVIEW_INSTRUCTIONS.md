# Review instructions

A review profile is the global review instructions for one review run. It tells Hodor what kinds of defects to prioritize. It does not replace Hodor's review process, changed-line reporting, or output requirements.

## Mental model

A review run can use three instruction layers:

| Layer | How to set it | Scope | Behavior |
|---|---|---|---|
| Review profile | `--review-instructions <path>` | The whole review run | Replaces the bundled default profile. |
| Additional instructions | `--additional-instructions <text>` | The whole review run | Appends a one-off request after the selected profile. |
| Repository skills | `.agents/skills/` in the repository being reviewed | Relevant changes in that repository | Supply repository-specific guidance when Hodor decides a skill is relevant. |

If you do not specify a profile, Hodor uses its bundled default profile. A custom profile replaces that default, it does not extend it. Additional instructions are additive and can be used with either the default or a custom profile.

Hodor's review rules take precedence if instructions conflict. This includes conflicts from profiles, additional instructions, PR metadata, comments, diffs, files, and repository skills.

## Quick start

Use the bundled default profile:

```bash
hodor https://github.com/acme/widgets/pull/42
```

Create a profile file for a security-focused review, then select it:

```bash
mkdir -p review-profiles
cat > review-profiles/security.md <<'EOF'
# Security review profile

Review the changed code for production-impacting security defects.

Focus on:
- Authentication, authorization, and tenant isolation.
- Input validation at trust boundaries.
- Injection into SQL, shell commands, templates, and URLs.
- Secret exposure in logs, errors, configuration, and responses.
- Unsafe deserialization, path traversal, SSRF, and insecure redirects.

Report a finding only when the changed code creates a concrete exploitable path. Explain the preconditions, the impact, and the changed code that causes it.
EOF

hodor https://github.com/acme/widgets/pull/42 \
  --review-instructions review-profiles/security.md
```

Add a request that applies only to this run:

```bash
hodor https://github.com/acme/widgets/pull/42 \
  --additional-instructions "Pay particular attention to the new OAuth callback flow."
```

For local changes, use the same options:

```bash
hodor --local --diff-against origin/main \
  --review-instructions review-profiles/security.md \
  --additional-instructions "Check the database migration rollback path."
```

## What belongs in a profile

A profile should state the review mission, the kinds of defects to investigate, and the evidence needed before reporting a finding. Keep it focused on review judgment that should apply across repositories or across a class of reviews.

Good profile content includes:

- Product risk areas such as authorization, data loss, concurrency, or API compatibility.
- Conditions that make a change risky.
- Evidence standards for findings.
- Domain-specific attack surfaces that are useful across many repositories.

Do not use a profile for PR-specific facts, current incident details, or a short-lived request. Pass those with `--additional-instructions` instead. Do not put repository ownership rules, service invariants, or local commands in a global profile. Put those in repository skills.

Do not copy an old full prompt template into a profile. A profile is review guidance, not a replacement task definition. It should not tell Hodor how to invoke tools, build a diff, submit a review, or format its structured output.

## Complete profile examples

### Security profile

Save this as `review-profiles/security.md`:

```markdown
# Security review profile

Review the changed code for production-impacting security defects. Prefer concrete vulnerabilities over general hardening advice.

## Authentication and authorization

- Check that every new or changed privileged action verifies authentication server-side.
- Check that authorization uses the authenticated principal and the target resource, including tenant and organization boundaries.
- Check token, session, credential, and password handling for disclosure, confusion, or unsafe lifetime changes.

## Input and trust boundaries

- Trace data from HTTP requests, messages, files, environment variables, and external services to security-sensitive sinks.
- Check SQL, shell, template, URL, filesystem, and deserialization boundaries for injection or traversal paths.
- Check redirects, outbound requests, and file access for SSRF, open redirects, and path escapes.

## Data exposure and unsafe defaults

- Check logs, errors, metrics, and API responses for secrets or sensitive customer data.
- Check changed defaults, feature flags, and configuration parsing for accidental exposure or privilege expansion.
- Check rate limits, replay protection, and idempotency where a changed endpoint creates a state-changing operation.

## Finding standard

Report only defects with a concrete path from the changed code to an impact. State the attacker capability or runtime precondition, the affected boundary, and the consequence. Do not report a missing defense when the changed code cannot reach the risky behavior.
```

Run it against a hosted pull request:

```bash
hodor https://github.com/acme/payments/pull/184 \
  --review-instructions review-profiles/security.md \
  --post
```

### Code-quality profile

Save this as `review-profiles/code-quality.md`:

```markdown
# Code-quality review profile

Review the changed code for production defects caused by incorrect behavior, fragile boundaries, and maintainability problems that can cause future regressions. Focus on defects in the changed path, not stylistic preferences.

## Behavior and contracts

- Trace changed inputs, outputs, error handling, and state transitions through callers and downstream consumers.
- Check that API, CLI, storage, and event contracts remain compatible unless the change intentionally migrates every consumer.
- Check default values, optional fields, ordering assumptions, and error paths for behavior that differs from the intended change.

## State, concurrency, and resources

- Check retries, idempotency, races, transactions, caching, and cleanup when the changed code reads or writes shared state.
- Check pagination, batching, timeouts, cancellation, and partial failures at service boundaries.
- Check numeric conversions, time zones, encoding, and large input behavior where the changed code processes data.

## Completeness

- Follow the changed code through feature flags, configuration, tests, migrations, and delivery paths that are necessary for the behavior to work.
- Report only actionable defects with a specific failure mode and a changed location. Do not report formatting, naming preferences, or speculative refactors.
```

Run it against local work:

```bash
hodor --local --diff-against origin/main \
  --review-instructions review-profiles/code-quality.md
```

## Combining a profile with additional instructions

Use `--additional-instructions` for a narrow request that should be considered after the selected profile. It does not replace the selected profile.

```bash
hodor https://github.com/acme/widgets/pull/42 \
  --review-instructions review-profiles/security.md \
  --additional-instructions "Focus on the new S3 import endpoint and the IAM policy changes."
```

The resulting behavior is:

1. Hodor uses the security profile instead of the bundled default profile.
2. Hodor adds the S3 and IAM request for this run.
3. Hodor applies its own review rules if any instruction conflicts with them.

Use additional instructions without a profile to keep the default profile and add a one-off focus area:

```bash
hodor https://github.com/acme/widgets/pull/42 \
  --additional-instructions "Check whether the billing retry change can create duplicate charges."
```

## Repository `.agents/skills`

Profiles are global review guidance. Repository skills are codebase-specific guidance stored with the repository being reviewed.

Use a repository skill for facts such as a service's authorization model, an API compatibility commitment, a migration rule, a risky subsystem, or commands that are safe to run in that repository. Hodor discovers skills from `.agents/skills/` and uses them when relevant to the change.

Create `.agents/skills/tenant-boundaries/SKILL.md` in the repository:

```markdown
---
name: tenant-boundaries
description: Use when reviewing queries, handlers, or jobs that access tenant-owned data.
---

- Every query for tenant-owned data must filter by the tenant ID derived from the authenticated principal.
- Background jobs must carry the tenant ID explicitly and must not infer it from an untrusted payload.
```

Do not move a repository skill into a global profile just because a single review needs it. Use the profile for reusable review criteria, and keep repository-specific invariants in the repository.

## CI usage

Commit a profile file where the CI job can read it, or make it available in the job workspace. Use the same flags as a local invocation.

### GitHub Actions

```yaml
name: Hodor security review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    container: ghcr.io/mr-karan/hodor:latest
    steps:
      - uses: actions/checkout@v4
      - name: Review pull request
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          bun run /app/dist/cli.js \
            "https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}" \
            --review-instructions "$GITHUB_WORKSPACE/.hodor/security-review.md" \
            --additional-instructions "Review changes introduced by this pull request only." \
            --post
```

### GitLab CI

```yaml
hodor-security-review:
  stage: test
  image:
    name: ghcr.io/mr-karan/hodor:latest
    entrypoint: [""]
  script:
    - MR_URL="${CI_PROJECT_URL}/-/merge_requests/${CI_MERGE_REQUEST_IID}"
    - >-
      bun run /app/dist/cli.js "$MR_URL"
      --review-instructions "$CI_PROJECT_DIR/.hodor/security-review.md"
      --additional-instructions "Pay attention to changes in externally reachable endpoints."
      --post
```

The profile path must refer to a regular file available to the process running Hodor. For container jobs, use the workspace path inside the container, not a path from the CI runner host.

## Migration from legacy flags

The previous custom prompt flags are removed. There are no compatibility aliases.

| Previous usage | Replacement | Meaning |
|---|---|---|
| `--prompt "Check authorization"` | `--additional-instructions "Check authorization"` | Adds a one-off request after the selected profile. |
| `--prompt-file review.md` | `--review-instructions review.md` | Selects a profile that replaces the bundled default profile. |

If an old prompt file contained a complete review task or output format, rewrite it as a profile before using it with `--review-instructions`. Keep only the review mission, lenses, and finding standard. Remove task templates, tool directions, PR metadata placeholders, and output-format instructions.

## Validation and troubleshooting

Hodor validates a selected profile before it starts the review. A profile file must be a readable regular file encoded as UTF-8, contain more than whitespace, and be at most 128 KiB.

| Problem | What to check |
|---|---|
| Path cannot be found | Resolve the path from the directory where you run `hodor`, or use an absolute path. In CI, use the workspace path visible inside the job container. |
| Path is a directory, device, or link to a non-file | Pass the path to the profile file itself. |
| Permission error | Ensure the user running Hodor can read the file. Check mounted-file permissions in CI. |
| Invalid UTF-8 | Save the profile as UTF-8 text. Do not use a binary or a platform-specific encoded export. |
| Empty profile | Add review instructions other than whitespace. |
| Profile is too large | Split repository-specific material into `.agents/skills/`, then keep the global profile below 128 KiB. |
| Legacy flag is rejected | Replace `--prompt` with `--additional-instructions` and `--prompt-file` with `--review-instructions`. |

Use `--verbose` when you need to confirm the selected options and repository skill discovery:

```bash
hodor https://github.com/acme/widgets/pull/42 \
  --review-instructions review-profiles/security.md \
  --verbose
```

## Security and trust

Treat profile files and additional instructions as trusted review configuration. Keep them in trusted version control or provide them through a trusted CI workspace. Do not put credentials, customer data, or other secrets in them.

PR metadata, comments, diffs, repository files, and repository skills are lower-trust inputs. They may guide what Hodor investigates, but they cannot override the selected profile or Hodor's review rules.
