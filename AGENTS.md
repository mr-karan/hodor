# Repository Guidelines

## Project Structure & Module Organization
- The runtime package lives in `hodor/`: `cli.py` exposes the Click entrypoint, `agent.py` manages the review loop, while `tools/` and `utils/` host integration helpers and shared logic.
- Reusable prompt variants are under `prompts/`; author-facing notes and walkthroughs belong in `docs/`.
- Keep credentials and local overrides out of git by editing `.env` (mirrors `.env.example`).
- Place regression assets, sample diffs, or screenshots next to the docs they support.
- New tests and fixtures should land in `tests/`, which pytest auto-discovers via `test_*.py`.

## Build, Test, and Development Commands
Use `uv` via Just to guarantee the locked toolchain:
```bash
just sync        # create/update the uv environment
just check       # run formatting, lint, and type checks
just test-cov    # execute pytest with HTML + terminal coverage
just review URL  # shortcut for `uv run hodor URL [flags]`
```
Docker workflows exist for CI: `docker-build` for local smoke tests, `docker-build-multi` before publishing.

## Coding Style & Naming Conventions
Target Python 3.13, Black formatting (120-char lines, 4-space indents), and Ruff linting with the same width. Prefer descriptive module-level names (`github_tools.py`, `file_classifier.py`) and snake_case for functions, UPPER_SNAKE_CASE for constants, and CapWords for classes. Run `just fmt` before committing to avoid noisy diffs.

## Testing Guidelines
Pytest is configured via `pyproject.toml` with `tests` as the root and files/functions named `test_*`. Aim to keep coverage near the default `pytest --cov=hodor` output (term-missing must stay clean). Reach for `just test-watch` when iterating locally, and regenerate the HTML report with `just test-cov` when validating larger refactors.

## Commit & Pull Request Guidelines
Follow the existing history: short, imperative subjects (`Add Hodor - …`) under 72 chars plus optional body. Squash unrelated changes, reference issues (`Fixes #123`) when applicable, and attach before/after screenshots for CLI output tweaks. Every PR description should list the motivation, testing proof (commands run), and any follow-up TODOs so reviewers can reproduce your setup.

## Security & Configuration Tips
Store API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GITHUB_TOKEN`, `GITLAB_TOKEN`) in `.env` and pass them through uv or Docker (`just docker-run` respects exported variables). Never commit credentials, lockfiles containing secrets, or sanitized logs—scrub with `tools/file_classifier.py` if unsure. When adding new prompts, document their risk posture in `docs/` so operators can pick the right persona during deployments.
