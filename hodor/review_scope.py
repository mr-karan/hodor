"""Review scope filtering for include/exclude file patterns."""

import logging
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewScope:
    """Effective review scope patterns."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


def load_review_scope_config(workspace: Path) -> ReviewScope:
    """Load review scope from .hodor/config.toml in workspace root."""
    config_path = workspace / ".hodor" / "config.toml"
    if not config_path.exists():
        return ReviewScope()

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse {config_path}: {e}") from e

    include = _coerce_pattern_list(data.get("include"), "include", config_path)
    exclude = _coerce_pattern_list(data.get("exclude"), "exclude", config_path)
    return ReviewScope(include=tuple(include), exclude=tuple(exclude))


def resolve_review_scope(
    workspace: Path,
    cli_include: tuple[str, ...] = (),
    cli_exclude: tuple[str, ...] = (),
) -> ReviewScope:
    """Resolve review scope with CLI flags overriding config values."""
    config_scope = load_review_scope_config(workspace)
    include = cli_include if cli_include else config_scope.include
    exclude = cli_exclude if cli_exclude else config_scope.exclude
    return ReviewScope(include=tuple(include), exclude=tuple(exclude))


def get_filtered_diff_files(
    workspace: Path,
    target_branch: str,
    diff_base_sha: str | None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """List changed files from diff and apply include/exclude patterns."""
    if diff_base_sha:
        diff_args = ["git", "--no-pager", "diff", diff_base_sha, "HEAD", "--name-only"]
    else:
        diff_args = ["git", "--no-pager", "diff", f"origin/{target_branch}...HEAD", "--name-only"]

    result = subprocess.run(
        diff_args,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    changed_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    filtered_files = filter_files(changed_files, include=include, exclude=exclude)

    if include or exclude:
        logger.info(
            "Applied review scope filters: include=%s exclude=%s (changed=%d, in_scope=%d)",
            list(include),
            list(exclude),
            len(changed_files),
            len(filtered_files),
        )

    return filtered_files


def filter_files(
    paths: list[str],
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[str]:
    """Filter file paths using gitignore-style patterns.

    Include is applied first, then exclude removes from that set.
    """
    filtered = paths
    if include:
        include_spec = pathspec.PathSpec.from_lines("gitignore", include)
        filtered = [path for path in filtered if include_spec.match_file(path)]

    if exclude:
        exclude_spec = pathspec.PathSpec.from_lines("gitignore", exclude)
        filtered = [path for path in filtered if not exclude_spec.match_file(path)]

    return filtered


def _coerce_pattern_list(value: object, key: str, config_path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"Invalid {key!r} in {config_path}: expected array of strings")
    patterns: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise RuntimeError(f"Invalid {key!r} in {config_path}: expected array of strings")
        pattern = item.strip()
        if pattern:
            patterns.append(pattern)
    return patterns
