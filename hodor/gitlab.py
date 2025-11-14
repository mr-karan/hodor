"""GitLab helper utilities for Hodor.

Provides wrappers around the `glab` CLI so we can fetch merge request
metadata and reuse it across workspace setup and prompt generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)


class GitLabAPIError(RuntimeError):
    """Raised when glab fails or returns invalid data."""


def _run_glab_json_command(args: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a glab command that returns JSON and parse the output."""

    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - passthrough path
        error_msg = exc.stderr if getattr(exc, "stderr", None) else str(exc)
        raise GitLabAPIError(error_msg) from exc

    output = result.stdout.strip()
    json_start = output.find("{")
    if json_start == -1:
        raise GitLabAPIError("glab output did not contain JSON payload")

    if json_start > 0:
        logger.debug("Skipping non-JSON prefix in glab output: %s", output[:json_start])
        output = output[json_start:]

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - passthrough path
        raise GitLabAPIError(f"Unable to parse glab JSON output: {exc}") from exc


def fetch_gitlab_mr_info(
    owner: str,
    repo: str,
    mr_number: str | int,
    host: str | None = None,
    *,
    include_comments: bool = False,
) -> dict[str, Any]:
    """Return the JSON metadata for a GitLab merge request via glab."""

    gitlab_host = host or os.getenv("GITLAB_HOST", "gitlab.com")
    repo_full_path = f"{owner}/{repo}"

    glab_env = os.environ.copy()
    glab_env["GITLAB_HOST"] = gitlab_host
    glab_env.pop("GLAMOUR_STYLE", None)  # ensure no ANSI art sneaks into stdout

    args = [
        "glab",
        "mr",
        "view",
        str(mr_number),
        "--repo",
        repo_full_path,
        "-F",
        "json",
    ]

    if include_comments:
        args.append("--comments")

    return _run_glab_json_command(args, env=glab_env)


def _condense_whitespace(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def summarize_gitlab_notes(
    notes: list[dict[str, Any]] | None,
    *,
    max_entries: int = 10,
) -> str:
    """Return a human-readable bullet list for the most relevant notes."""

    if not notes:
        return ""

    lines: list[str] = []
    for note in notes:
        if note.get("system"):
            continue
        body = note.get("body", "").strip()
        if not body:
            continue

        author = note.get("author", {})
        username = author.get("username") or author.get("name") or "unknown"
        created_at = note.get("created_at") or ""
        try:
            timestamp = datetime.fromisoformat(created_at.rstrip("Z"))
            created_str = timestamp.strftime("%Y-%m-%d %H:%M")
        except ValueError:  # pragma: no cover - formatting fallback
            created_str = created_at

        formatted_body = body.strip() or "(empty comment)"
        lines.append(f"- {created_str} @{username}:\n{formatted_body}")

        if len(lines) >= max_entries:
            break

    return "\n".join(lines)
