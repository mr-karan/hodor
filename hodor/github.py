"""GitHub helper utilities for Hodor."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any


logger = logging.getLogger(__name__)


class GitHubAPIError(RuntimeError):
    """Raised when gh fails or returns invalid data."""


def _run_gh_json_command(args: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
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
        raise GitHubAPIError(error_msg) from exc

    output = result.stdout.strip()
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:  # pragma: no cover - passthrough path
        raise GitHubAPIError(f"Unable to parse gh JSON output: {exc}") from exc


def fetch_github_pr_info(
    owner: str,
    repo: str,
    pr_number: str | int,
) -> dict[str, Any]:
    fields = [
        "number",
        "title",
        "body",
        "author",
        "baseRefName",
        "headRefName",
        "baseRefOid",
        "headRefOid",
        "changedFiles",
        "labels",
        "comments",
        "state",
        "isDraft",
        "createdAt",
        "updatedAt",
        "mergeable",
        "url",
    ]
    repo_full_path = f"{owner}/{repo}"
    args = [
        "gh",
        "pr",
        "view",
        str(pr_number),
        "-R",
        repo_full_path,
        "--json",
        ",".join(fields),
    ]
    return _run_gh_json_command(args)


def normalize_github_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "title": raw.get("title"),
        "description": raw.get("body", ""),
        "source_branch": raw.get("headRefName"),
        "target_branch": raw.get("baseRefName"),
        "changes_count": raw.get("changedFiles"),
        "labels": [{"name": lbl.get("name") or lbl.get("id")} for lbl in (raw.get("labels") or [])],
        "author": {
            "username": raw.get("author", {}).get("login") or raw.get("author", {}).get("name"),
            "name": raw.get("author", {}).get("name"),
        },
        "Notes": _github_comments_to_notes(raw.get("comments")),
    }
    return metadata


def _github_comments_to_notes(comments: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not comments:
        return []
    nodes = comments.get("nodes") or []
    notes: list[dict[str, Any]] = []
    for node in nodes:
        notes.append(
            {
                "body": node.get("body", ""),
                "author": {
                    "username": node.get("author", {}).get("login") or node.get("author", {}).get("name"),
                    "name": node.get("author", {}).get("name"),
                },
                "created_at": node.get("createdAt"),
            }
        )
    return notes
