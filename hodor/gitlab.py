"""GitLab helper utilities for Hodor.

Provides wrappers around python-gitlab SDK for fetching merge request
metadata and posting review comments.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import gitlab
from gitlab import exceptions as gitlab_exceptions


logger = logging.getLogger(__name__)

DEFAULT_GITLAB_HOST = "gitlab.com"


class GitLabAPIError(RuntimeError):
    """Raised when the GitLab API fails or returns invalid data."""


def _normalize_gitlab_base_url(host: str | None = None) -> str:
    """Return the base URL (with scheme) for the GitLab instance."""

    candidate = host or os.getenv("GITLAB_HOST") or os.getenv("CI_SERVER_URL") or DEFAULT_GITLAB_HOST
    candidate = candidate.strip()
    if not candidate:
        candidate = DEFAULT_GITLAB_HOST

    if candidate.startswith(("http://", "https://")):
        base_url = candidate
    else:
        base_url = f"https://{candidate}"

    return base_url.rstrip("/")


def _gitlab_auth_kwargs() -> dict[str, str]:
    """Build authentication kwargs for python-gitlab client."""

    private_token = os.getenv("GITLAB_TOKEN") or os.getenv("GITLAB_PRIVATE_TOKEN")
    oauth_token = os.getenv("GITLAB_OAUTH_TOKEN")
    job_token = os.getenv("CI_JOB_TOKEN")

    if private_token:
        return {"private_token": private_token}
    if oauth_token:
        return {"oauth_token": oauth_token}
    if job_token:
        return {"job_token": job_token}
    return {}


def _create_gitlab_client(host: str | None = None) -> gitlab.Gitlab:
    """Instantiate a python-gitlab client with the right base URL and auth."""

    base_url = _normalize_gitlab_base_url(host)
    auth_kwargs = _gitlab_auth_kwargs()

    try:
        client = gitlab.Gitlab(base_url, **auth_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        raise GitLabAPIError(f"Failed to initialize GitLab client for {base_url}: {exc}") from exc

    logger.debug("Initialized GitLab client for %s (auth=%s)", base_url, "yes" if auth_kwargs else "anonymous")
    return client


def _get_project(client: gitlab.Gitlab, owner: str, repo: str) -> Any:
    """Return the GitLab project reference for owner/repo."""

    project_path = "/".join(part for part in [owner.strip("/"), repo.strip("/")] if part).strip("/")

    try:
        return client.projects.get(project_path)
    except gitlab_exceptions.GitlabAuthenticationError as exc:
        raise GitLabAPIError(
            "GitLab authentication failed. Set GITLAB_TOKEN (or CI_JOB_TOKEN) with api scope access."
        ) from exc
    except gitlab_exceptions.GitlabGetError as exc:
        raise GitLabAPIError(
            f"Unable to find GitLab project '{project_path}'. "
            f"Verify the URL and ensure your token has access. ({exc.error_message or exc})"
        ) from exc
    except gitlab_exceptions.GitlabError as exc:  # pragma: no cover - defensive
        raise GitLabAPIError(f"Unexpected GitLab error while fetching project '{project_path}': {exc}") from exc


def _get_merge_request(project: Any, mr_number: str | int) -> Any:
    """Fetch a merge request safely."""

    try:
        return project.mergerequests.get(int(mr_number))
    except gitlab_exceptions.GitlabAuthenticationError as exc:
        raise GitLabAPIError(
            "GitLab authentication failed while fetching the merge request. "
            "Ensure GITLAB_TOKEN (or CI_JOB_TOKEN) is valid."
        ) from exc
    except gitlab_exceptions.GitlabGetError as exc:
        raise GitLabAPIError(
            f"Could not fetch merge request !{mr_number}: {exc.error_message or exc}"
        ) from exc


def _serialize_notes(mr: Any) -> list[dict[str, Any]]:
    """Return serialized note dictionaries for an MR."""

    try:
        notes = mr.notes.list(all=True, sort="asc", per_page=100)
    except gitlab_exceptions.GitlabError as exc:
        raise GitLabAPIError(f"Failed to fetch notes for merge request !{mr.iid}: {exc}") from exc

    return [note.attributes for note in notes]


def fetch_gitlab_mr_info(
    owner: str,
    repo: str,
    mr_number: str | int,
    host: str | None = None,
    *,
    include_comments: bool = False,
) -> dict[str, Any]:
    """Return merge request metadata using python-gitlab."""

    client = _create_gitlab_client(host)
    project = _get_project(client, owner, repo)
    mr = _get_merge_request(project, mr_number)
    mr_data = dict(mr.attributes)  # copy to detach from SDK object

    if include_comments:
        mr_data["Notes"] = _serialize_notes(mr)

    return mr_data


def post_gitlab_mr_comment(
    owner: str,
    repo: str,
    mr_number: str | int,
    body: str,
    *,
    host: str | None = None,
) -> dict[str, Any]:
    """Post a top-level note on a GitLab merge request."""

    client = _create_gitlab_client(host)
    project = _get_project(client, owner, repo)
    mr = _get_merge_request(project, mr_number)

    try:
        note = mr.notes.create({"body": body})
    except gitlab_exceptions.GitlabAuthenticationError as exc:
        raise GitLabAPIError("GitLab authentication failed when posting the review comment.") from exc
    except gitlab_exceptions.GitlabCreateError as exc:
        raise GitLabAPIError(f"GitLab rejected the review comment: {exc.error_message or exc}") from exc
    except gitlab_exceptions.GitlabError as exc:  # pragma: no cover - defensive
        raise GitLabAPIError(f"Failed to post comment to merge request !{mr_number}: {exc}") from exc

    return note.attributes


def find_hodor_comment(
    owner: str,
    repo: str,
    mr_number: str | int,
    host: str | None = None,
) -> dict[str, Any] | None:
    """Find the existing Hodor review comment on a GitLab MR, if any."""

    client = _create_gitlab_client(host)
    project = _get_project(client, owner, repo)
    mr = _get_merge_request(project, mr_number)

    try:
        notes = mr.notes.list(all=True, sort="desc", per_page=100)
    except gitlab_exceptions.GitlabError as exc:
        logger.warning(f"Failed to list notes for MR !{mr_number}: {exc}")
        return None

    for note in notes:
        if "Review generated by Hodor" in (note.body or ""):
            return {"id": note.id, "body": note.body}
    return None


def update_gitlab_mr_comment(
    owner: str,
    repo: str,
    mr_number: str | int,
    note_id: int,
    body: str,
    *,
    host: str | None = None,
) -> dict[str, Any]:
    """Update an existing note on a GitLab merge request."""

    client = _create_gitlab_client(host)
    project = _get_project(client, owner, repo)
    mr = _get_merge_request(project, mr_number)

    try:
        note = mr.notes.get(note_id)
        note.body = body
        note.save()
    except gitlab_exceptions.GitlabError as exc:
        raise GitLabAPIError(f"Failed to update note {note_id} on MR !{mr_number}: {exc}") from exc

    return note.attributes


def summarize_gitlab_notes(
    notes: list[dict[str, Any]] | None,
    *,
    max_entries: int = 5,
) -> str:
    """Return a human-readable bullet list for the most relevant notes.

    Preserves full comment text and multi-line formatting to provide complete context.
    Filters out trivial comments to optimize token usage.
    """

    if not notes:
        return ""

    # Trivial comment patterns to skip
    trivial_patterns = {
        "lgtm", "+1", "-1", "👍", "👎", "thanks", "thank you",
        "looks good", "approved", "🚀", "✅", "❌"
    }

    filtered = []
    for note in notes:
        body = note.get("body", "").strip()
        if not body:
            continue

        author = note.get("author", {})
        username = author.get("username") or author.get("name") or "unknown"
        is_system = note.get("system", False)

        # Skip GitLab system notes (merge events, label changes, etc.)
        if is_system:
            continue

        # Skip very short comments (likely not substantive)
        if len(body) < 20:
            continue

        # Skip trivial comments
        body_lower = body.lower()
        if any(pattern in body_lower for pattern in trivial_patterns):
            # Only skip if the comment is ONLY the trivial pattern (short comment)
            if len(body) < 50:
                continue

        # Include meaningful human comments
        filtered.append((username, body, note.get("created_at", "")))

    # Sort by date (oldest first)
    filtered.sort(key=lambda x: x[2])

    # Take most recent
    recent = filtered[-max_entries:]

    lines = []
    for username, body, created_at in recent:
        # Parse timestamp for better display
        timestamp_str = ""
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                timestamp_str = created_at[:10] if len(created_at) >= 10 else created_at

        # Format the comment header with timestamp and author
        if timestamp_str:
            header = f"- {timestamp_str} @{username}:"
        else:
            header = f"- @{username}:"

        # Indent the body for better readability if multi-line
        # Preserve full comment text without truncation
        indented_body = "\n  ".join(body.split("\n"))
        lines.append(f"{header}\n  {indented_body}")

    return "\n".join(lines)
