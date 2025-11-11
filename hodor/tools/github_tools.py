"""GitHub API tool implementations."""

import logging
import os
from threading import Lock
from typing import Any

from github import Github, Auth
from github.GithubException import GithubException

logger = logging.getLogger(__name__)

_PR_FILE_MAP_CACHE: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = {}
_REPO_TREE_CACHE: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
_PR_FILE_MAP_LOCK = Lock()
_REPO_TREE_LOCK = Lock()


def _get_github_client(token: str | None = None) -> Github:
    """Create authenticated GitHub client."""
    auth_token = token or os.getenv("GITHUB_TOKEN")

    if auth_token:
        auth = Auth.Token(auth_token)
        return Github(auth=auth)
    else:
        logger.warning("No GitHub token provided - API rate limits will be very low")
        return Github()


def _get_repository(owner: str, repo: str, github_token: str | None = None):
    """Return a repository handle."""
    g = _get_github_client(github_token)
    return g.get_repo(f"{owner}/{repo}")


def _get_pull_request(owner: str, repo: str, pr_number: int, github_token: str | None = None):
    """Fetch and return (repository, pull_request)."""
    repository = _get_repository(owner, repo, github_token)
    pr = repository.get_pull(pr_number)
    return repository, pr


def _get_pr_file_map(
    owner: str, repo: str, pr_number: int, github_token: str | None = None
) -> dict[str, dict[str, Any]]:
    """Cache and return the per-file diff metadata for a PR."""
    repository, pr = _get_pull_request(owner, repo, pr_number, github_token)
    head_sha = pr.head.sha
    cache_key = (owner, repo, pr_number, head_sha)

    with _PR_FILE_MAP_LOCK:
        cached = _PR_FILE_MAP_CACHE.get(cache_key)
        if cached is not None:
            return cached

    files_map: dict[str, dict[str, Any]] = {}
    for file in pr.get_files():
        files_map[file.filename] = {
            "filename": file.filename,
            "status": file.status,
            "additions": file.additions,
            "deletions": file.deletions,
            "changes": file.changes,
            "patch": file.patch or "",
            "previous_filename": getattr(file, "previous_filename", None),
        }

    with _PR_FILE_MAP_LOCK:
        # Drop stale cache entries for the same PR.
        stale_keys = [key for key in _PR_FILE_MAP_CACHE if key[:3] == cache_key[:3] and key != cache_key]
        for stale_key in stale_keys:
            _PR_FILE_MAP_CACHE.pop(stale_key, None)
        _PR_FILE_MAP_CACHE[cache_key] = files_map

    return files_map


def _get_repo_tree(owner: str, repo: str, ref: str, github_token: str | None = None) -> list[dict[str, Any]]:
    """Cache and return the repository tree at a given ref."""
    cache_key = (owner, repo, ref)
    with _REPO_TREE_LOCK:
        cached = _REPO_TREE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    repository = _get_repository(owner, repo, github_token)
    tree = repository.get_git_tree(ref, recursive=True)
    tree_entries: list[dict[str, Any]] = [
        {"path": entry.path, "type": entry.type, "size": entry.size, "sha": entry.sha} for entry in tree.tree
    ]

    with _REPO_TREE_LOCK:
        _REPO_TREE_CACHE[cache_key] = tree_entries

    return tree_entries


def fetch_pr_metadata(owner: str, repo: str, pr_number: int, github_token: str | None = None) -> dict[str, Any]:
    """
    Fetch PR metadata including title, description, author, timestamps, labels.

    Returns:
        Dictionary with PR metadata
    """
    logger.info(f"Fetching PR metadata for {owner}/{repo}/pull/{pr_number}")

    try:
        _, pr = _get_pull_request(owner, repo, pr_number, github_token)

        return {
            "title": pr.title,
            "description": pr.body or "",
            "state": pr.state,
            "author": pr.user.login,
            "created_at": pr.created_at.isoformat(),
            "updated_at": pr.updated_at.isoformat(),
            "merged": pr.merged,
            "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
            "base_branch": pr.base.ref,
            "head_branch": pr.head.ref,
            "labels": [label.name for label in pr.labels],
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
            "url": pr.html_url,
        }

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to fetch PR metadata: {e}")


def fetch_pr_files(owner: str, repo: str, pr_number: int, github_token: str | None = None) -> dict[str, Any]:
    """
    Fetch list of all changed files with addition/deletion stats.

    Returns:
        Dictionary with list of changed files
    """
    logger.info(f"Fetching PR files for {owner}/{repo}/pull/{pr_number}")

    try:
        files_map = _get_pr_file_map(owner, repo, pr_number, github_token)
        files = [
            {
                "filename": data["filename"],
                "status": data["status"],
                "additions": data["additions"],
                "deletions": data["deletions"],
                "changes": data["changes"],
                "patch": (data["patch"][:500] if data["patch"] else None),
            }
            for data in files_map.values()
        ]

        return {"total_files": len(files), "files": files}

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to fetch PR files: {e}")


def fetch_file_diff(
    owner: str, repo: str, pr_number: int, file_path: str, github_token: str | None = None
) -> dict[str, Any]:
    """
    Fetch detailed diff for a specific file.

    Returns:
        Dictionary with file diff information
    """
    logger.info(f"Fetching diff for {file_path} in {owner}/{repo}/pull/{pr_number}")

    try:
        files_map = _get_pr_file_map(owner, repo, pr_number, github_token)
        if file_path not in files_map:
            return {"error": f"File {file_path} not found in PR"}

        file = files_map[file_path]
        return {
            "filename": file["filename"],
            "status": file["status"],
            "additions": file["additions"],
            "deletions": file["deletions"],
            "changes": file["changes"],
            "patch": file["patch"],
            "previous_filename": file.get("previous_filename"),
        }

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to fetch file diff: {e}")


def fetch_pr_commits(owner: str, repo: str, pr_number: int, github_token: str | None = None) -> dict[str, Any]:
    """
    Fetch list of commits in the PR.

    Returns:
        Dictionary with commit list
    """
    logger.info(f"Fetching commits for {owner}/{repo}/pull/{pr_number}")

    try:
        _, pr = _get_pull_request(owner, repo, pr_number, github_token)

        commits = []
        for commit in pr.get_commits():
            commits.append(
                {
                    "sha": commit.sha,
                    "message": commit.commit.message,
                    "author": commit.commit.author.name,
                    "author_email": commit.commit.author.email,
                    "date": commit.commit.author.date.isoformat(),
                    "url": commit.html_url,
                }
            )

        return {"total_commits": len(commits), "commits": commits}

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to fetch PR commits: {e}")


def fetch_ci_status(owner: str, repo: str, pr_number: int, github_token: str | None = None) -> dict[str, Any]:
    """
    Fetch CI/CD check status for the PR.

    Returns:
        Dictionary with CI status information
    """
    logger.info(f"Fetching CI status for {owner}/{repo}/pull/{pr_number}")

    try:
        _, pr = _get_pull_request(owner, repo, pr_number, github_token)

        # Get the latest commit
        commits = list(pr.get_commits())
        if not commits:
            return {"error": "No commits found in PR"}

        latest_commit = commits[-1]

        # Get check runs
        check_runs = latest_commit.get_check_runs()

        checks = []
        for check in check_runs:
            checks.append(
                {
                    "name": check.name,
                    "status": check.status,  # queued, in_progress, completed
                    "conclusion": check.conclusion,  # success, failure, neutral, cancelled, skipped, timed_out, action_required
                    "started_at": check.started_at.isoformat() if check.started_at else None,
                    "completed_at": check.completed_at.isoformat() if check.completed_at else None,
                    "url": check.html_url,
                }
            )

        # Get combined status
        combined_status = latest_commit.get_combined_status()
        statuses = []
        for status in combined_status.statuses:
            statuses.append(
                {
                    "context": status.context,
                    "state": status.state,  # pending, success, failure, error
                    "description": status.description,
                    "target_url": status.target_url,
                }
            )

        return {
            "commit_sha": latest_commit.sha,
            "combined_state": combined_status.state,
            "total_checks": len(checks),
            "checks": checks,
            "total_statuses": len(statuses),
            "statuses": statuses,
        }

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to fetch CI status: {e}")


def search_tests(
    owner: str, repo: str, pr_number: int, file_path: str, github_token: str | None = None
) -> dict[str, Any]:
    """
    Search for test files related to the given source file.

    Returns:
        Dictionary with list of potential test files
    """
    logger.info(f"Searching for tests related to {file_path} in {owner}/{repo}")

    try:
        import os.path

        _, pr = _get_pull_request(owner, repo, pr_number, github_token)
        ref = pr.head.sha

        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0].lower()
        dir_name = os.path.dirname(file_path).rstrip("/")

        test_patterns = [
            f"test_{name_without_ext}",
            f"{name_without_ext}_test",
            f"{name_without_ext}.test",
            f"{name_without_ext}.spec",
        ]
        pattern_basenames = {os.path.splitext(pattern)[0] for pattern in test_patterns}

        tree_entries = _get_repo_tree(owner, repo, ref, github_token)
        matches: list[dict[str, Any]] = []

        for entry in tree_entries:
            if entry["type"] != "blob":
                continue

            entry_base = os.path.basename(entry["path"])
            entry_base_no_ext = os.path.splitext(entry_base)[0].lower()

            reason = None
            if entry_base_no_ext in pattern_basenames:
                reason = "pattern_match"
            elif entry_base_no_ext.startswith(f"test_{name_without_ext}") or entry_base_no_ext.endswith(
                f"_{name_without_ext}"
            ):
                reason = "name_variant"
            elif name_without_ext and name_without_ext in entry_base_no_ext:
                reason = "name_overlap"

            if not reason:
                continue

            proximity = 2 if dir_name and entry["path"].startswith(f"{dir_name}/") else 1
            matches.append(
                {
                    "path": entry["path"],
                    "sha": entry["sha"],
                    "size": entry["size"],
                    "match_reason": reason,
                    "proximity": proximity,
                }
            )

        matches.sort(key=lambda item: (-item["proximity"], item["path"]))
        limited_matches = matches[:25]

        return {
            "source_file": file_path,
            "ref": ref,
            "test_patterns": test_patterns,
            "found_tests": limited_matches,
            "total_candidates": len(matches),
            "note": "Results computed via repository tree scan of PR head",
        }

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to search tests: {e}")


def fetch_file_content(
    owner: str,
    repo: str,
    pr_number: int,
    file_path: str,
    ref: str | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """Return the contents of a file at the PR head (or supplied ref)."""
    logger.info(f"Fetching file contents for {file_path} in {owner}/{repo}@{ref or 'PR head'}")

    try:
        repository, pr = _get_pull_request(owner, repo, pr_number, github_token)
        ref_to_use = ref or pr.head.sha
        contents = repository.get_contents(file_path, ref=ref_to_use)
        decoded = contents.decoded_content.decode("utf-8", errors="replace")

        return {
            "path": file_path,
            "ref": ref_to_use,
            "size": len(decoded),
            "encoding": "utf-8",
            "content": decoded,
        }

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to fetch file contents: {e}")


def list_repo_tree(
    owner: str,
    repo: str,
    pr_number: int,
    path: str | None = None,
    recursive: bool = True,
    ref: str | None = None,
    github_token: str | None = None,
) -> dict[str, Any]:
    """List repository entries beneath a path for the PR head."""
    logger.info(f"Listing repo tree for {owner}/{repo} path={path or '.'}")

    try:
        _, pr = _get_pull_request(owner, repo, pr_number, github_token)
        ref_to_use = ref or pr.head.sha
        tree_entries = _get_repo_tree(owner, repo, ref_to_use, github_token)

        normalized_path = (path or "").strip("/")
        results: list[dict[str, Any]] = []
        prefix = f"{normalized_path}/" if normalized_path else ""

        for entry in tree_entries:
            entry_path = entry["path"]

            if normalized_path:
                if entry_path == normalized_path:
                    results.append(entry)
                elif not entry_path.startswith(prefix):
                    continue
                elif not recursive:
                    relative = entry_path[len(prefix) :]
                    if "/" in relative:
                        continue
                    results.append(entry)
                else:
                    results.append(entry)
            else:
                if not recursive and "/" in entry_path:
                    continue
                results.append(entry)

        return {
            "ref": ref_to_use,
            "path": normalized_path or ".",
            "entries": results[:500],
            "total_entries": len(results),
            "recursive": recursive,
        }

    except GithubException as e:
        logger.error(f"GitHub API error: {e}")
        raise Exception(f"Failed to list repository tree: {e}")


def post_pr_comment(
    owner: str, repo: str, pr_number: int, comment_body: str, github_token: str | None = None
) -> dict[str, Any]:
    """
    Post a comment on a GitHub pull request.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: Pull request number
        comment_body: Comment text (supports markdown)
        github_token: GitHub API token

    Returns:
        Dictionary with comment information

    Raises:
        Exception: If posting comment fails
    """
    logger.info(f"Posting comment on {owner}/{repo}/pull/{pr_number}")

    try:
        g = _get_github_client(github_token)
        repository = g.get_repo(f"{owner}/{repo}")
        pr = repository.get_pull(pr_number)

        # Create a comment on the PR
        comment = pr.create_issue_comment(comment_body)

        return {
            "success": True,
            "comment_id": comment.id,
            "comment_url": comment.html_url,
            "message": "Comment posted successfully",
        }

    except GithubException as e:
        logger.error(f"GitHub API error when posting comment: {e}")
        raise Exception(f"Failed to post PR comment: {e}")
    except Exception as e:
        logger.error(f"Error posting comment: {e}")
        raise Exception(f"Failed to post PR comment: {e}")
