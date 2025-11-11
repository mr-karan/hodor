"""GitLab API tool implementations."""

import base64
import logging
import os
from threading import Lock
from typing import Any

import gitlab
from gitlab.exceptions import GitlabError

logger = logging.getLogger(__name__)

_MR_FILE_MAP_CACHE: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = {}
_REPO_TREE_CACHE: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
_MR_FILE_MAP_LOCK = Lock()
_REPO_TREE_LOCK = Lock()


def _get_gitlab_client(token: str | None = None, gitlab_url: str = "https://gitlab.com") -> gitlab.Gitlab:
    """Create authenticated GitLab client.

    Args:
        token: GitLab API token
        gitlab_url: GitLab instance URL (default: https://gitlab.com for public GitLab)
    """
    auth_token = token or os.getenv("GITLAB_TOKEN")

    if auth_token:
        gl = gitlab.Gitlab(gitlab_url, private_token=auth_token)
        gl.auth()
        return gl
    else:
        logger.warning("No GitLab token provided - API rate limits will be very low")
        return gitlab.Gitlab(gitlab_url)


def _get_gitlab_project(owner: str, repo: str, github_token: str | None, gitlab_url: str):
    """Return the project handle for a namespace/repo combination."""
    gl = _get_gitlab_client(github_token, gitlab_url)
    return gl.projects.get(f"{owner}/{repo}")


def _get_merge_request(
    owner: str, repo: str, pr_number: int, github_token: str | None, gitlab_url: str
) -> tuple[Any, Any]:
    """Fetch and return (project, merge_request)."""
    project = _get_gitlab_project(owner, repo, github_token, gitlab_url)
    mr = project.mergerequests.get(pr_number)
    return project, mr


def _get_mr_head_sha(mr: Any) -> str:
    """Return a stable identifier for the MR head commit."""
    if hasattr(mr, "sha") and mr.sha:
        return mr.sha
    diff_refs = getattr(mr, "diff_refs", None) or {}
    return diff_refs.get("head_sha") or diff_refs.get("head_sha".upper(), "unknown")


def _get_mr_changes_map(
    owner: str, repo: str, pr_number: int, github_token: str | None, gitlab_url: str
) -> dict[str, dict[str, Any]]:
    """Cache and return MR change metadata keyed by file path."""
    project, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)
    head_sha = _get_mr_head_sha(mr)
    cache_key = (owner, repo, pr_number, head_sha)

    with _MR_FILE_MAP_LOCK:
        cached = _MR_FILE_MAP_CACHE.get(cache_key)
        if cached is not None:
            return cached

    changes = mr.changes()
    files_map: dict[str, dict[str, Any]] = {}
    for change in changes["changes"]:
        diff_lines = change.get("diff", "").split("\n")
        additions = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
        filename = change["new_path"]

        if change.get("renamed_file"):
            files_map[change["old_path"]] = {
                "filename": change["old_path"],
                "status": "renamed",
                "additions": additions,
                "deletions": deletions,
                "changes": additions + deletions,
                "patch": change.get("diff", ""),
                "previous_filename": change.get("old_path"),
            }

        files_map[filename] = {
            "filename": filename,
            "status": (
                "added"
                if change.get("new_file")
                else (
                    "deleted" if change.get("deleted_file") else "renamed" if change.get("renamed_file") else "modified"
                )
            ),
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
            "patch": change.get("diff", ""),
            "previous_filename": change.get("old_path") if change.get("renamed_file") else None,
        }

    with _MR_FILE_MAP_LOCK:
        stale_keys = [key for key in _MR_FILE_MAP_CACHE if key[:3] == cache_key[:3] and key != cache_key]
        for stale_key in stale_keys:
            _MR_FILE_MAP_CACHE.pop(stale_key, None)
        _MR_FILE_MAP_CACHE[cache_key] = files_map

    return files_map


def _get_repo_tree(owner: str, repo: str, ref: str, github_token: str | None, gitlab_url: str) -> list[dict[str, Any]]:
    """Cache and return the repository tree at a given ref."""
    cache_key = (owner, repo, ref)
    with _REPO_TREE_LOCK:
        cached = _REPO_TREE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    project = _get_gitlab_project(owner, repo, github_token, gitlab_url)
    tree = project.repository_tree(ref=ref, recursive=True, all=True)

    with _REPO_TREE_LOCK:
        _REPO_TREE_CACHE[cache_key] = tree

    return tree


def parse_repo_url(repo_url: str) -> tuple[str, str, str | None, str]:
    """
    Parse GitLab repository URL to extract owner, repo, ref, and base URL.

    Examples:
        https://gitlab.com/owner/repo → ('owner', 'repo', None, 'https://gitlab.com')
        https://gitlab.example.com/owner/repo/-/tree/branch → ('owner', 'repo', 'branch', 'https://gitlab.example.com')
        https://gitlab.com/owner/repo/-/merge_requests/123 → ('owner', 'repo', None, 'https://gitlab.com')

    Returns:
        Tuple of (owner, repo, ref, gitlab_url)
    """
    from urllib.parse import urlparse

    parsed = urlparse(repo_url)
    path_parts = [p for p in parsed.path.split("/") if p]

    # Extract base URL (scheme + netloc)
    gitlab_url = f"{parsed.scheme}://{parsed.netloc}"

    if len(path_parts) >= 2:
        owner = path_parts[0]
        repo = path_parts[1]

        # Check if URL has a branch reference
        ref = None
        if len(path_parts) >= 4 and path_parts[2] == "-" and path_parts[3] == "tree":
            ref = path_parts[4] if len(path_parts) > 4 else None

        return owner, repo, ref, gitlab_url

    raise ValueError(f"Invalid GitLab URL format: {repo_url}")


def fetch_pr_metadata(
    owner: str, repo: str, pr_number: int, github_token: str | None = None, gitlab_url: str = "https://gitlab.com"
) -> dict[str, Any]:
    """
    Fetch merge request metadata including title, description, author, timestamps.

    Note: Parameter named 'github_token' for compatibility but uses GITLAB_TOKEN.

    Returns:
        Dictionary with MR metadata
    """
    logger.info(f"Fetching MR metadata for {owner}/{repo}/merge_requests/{pr_number}")

    try:
        _, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)

        return {
            "title": mr.title,
            "description": mr.description or "",
            "state": mr.state,  # opened, closed, merged
            "author": mr.author["username"],
            "created_at": mr.created_at,
            "updated_at": mr.updated_at,
            "merged": mr.merged_at is not None,
            "merged_at": mr.merged_at,
            "source_branch": mr.source_branch,
            "target_branch": mr.target_branch,
            "labels": mr.labels,
            "url": mr.web_url,
            "changes_count": mr.changes_count,
        }

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to fetch MR metadata: {e}")


def fetch_pr_files(
    owner: str, repo: str, pr_number: int, github_token: str | None = None, gitlab_url: str = "https://gitlab.com"
) -> dict[str, Any]:
    """
    Fetch list of all changed files with addition/deletion stats.

    Returns:
        Dictionary with list of changed files
    """
    logger.info(f"Fetching MR files for {owner}/{repo}/merge_requests/{pr_number}")

    try:
        files_map = _get_mr_changes_map(owner, repo, pr_number, github_token, gitlab_url)
        files = [
            {
                "filename": data["filename"],
                "status": data["status"],
                "additions": data["additions"],
                "deletions": data["deletions"],
                "changes": data["changes"],
                "patch": (data["patch"][:500] if data["patch"] else None),
                "previous_filename": data.get("previous_filename"),
            }
            for data in files_map.values()
        ]

        return {"total_files": len(files), "files": files}

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to fetch MR files: {e}")


def fetch_file_diff(
    owner: str,
    repo: str,
    pr_number: int,
    file_path: str,
    github_token: str | None = None,
    gitlab_url: str = "https://gitlab.com",
) -> dict[str, Any]:
    """
    Fetch detailed diff for a specific file.

    Returns:
        Dictionary with file diff information
    """
    logger.info(f"Fetching diff for {file_path} in {owner}/{repo}/merge_requests/{pr_number}")

    try:
        files_map = _get_mr_changes_map(owner, repo, pr_number, github_token, gitlab_url)
        if file_path not in files_map:
            return {"error": f"File {file_path} not found in MR"}

        data = files_map[file_path]
        return {
            "filename": data["filename"],
            "status": data["status"],
            "additions": data["additions"],
            "deletions": data["deletions"],
            "changes": data["changes"],
            "patch": data["patch"],
            "previous_filename": data.get("previous_filename"),
        }

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to fetch file diff: {e}")


def fetch_pr_commits(
    owner: str, repo: str, pr_number: int, github_token: str | None = None, gitlab_url: str = "https://gitlab.com"
) -> dict[str, Any]:
    """
    Fetch list of commits in the merge request.

    Returns:
        Dictionary with commit list
    """
    logger.info(f"Fetching commits for {owner}/{repo}/merge_requests/{pr_number}")

    try:
        _, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)

        commits = []
        for commit in mr.commits():
            commits.append(
                {
                    "sha": commit["id"],
                    "message": commit["message"],
                    "author": commit["author_name"],
                    "author_email": commit["author_email"],
                    "date": commit["created_at"],
                    "url": commit["web_url"],
                }
            )

        return {"total_commits": len(commits), "commits": commits}

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to fetch MR commits: {e}")


def fetch_ci_status(
    owner: str, repo: str, pr_number: int, github_token: str | None = None, gitlab_url: str = "https://gitlab.com"
) -> dict[str, Any]:
    """
    Fetch CI/CD pipeline status for the merge request.

    Returns:
        Dictionary with CI status information
    """
    logger.info(f"Fetching CI status for {owner}/{repo}/merge_requests/{pr_number}")

    try:
        project, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)

        pipelines = mr.pipelines.list()
        if not pipelines:
            return {"error": "No pipelines found for this MR", "pipelines": []}

        latest_pipeline = project.pipelines.get(pipelines[0]["id"])
        jobs = latest_pipeline.jobs.list()

        job_list = [
            {
                "name": job.name,
                "status": job.status,
                "stage": job.stage,
                "started_at": getattr(job, "started_at", None),
                "finished_at": getattr(job, "finished_at", None),
                "url": job.web_url,
            }
            for job in jobs
        ]

        return {
            "pipeline_id": latest_pipeline.id,
            "pipeline_status": latest_pipeline.status,
            "pipeline_url": latest_pipeline.web_url,
            "total_jobs": len(job_list),
            "jobs": job_list,
        }

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to fetch CI status: {e}")


def search_tests(
    owner: str,
    repo: str,
    pr_number: int,
    file_path: str,
    github_token: str | None = None,
    gitlab_url: str = "https://gitlab.com",
) -> dict[str, Any]:
    """
    Search for test files related to the given source file.

    Returns:
        Dictionary with list of potential test files
    """
    logger.info(f"Searching for tests related to {file_path} in {owner}/{repo}")

    try:
        import os.path

        project, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)
        ref = _get_mr_head_sha(mr)

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

        tree_entries = _get_repo_tree(owner, repo, ref, github_token, gitlab_url)
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
                    "id": entry["id"],
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
            "note": "Results derived from repository tree scan of MR head",
        }

    except Exception as e:
        logger.error(f"Error searching tests: {e}")
        raise Exception(f"Failed to search tests: {e}")


def fetch_file_content(
    owner: str,
    repo: str,
    pr_number: int,
    file_path: str,
    ref: str | None = None,
    github_token: str | None = None,
    gitlab_url: str = "https://gitlab.com",
) -> dict[str, Any]:
    """Return the file contents from the MR head or supplied ref."""
    logger.info(f"Fetching file contents for {file_path} in {owner}/{repo}@{ref or 'MR head'}")

    try:
        project, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)
        ref_to_use = ref or _get_mr_head_sha(mr)
        remote_file = project.files.get(file_path=file_path, ref=ref_to_use)
        decoded_bytes = base64.b64decode(remote_file.content)
        decoded = decoded_bytes.decode("utf-8", errors="replace")

        return {
            "path": file_path,
            "ref": ref_to_use,
            "size": len(decoded),
            "encoding": "utf-8",
            "content": decoded,
        }

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to fetch file contents: {e}")


def list_repo_tree(
    owner: str,
    repo: str,
    pr_number: int,
    path: str | None = None,
    recursive: bool = True,
    ref: str | None = None,
    github_token: str | None = None,
    gitlab_url: str = "https://gitlab.com",
) -> dict[str, Any]:
    """List repository entries beneath a path for the MR head."""
    logger.info(f"Listing repo tree for {owner}/{repo} path={path or '.'}")

    try:
        _, mr = _get_merge_request(owner, repo, pr_number, github_token, gitlab_url)
        ref_to_use = ref or _get_mr_head_sha(mr)
        tree_entries = _get_repo_tree(owner, repo, ref_to_use, github_token, gitlab_url)

        normalized_path = (path or "").strip("/")
        prefix = f"{normalized_path}/" if normalized_path else ""
        results: list[dict[str, Any]] = []

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

    except GitlabError as e:
        logger.error(f"GitLab API error: {e}")
        raise Exception(f"Failed to list repository tree: {e}")


def post_mr_comment(
    owner: str,
    repo: str,
    mr_number: int,
    comment_body: str,
    github_token: str | None = None,
    gitlab_url: str = "https://gitlab.com",
) -> dict[str, Any]:
    """
    Post a comment on a GitLab merge request.

    Args:
        owner: Project owner/namespace
        repo: Project name
        mr_number: Merge request number
        comment_body: Comment text (supports markdown)
        github_token: GitLab API token (named for compatibility)
        gitlab_url: GitLab instance URL (default: https://gitlab.com for public GitLab)

    Returns:
        Dictionary with comment information

    Raises:
        Exception: If posting comment fails
    """
    logger.info(f"Posting comment on {owner}/{repo}/merge_requests/{mr_number}")

    try:
        gl = _get_gitlab_client(github_token, gitlab_url)
        project = gl.projects.get(f"{owner}/{repo}")
        mr = project.mergerequests.get(mr_number)

        # Create a note (comment) on the MR
        note = mr.notes.create({"body": comment_body})

        return {
            "success": True,
            "comment_id": note.id,
            "comment_url": f"{gitlab_url}/{owner}/{repo}/-/merge_requests/{mr_number}#note_{note.id}",
            "message": "Comment posted successfully",
        }

    except GitlabError as e:
        logger.error(f"GitLab API error when posting comment: {e}")
        raise Exception(f"Failed to post MR comment: {e}")
    except Exception as e:
        logger.error(f"Error posting comment: {e}")
        raise Exception(f"Failed to post MR comment: {e}")
