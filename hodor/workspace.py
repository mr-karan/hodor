"""Workspace management for PR review operations.

Handles cloning repositories and checking out PR branches for review.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from .gitlab import GitLabAPIError, fetch_gitlab_mr_info

logger = logging.getLogger(__name__)

Platform = Literal["github", "gitlab"]


class WorkspaceError(Exception):
    """Raised when workspace setup fails."""

    pass


def _detect_ci_workspace(owner: str, repo: str, pr_number: str) -> tuple[Path | None, str | None, str | None]:
    """Detect if running in CI environment with repo already cloned.

    Checks for GitLab CI and GitHub Actions environments.

    Args:
        owner: Repository owner/group
        repo: Repository name
        pr_number: Pull/merge request number

    Returns:
        Tuple of (workspace_path, target_branch, diff_base_sha) if in CI, (None, None, None) otherwise
        diff_base_sha is only available for GitLab CI (CI_MERGE_REQUEST_DIFF_BASE_SHA)
    """
    # GitLab CI detection
    if os.getenv("GITLAB_CI") == "true":
        project_dir = os.getenv("CI_PROJECT_DIR")
        project_path = os.getenv("CI_PROJECT_PATH")  # e.g., "group/subgroup/repo"
        mr_iid = os.getenv("CI_MERGE_REQUEST_IID")
        target_branch = os.getenv("CI_MERGE_REQUEST_TARGET_BRANCH_NAME")  # e.g., "main", "develop"
        diff_base_sha = os.getenv("CI_MERGE_REQUEST_DIFF_BASE_SHA")  # GitLab's calculated merge base

        if project_dir and project_path:
            # Check if this matches our target repo
            expected_path = f"{owner}/{repo}"
            if project_path == expected_path or project_path.endswith(f"/{expected_path}"):
                logger.info(
                    f"Detected GitLab CI environment (MR IID: {mr_iid}, target: {target_branch or 'unknown'}, "
                    f"diff_base_sha: {diff_base_sha[:8] if diff_base_sha else 'unknown'})"
                )
                return Path(project_dir), target_branch, diff_base_sha

    # GitHub Actions detection
    if os.getenv("GITHUB_ACTIONS") == "true":
        workspace_dir = os.getenv("GITHUB_WORKSPACE")
        repository = os.getenv("GITHUB_REPOSITORY")  # e.g., "owner/repo"
        base_ref = os.getenv("GITHUB_BASE_REF")  # Base branch for PRs

        if workspace_dir and repository:
            expected_repo = f"{owner}/{repo}"
            if repository == expected_repo:
                logger.info(f"Detected GitHub Actions environment (base: {base_ref or 'unknown'})")
                return Path(workspace_dir), base_ref, None  # GitHub doesn't provide diff_base_sha

    return None, None, None


def _is_same_repo(workspace: Path, platform: Platform, owner: str, repo: str) -> bool:
    """Check if workspace contains the same repository.

    Args:
        workspace: Workspace directory
        platform: "github" or "gitlab"
        owner: Repository owner/group
        repo: Repository name

    Returns:
        True if workspace has the same repo, False otherwise
    """
    git_dir = workspace / ".git"
    if not git_dir.exists():
        return False

    try:
        # Get remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        )
        remote_url = result.stdout.strip()

        # Check if it matches expected repo
        repo_identifier = f"{owner}/{repo}"
        return repo_identifier in remote_url
    except subprocess.CalledProcessError:
        return False


def setup_workspace(
    platform: Platform,
    owner: str,
    repo: str,
    pr_number: str,
    host: str | None = None,
    base_branch: str | None = None,
    working_dir: Path | None = None,
    reuse: bool = True,
) -> tuple[Path, str, str | None]:
    """Setup workspace by cloning repo and checking out PR branch.

    Args:
        platform: "github" or "gitlab"
        owner: Repository owner/group
        repo: Repository name
        pr_number: Pull request number
        host: Git host (e.g., 'github.com', 'gitlab.example.com'). Optional, uses default if not provided.
        base_branch: Base branch name (optional, auto-detected if not provided)
        working_dir: Directory to use (if None, creates temp directory)
        reuse: If True and workspace exists with same repo, reuse it (faster)

    Returns:
        Tuple of (workspace_path, target_branch, diff_base_sha)
        diff_base_sha is only available for GitLab CI (None for other environments)

    Raises:
        WorkspaceError: If setup fails
    """
    try:
        # Check if running in CI environment with repo already cloned
        ci_workspace, ci_target_branch, ci_diff_base_sha = _detect_ci_workspace(owner, repo, pr_number)
        detected_target_branch = ci_target_branch  # Track detected target branch
        detected_diff_base_sha = ci_diff_base_sha  # Track diff base SHA (GitLab only)

        if ci_workspace:
            workspace = ci_workspace
            # Skip cloning, just setup the PR branch
        elif working_dir is None:
            workspace = Path(tempfile.mkdtemp(prefix="hodor-review-"))
            logger.info(f"Created temporary workspace: {workspace}")
        else:
            workspace = working_dir
            workspace.mkdir(parents=True, exist_ok=True)

            # Check if we can reuse existing workspace
            if reuse and _is_same_repo(workspace, platform, owner, repo):
                logger.info(f"Reusing existing workspace: {workspace}")
                # Just fetch latest and checkout PR branch
                subprocess.run(
                    ["git", "fetch", "origin"],
                    cwd=workspace,
                    check=True,
                    capture_output=True,
                )
                logger.info("Fetched latest changes")
            else:
                logger.info(f"Using workspace: {workspace}")

        # Skip workspace setup entirely if in CI (repo already cloned and on the right branch)
        if not ci_workspace:
            if platform == "github":
                target_branch = _setup_github_workspace(workspace, owner, repo, pr_number)
                if detected_target_branch is None:
                    detected_target_branch = target_branch
            elif platform == "gitlab":
                target_branch = _setup_gitlab_workspace(workspace, owner, repo, pr_number, host)
                if detected_target_branch is None:
                    detected_target_branch = target_branch
            else:
                raise WorkspaceError(f"Unsupported platform: {platform}")

        # Fallback to "main" if target branch couldn't be detected
        final_target_branch = detected_target_branch or base_branch or "main"
        logger.info(
            f"Workspace ready at: {workspace} (target branch: {final_target_branch}, "
            f"diff_base_sha: {detected_diff_base_sha[:8] if detected_diff_base_sha else 'N/A'})"
        )
        return workspace, final_target_branch, detected_diff_base_sha

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        logger.error(f"Exit code: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        raise WorkspaceError(f"Failed to setup workspace: {e}") from e
    except Exception as e:
        logger.error(f"Workspace setup failed: {e}")
        raise WorkspaceError(f"Failed to setup workspace: {e}") from e


def _setup_github_workspace(workspace: Path, owner: str, repo: str, pr_number: str) -> str:
    """Setup GitHub PR workspace using gh CLI.

    Args:
        workspace: Target workspace directory
        owner: Repository owner
        repo: Repository name
        pr_number: PR number

    Returns:
        Base branch name (target branch of the PR)
    """
    logger.info(f"Setting up GitHub workspace for {owner}/{repo}/pull/{pr_number}")

    # Check if gh CLI is available
    try:
        subprocess.run(
            ["gh", "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise WorkspaceError(
            f"GitHub CLI (gh) is not available. Please install it: https://cli.github.com\n" f"Error: {e}"
        ) from e

    # Check if we have authentication credentials
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning("GITHUB_TOKEN not set. Ensure you're authenticated with: gh auth login")

    # Clone repository
    logger.info(f"Cloning repository {owner}/{repo}...")
    try:
        subprocess.run(
            ["gh", "repo", "clone", f"{owner}/{repo}", str(workspace)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, "stderr") and e.stderr else str(e)
        logger.error(f"gh repo clone failed: {error_msg}")
        raise WorkspaceError(
            f"Failed to clone repository {owner}/{repo}\n"
            f"Command: gh repo clone {owner}/{repo}\n"
            f"Error: {error_msg}\n"
            f"Troubleshooting:\n"
            f"  1. Verify repository exists: https://github.com/{owner}/{repo}\n"
            f"  2. Check authentication: gh auth status\n"
            f"  3. Verify GITHUB_TOKEN is set and has repo access"
        ) from e

    # Change to workspace directory
    original_dir = Path.cwd()
    os.chdir(workspace)

    try:
        # Checkout PR branch
        logger.info(f"Checking out PR #{pr_number}...")
        try:
            subprocess.run(
                ["gh", "pr", "checkout", pr_number],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, "stderr") and e.stderr else str(e)
            logger.error(f"gh pr checkout failed: {error_msg}")
            raise WorkspaceError(
                f"Failed to checkout PR #{pr_number}\n"
                f"Command: gh pr checkout {pr_number}\n"
                f"Error: {error_msg}\n"
                f"Verify PR exists: https://github.com/{owner}/{repo}/pull/{pr_number}"
            ) from e

        # Get PR info for base branch detection
        base_branch = "main"  # Default fallback
        try:
            result = subprocess.run(
                ["gh", "pr", "view", pr_number, "--json", "headRefName,baseRefName"],
                check=True,
                capture_output=True,
                text=True,
            )
            import json

            pr_info = json.loads(result.stdout)
            base_branch = pr_info.get("baseRefName", "main")
            logger.info(f"Checked out PR branch: {pr_info.get('headRefName')}")
            logger.info(f"Base branch: {base_branch}")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            # Non-critical - just log and continue
            logger.warning(f"Could not fetch PR metadata: {e}")

    finally:
        # Restore original directory
        os.chdir(original_dir)

    return base_branch


def _setup_gitlab_workspace(workspace: Path, owner: str, repo: str, pr_number: str, host: str | None = None) -> str:
    """Setup GitLab MR workspace using glab CLI.

    Supports self-hosted GitLab instances via host parameter or GITLAB_HOST environment variable.

    Args:
        workspace: Target workspace directory
        owner: Repository owner/group
        repo: Repository name
        pr_number: Merge request number
        host: GitLab host (e.g., 'gitlab.com', 'gitlab.example.com'). Falls back to GITLAB_HOST env var or 'gitlab.com'.

    Returns:
        Target branch name (base branch of the MR)
    """
    # Priority: host parameter > GITLAB_HOST env var > default to gitlab.com
    gitlab_host = host or os.getenv("GITLAB_HOST", "gitlab.com")
    logger.info(f"Setting up GitLab workspace for {owner}/{repo}/merge_requests/{pr_number}")
    logger.info(f"GitLab host: {gitlab_host}")

    # Check if glab CLI is available
    # Note: We skip auth status check because `glab auth status` checks ALL hosts
    # and fails if any host (like gitlab.com) has issues, even if our target host is fine.
    # Instead, we'll let the git clone operation fail naturally if auth is bad.
    try:
        subprocess.run(
            ["glab", "version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise WorkspaceError(
            f"GitLab CLI (glab) is not available. Please install it: https://gitlab.com/gitlab-org/cli\n" f"Error: {e}"
        ) from e

    # Check if we have authentication credentials
    gitlab_token = os.getenv("GITLAB_TOKEN")
    if not gitlab_token:
        logger.warning(
            f"GITLAB_TOKEN not set. Ensure you're authenticated with: glab auth login --hostname {gitlab_host}"
        )

    # Clone repository
    # Format: owner/repo or group/subgroup/repo
    repo_full_path = f"{owner}/{repo}"
    logger.info(f"Cloning repository {repo_full_path}...")

    # Use HTTPS clone URL for consistency
    clone_url = f"https://{gitlab_host}/{owner}/{repo}.git"

    subprocess.run(
        ["git", "clone", clone_url, str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )

    # Change to workspace directory
    original_dir = Path.cwd()
    os.chdir(workspace)

    try:
        # GitLab uses "merge requests" (MR) instead of "pull requests" (PR)
        logger.info(f"Checking out MR !{pr_number}...")

        # Get MR info to find the source branch
        # Specify repo explicitly to avoid ambiguity
        try:
            mr_info = fetch_gitlab_mr_info(owner, repo, pr_number, gitlab_host)
        except GitLabAPIError as e:
            error_msg = str(e)
            raise WorkspaceError(
                f"Failed to fetch MR info for !{pr_number} from {gitlab_host}/{repo_full_path}\n"
                f"Command: glab mr view {pr_number} --repo {repo_full_path} -F json\n"
                f"Error: {error_msg}\n"
                f"Troubleshooting:\n"
                f"  1. Verify MR exists: https://{gitlab_host}/{repo_full_path}/-/merge_requests/{pr_number}\n"
                f"  2. Check authentication: glab auth status\n"
                f"  3. Verify GITLAB_TOKEN is set for {gitlab_host}"
            ) from e

        source_branch = mr_info.get("source_branch")
        target_branch = mr_info.get("target_branch")

        if not source_branch:
            raise WorkspaceError(
                f"Could not determine source branch for MR !{pr_number}. "
                f"MR info: {mr_info.get('title', 'N/A')} (state: {mr_info.get('state', 'unknown')})"
            )

        logger.info(f"Source branch: {source_branch}, Target branch: {target_branch}")

        # Fetch all branches to ensure we have the source branch
        try:
            fetch_result = subprocess.run(
                ["git", "fetch", "--all"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"Git fetch completed: {fetch_result.stdout[:100]}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, "stderr") and e.stderr else str(e)
            logger.warning(f"Git fetch had issues (continuing anyway): {error_msg}")

        # Checkout the source branch (try origin/branch first, then just branch name)
        try:
            logger.info(f"Attempting checkout: git checkout -b {source_branch} origin/{source_branch}")
            subprocess.run(
                ["git", "checkout", "-b", source_branch, f"origin/{source_branch}"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(f"Successfully checked out MR branch: {source_branch}")
        except subprocess.CalledProcessError as e1:
            # If that fails, try checking out the branch directly
            logger.debug(
                f"First checkout attempt failed, trying direct checkout: {e1.stderr if hasattr(e1, 'stderr') else ''}"
            )
            try:
                subprocess.run(
                    ["git", "checkout", source_branch],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Checked out existing branch: {source_branch}")
            except subprocess.CalledProcessError as e2:
                error_msg = e2.stderr if hasattr(e2, "stderr") and e2.stderr else str(e2)
                logger.error(f"Failed to checkout branch {source_branch}: {error_msg}")
                raise WorkspaceError(
                    f"Failed to checkout MR branch '{source_branch}'\n"
                    f"Tried:\n"
                    f"  1. git checkout -b {source_branch} origin/{source_branch}\n"
                    f"  2. git checkout {source_branch}\n"
                    f"Error: {error_msg}\n"
                    f"Available branches: Run 'git branch -a' in workspace to debug"
                ) from e2

    finally:
        # Restore original directory
        os.chdir(original_dir)

    # Return target branch (fallback to "main" if not available)
    return target_branch or "main"


def cleanup_workspace(workspace: Path) -> None:
    """Clean up workspace directory.

    Args:
        workspace: Workspace directory to remove
    """
    import shutil

    try:
        if workspace.exists() and workspace.is_dir():
            shutil.rmtree(workspace)
            logger.info(f"Cleaned up workspace: {workspace}")
    except Exception as e:
        logger.warning(f"Failed to cleanup workspace {workspace}: {e}")
