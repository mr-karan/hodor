"""PR Review Prompt Builder for OpenHands-based Hodor.

This module provides prompt templates and builders for conducting PR reviews
using OpenHands' bash-based tool system instead of custom API tools.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Get the directory containing this file
PROMPTS_DIR = Path(__file__).parent
TEMPLATES_DIR = PROMPTS_DIR / "templates"


def build_pr_review_prompt(
    pr_url: str,
    owner: str,
    repo: str,
    pr_number: str,
    platform: str,
    target_branch: str = "main",
    diff_base_sha: str | None = None,
    custom_instructions: str | None = None,
    custom_prompt_file: Path | None = None,
) -> str:
    """Build a PR review prompt for OpenHands agent.

    Args:
        pr_url: Full PR URL
        owner: Repository owner
        repo: Repository name
        pr_number: PR number
        platform: "github" or "gitlab"
        target_branch: Target/base branch of the PR (e.g., "main", "develop")
        diff_base_sha: GitLab's calculated merge base SHA (most reliable for GitLab CI)
        custom_instructions: Optional custom prompt text (inline)
        custom_prompt_file: Optional path to custom prompt file

    Returns:
        Complete prompt for OpenHands agent
    """
    # Priority: custom_instructions > custom_prompt_file > default template
    if custom_instructions:
        logger.info("Using custom inline prompt")
        return custom_instructions

    # Determine which template to use
    template_file = custom_prompt_file if custom_prompt_file else TEMPLATES_DIR / "default_review.md"

    logger.info(f"Loading prompt from template: {template_file}")
    try:
        with open(template_file, "r", encoding="utf-8") as f:
            template_text = f.read()
    except FileNotFoundError:
        logger.warning(f"Template file not found: {template_file}, using built-in prompt")
        template_text = None
    except Exception as e:
        logger.error(f"Failed to load template file: {e}")
        raise

    # Prepare platform-specific commands and explanations for interpolation
    if platform == "github":
        cli_tool = "gh"
        pr_view_cmd = f"gh pr view {pr_number}"
        pr_diff_cmd = f"gh pr diff {pr_number}"
        pr_checks_cmd = f"gh pr checks {pr_number}"
        # GitHub specific diff command (fallback)
        git_diff_cmd = f"git --no-pager diff origin/{target_branch}...HEAD"
    else:  # gitlab
        cli_tool = "glab"
        pr_view_cmd = f"glab mr view {pr_number}"
        pr_diff_cmd = f"glab mr diff {pr_number}"
        pr_checks_cmd = f"glab ci view"
        # GitLab specific diff command - use diff_base_sha if available (most reliable)
        if diff_base_sha:
            git_diff_cmd = f"git --no-pager diff {diff_base_sha} HEAD"
            logger.info(f"Using GitLab CI_MERGE_REQUEST_DIFF_BASE_SHA: {diff_base_sha[:8]}")
        else:
            git_diff_cmd = f"git --no-pager diff origin/{target_branch}...HEAD"

    # Prepare diff explanation based on platform and available SHA
    if diff_base_sha:
        diff_explanation = (
            f"**GitLab CI Advantage**: This uses GitLab's pre-calculated merge base SHA "
            f"(`CI_MERGE_REQUEST_DIFF_BASE_SHA`), which matches exactly what the GitLab UI shows. "
            f"This is more reliable than three-dot syntax because it handles force pushes, rebases, "
            f"and messy histories correctly."
        )
    else:
        diff_explanation = (
            f"**Three-dot syntax** shows ONLY changes introduced on the source branch, "
            f"excluding changes already on `{target_branch}`."
        )

    # Template must load successfully - no fallback
    if not template_text:
        raise RuntimeError(
            f"Failed to load prompt template from {template_file}. "
            f"Template file is required for code review."
        )

    # Interpolate template variables
    try:
        prompt = template_text.format(
            pr_url=pr_url,
            pr_diff_cmd=pr_diff_cmd,
            git_diff_cmd=git_diff_cmd,
            target_branch=target_branch,
            diff_explanation=diff_explanation,
        )
        logger.info("Successfully interpolated template")
        return prompt
    except KeyError as e:
        raise RuntimeError(
            f"Template interpolation failed - missing variable: {e}. "
            f"Template file: {template_file}"
        ) from e

    # OLD BUILT-IN FALLBACK REMOVED - Template is now required
    # This eliminates duplication and makes templates the single source of truth
