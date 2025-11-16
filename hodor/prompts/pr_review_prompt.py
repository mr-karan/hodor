"""PR Review Prompt Builder for OpenHands-based Hodor.

This module provides prompt templates and builders for conducting PR reviews
using OpenHands' bash-based tool system instead of custom API tools.
"""

import logging
from pathlib import Path
from typing import Any

from ..gitlab import summarize_gitlab_notes

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
    mr_metadata: dict[str, Any] | None = None,
    custom_instructions: str | None = None,
    custom_prompt_file: Path | None = None,
    output_format: str = "markdown",
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
        custom_instructions: Optional additional instructions to append to template
        custom_prompt_file: Optional path to custom template file (replaces base template)
        output_format: Output format - "markdown" or "json" (default: "markdown")

    Returns:
        Complete prompt for OpenHands agent
    """
    # Step 1: Determine which template to use
    # custom_prompt_file = full replacement, otherwise use format-specific template
    if custom_prompt_file:
        template_file = custom_prompt_file
        logger.info(f"Using custom prompt file: {template_file}")
    elif output_format == "json":
        template_file = TEMPLATES_DIR / "json_review.md"
        logger.info("Using JSON review template")
    else:
        template_file = TEMPLATES_DIR / "default_review.md"
        logger.info("Using default markdown template")

    # Step 2: Load template
    logger.info(f"Loading template from: {template_file}")
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
        # Use git diff --name-only instead of gh pr diff to avoid dumping full diff
        pr_diff_cmd = f"git --no-pager diff origin/{target_branch}...HEAD --name-only"
        # GitHub specific diff command (base)
        git_diff_cmd = f"git --no-pager diff origin/{target_branch}...HEAD"
    else:  # gitlab
        # GitLab specific diff command - use diff_base_sha if available (most reliable)
        if diff_base_sha:
            # Use git diff --name-only to list files first, not full diff
            pr_diff_cmd = f"git --no-pager diff {diff_base_sha} HEAD --name-only"
            git_diff_cmd = f"git --no-pager diff {diff_base_sha} HEAD"
            logger.info(f"Using GitLab CI_MERGE_REQUEST_DIFF_BASE_SHA: {diff_base_sha[:8]}")
        else:
            pr_diff_cmd = f"git --no-pager diff origin/{target_branch}...HEAD --name-only"
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

    # Step 3: Interpolate template variables
    mr_context_section, mr_notes_section, mr_reminder_section = _build_mr_sections(mr_metadata)

    try:
        prompt = template_text.format(
            pr_url=pr_url,
            pr_diff_cmd=pr_diff_cmd,
            git_diff_cmd=git_diff_cmd,
            target_branch=target_branch,
            diff_explanation=diff_explanation,
            mr_context_section=mr_context_section,
            mr_notes_section=mr_notes_section,
            mr_reminder_section=mr_reminder_section,
        )
        logger.info("Successfully interpolated template")
    except KeyError as e:
        raise RuntimeError(
            f"Template interpolation failed - missing variable: {e}. "
            f"Template file: {template_file}"
        ) from e

    # Step 4: Append custom_instructions if provided
    if custom_instructions:
        prompt += f"\n\n## Additional Instructions\n\n{custom_instructions}\n"
        logger.info("Appended custom instructions to prompt")

    return prompt


def _build_mr_sections(mr_metadata: dict[str, Any] | None) -> tuple[str, str, str]:
    if not mr_metadata:
        return "", "", ""

    context_lines: list[str] = []
    title = mr_metadata.get("title")
    if title:
        context_lines.append(f"- Title: {title}")

    author = mr_metadata.get("author", {}).get("username") or mr_metadata.get("author", {}).get("name")
    if author:
        context_lines.append(f"- Author: @{author}")

    source_branch = mr_metadata.get("source_branch")
    target_branch = mr_metadata.get("target_branch")
    if source_branch and target_branch:
        context_lines.append(f"- Branches: {source_branch} → {target_branch}")

    changes = mr_metadata.get("changes_count")
    if changes:
        context_lines.append(f"- Files changed: {changes}")

    pipeline = mr_metadata.get("pipeline") or {}
    pipeline_status = pipeline.get("status")
    pipeline_url = pipeline.get("web_url")
    if pipeline_status:
        status_text = pipeline_status.replace("_", " ")
        if pipeline_url:
            context_lines.append(f"- Pipeline: {status_text} ({pipeline_url})")
        else:
            context_lines.append(f"- Pipeline: {status_text}")

    label_names = _normalize_label_names(mr_metadata.get("label_details"))
    if not label_names:
        label_names = _normalize_label_names(mr_metadata.get("labels"))
    if label_names:
        context_lines.append(f"- Labels: {', '.join(label_names)}")

    description = mr_metadata.get("description", "").strip()
    description_section = ""
    if description:
        description_section = "**Author Description:**\n" + _truncate_block(description, 800)

    context_section = ""
    if context_lines or description_section:
        context_section = "## MR Context\n" + "\n".join(context_lines)
        if description_section:
            context_section += "\n\n" + description_section
        context_section += "\n"

    notes_section = ""
    notes = mr_metadata.get("Notes")
    notes_summary = summarize_gitlab_notes(notes)
    if notes_summary:
        notes_section = f"## Existing MR Notes\n{notes_summary}\n"

    reminder_section = ""
    if notes_summary:
        reminder_section = (
            "## Review Note Deduplication\n\n"
            "The discussions above may already cover some issues. Before reporting a finding:\n"
            "1. Check if it's already mentioned in existing notes\n"
            "2. Only report if your finding is materially different or more specific\n"
            "3. If an existing note is incorrect/outdated, explain why in your finding\n\n"
            "Focus on discovering NEW issues not yet discussed.\n"
        )

    return context_section, notes_section, reminder_section


def _truncate_block(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _normalize_label_names(raw_labels: Any) -> list[str]:
    """Convert assorted GitLab label payloads into user-facing strings."""
    if not raw_labels:
        return []

    # GitLab REST returns strings, python-gitlab may expose dicts, keep both.
    label_names: list[str] = []

    def add_label(value: Any) -> None:
        name = ""
        if isinstance(value, str):
            name = value.strip()
        elif isinstance(value, dict):
            label_value = value.get("name")
            if isinstance(label_value, str):
                name = label_value.strip()
        elif value is not None:
            name = str(value).strip()
        if name:
            label_names.append(name)

    if isinstance(raw_labels, list):
        for label in raw_labels:
            add_label(label)
    else:
        add_label(raw_labels)

    return label_names


