"""Tool execution dispatcher."""

import logging
from typing import Any, Literal

from .tool_definitions import TOOLS
from . import github_tools, gitlab_tools

logger = logging.getLogger(__name__)

Platform = Literal["github", "gitlab"]


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    platform: Platform,
    token: str | None = None,
    gitlab_url: str | None = None,
) -> Any:
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments for the tool
        platform: Platform (github or gitlab)
        token: API authentication token
        gitlab_url: GitLab instance URL (for self-hosted GitLab)

    Returns:
        Tool execution result

    Raises:
        ValueError: If tool name is unknown
    """
    logger.info(f"{tool_name}({', '.join(f'{k}={v}' for k, v in arguments.items())})")

    # Add token to arguments (using github_token param name for compatibility)
    tool_args = {**arguments, "github_token": token}

    # Add gitlab_url for GitLab platform
    if platform == "gitlab" and gitlab_url:
        tool_args["gitlab_url"] = gitlab_url

    # Select the appropriate module based on platform
    if platform == "github":
        module = github_tools
    elif platform == "gitlab":
        module = gitlab_tools
    else:
        raise ValueError(f"Unknown platform: {platform}")

    # Dispatch to appropriate tool
    if tool_name == "fetch_pr_metadata":
        return module.fetch_pr_metadata(**tool_args)
    elif tool_name == "fetch_pr_files":
        return module.fetch_pr_files(**tool_args)
    elif tool_name == "fetch_file_diff":
        return module.fetch_file_diff(**tool_args)
    elif tool_name == "fetch_pr_commits":
        return module.fetch_pr_commits(**tool_args)
    elif tool_name == "fetch_ci_status":
        return module.fetch_ci_status(**tool_args)
    elif tool_name == "search_tests":
        return module.search_tests(**tool_args)
    elif tool_name == "fetch_file_content":
        return module.fetch_file_content(**tool_args)
    elif tool_name == "list_repo_tree":
        return module.list_repo_tree(**tool_args)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# Export TOOLS for use in agent
__all__ = ["execute_tool", "TOOLS"]
