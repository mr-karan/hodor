"""Tool definitions for LiteLLM."""

# Tool definitions following LiteLLM format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_pr_metadata",
            "description": "Get PR title, description, author, timestamps, labels, and status",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                },
                "required": ["owner", "repo", "pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pr_files",
            "description": "List all changed files with addition/deletion stats. Essential for understanding PR scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                },
                "required": ["owner", "repo", "pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_file_diff",
            "description": "Get detailed unified diff for a specific file. Use this to analyze actual code changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                    "file_path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["owner", "repo", "pr_number", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pr_commits",
            "description": "Get list of commits in the PR with messages and metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                },
                "required": ["owner", "repo", "pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_ci_status",
            "description": "Get CI/CD check status (passed/failed/pending) for the PR",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request number"},
                },
                "required": ["owner", "repo", "pr_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tests",
            "description": "Find test files related to a specific source file. Useful for checking test coverage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request or merge request number"},
                    "file_path": {"type": "string", "description": "Source file path to find tests for"},
                },
                "required": ["owner", "repo", "pr_number", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_file_content",
            "description": "Fetch the full file contents from the PR head (or a supplied ref).",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner or namespace"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request or merge request number"},
                    "file_path": {"type": "string", "description": "Path to the file in the repository"},
                    "ref": {
                        "type": "string",
                        "description": "Optional commit SHA or branch to read from (defaults to PR head)",
                    },
                },
                "required": ["owner", "repo", "pr_number", "file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repo_tree",
            "description": "List repository files/directories at the PR head to understand project layout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner or namespace"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "pr_number": {"type": "integer", "description": "Pull request or merge request number"},
                    "path": {
                        "type": "string",
                        "description": "Optional subdirectory to list (defaults to repository root)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to include nested paths (defaults to true)",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Optional commit SHA or branch (defaults to PR head)",
                    },
                },
                "required": ["owner", "repo", "pr_number"],
            },
        },
    },
]
