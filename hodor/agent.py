"""Core agent loop for PR review."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import logging
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
import litellm
from litellm import completion

from .tools import github_tools
from .tools.tool_executor import execute_tool, TOOLS

# Load environment variables
load_dotenv()

# Drop unsupported params for models that don't support them
litellm.drop_params = True


def configure_litellm_logging(verbose: bool = False) -> None:
    """Keep LiteLLM chatter under control unless the user asks for it."""
    try:
        litellm.suppress_debug_info = not verbose
    except AttributeError:
        # Older LiteLLM versions may not expose this flag.
        pass

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.INFO if verbose else logging.WARNING)


# Default to quiet logging at import time
configure_litellm_logging(False)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

Platform = Literal["github", "gitlab"]


@dataclass
class ReviewContext:
    """State shared across tool calls to enforce coverage guarantees."""

    owner: str
    repo: str
    pr_number: int
    files_to_cover: set[str] = field(default_factory=set)
    diffed_files: set[str] = field(default_factory=set)
    lock: Lock = field(default_factory=Lock, repr=False)

    def update_files_from_tool(self, tool_result: dict[str, Any]) -> None:
        """Extract filenames that need diff coverage."""
        files = tool_result.get("files") if isinstance(tool_result, dict) else None
        if not files:
            return

        tracked: set[str] = set()
        for file_entry in files:
            filename = file_entry.get("filename")
            if not filename:
                continue
            if self._is_reviewable_file(file_entry):
                tracked.add(filename)
                previous = file_entry.get("previous_filename")
                if previous:
                    tracked.add(previous)

        if not tracked:
            return

        with self.lock:
            if self.files_to_cover:
                self.files_to_cover.update(tracked)
            else:
                self.files_to_cover = tracked

    @staticmethod
    def _is_reviewable_file(file_entry: dict[str, Any]) -> bool:
        """Heuristic: only track files with text patches (skip binaries/docs)."""
        patch = file_entry.get("patch")
        if patch:
            return True

        filename = (file_entry.get("filename") or "").lower()
        skip_suffixes = (
            ".md",
            ".rst",
            ".txt",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            ".pdf",
            ".lock",
        )
        return not filename.endswith(skip_suffixes)

    def mark_diffed(self, file_path: str | None, tool_result: dict[str, Any] | None = None) -> None:
        """Record that we have inspected a file's diff."""
        if not file_path:
            return

        with self.lock:
            self.diffed_files.add(file_path)
            if tool_result and isinstance(tool_result, dict):
                previous = tool_result.get("previous_filename")
                if previous:
                    self.diffed_files.add(previous)

    def missing_files(self) -> set[str]:
        with self.lock:
            return set(self.files_to_cover) - set(self.diffed_files)

    def is_known_file(self, file_path: str | None) -> bool:
        if not file_path:
            return False
        with self.lock:
            if not self.files_to_cover:
                return True
            return file_path in self.files_to_cover

    def process_tool_result(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        """Update context based on tool output."""
        if tool_name == "fetch_pr_files" and isinstance(result, dict):
            self.update_files_from_tool(result)
        elif tool_name == "fetch_file_diff" and isinstance(result, dict) and "error" not in result:
            file_path = result.get("filename") or arguments.get("file_path")
            self.mark_diffed(file_path, result)


def detect_platform(pr_url: str) -> Platform:
    """Detect the platform (GitHub or GitLab) from the PR URL."""
    parsed = urlparse(pr_url)
    hostname = parsed.hostname or ""

    # Check for GitLab-specific patterns first (works for both gitlab.com and self-hosted)
    if "/-/merge_requests/" in pr_url or "gitlab" in hostname:
        return "gitlab"
    # Check for GitHub-specific patterns
    elif "/pull/" in pr_url or "github" in hostname:
        return "github"
    else:
        logger.debug(f"Unknown platform for URL {pr_url}, defaulting to GitHub")
        return "github"


def format_review_as_markdown(review_json: dict, owner: str, repo: str, pr_number: int) -> str:
    """Format JSON review output as readable markdown."""
    md = [f"# Code Review for {owner}/{repo}/pull/{pr_number}\n"]

    # Overall correctness verdict
    correctness = review_json.get("overall_correctness", "unknown")
    confidence = review_json.get("overall_confidence_score", 0.0)
    explanation = review_json.get("overall_explanation", "")

    verdict_emoji = "✅" if "correct" in correctness else "⚠️"
    md.append(f"## {verdict_emoji} Overall: {correctness.title()} (confidence: {confidence:.0%})\n")
    md.append(f"{explanation}\n")

    # Findings
    findings = review_json.get("findings", [])
    if not findings:
        md.append("\n## No Issues Found\n")
        md.append("The patch appears to be correct with no blocking issues identified.\n")
        return "\n".join(md)

    # Group findings by priority
    p0_findings = [f for f in findings if f.get("priority") == 0]
    p1_findings = [f for f in findings if f.get("priority") == 1]
    p2_findings = [f for f in findings if f.get("priority") == 2]
    p3_findings = [f for f in findings if f.get("priority") == 3]

    # Format each priority group
    if p0_findings:
        md.append("\n## 🚨 P0 - Critical (Drop Everything)\n")
        for finding in p0_findings:
            md.append(format_finding(finding))

    if p1_findings:
        md.append("\n## 🔴 P1 - Urgent\n")
        for finding in p1_findings:
            md.append(format_finding(finding))

    if p2_findings:
        md.append("\n## 🟡 P2 - Normal\n")
        for finding in p2_findings:
            md.append(format_finding(finding))

    if p3_findings:
        md.append("\n## 🔵 P3 - Low Priority\n")
        for finding in p3_findings:
            md.append(format_finding(finding))

    return "\n".join(md)


def format_finding(finding: dict) -> str:
    """Format a single finding as markdown."""
    title = finding.get("title", "Untitled")
    body = finding.get("body", "")
    confidence = finding.get("confidence_score", 0.0)
    location = finding.get("code_location", {})
    file_path = location.get("absolute_file_path", "unknown")
    line_range = location.get("line_range", {})
    start = line_range.get("start", "?")
    end = line_range.get("end", "?")

    lines = []
    lines.append(f"### {title}")
    lines.append(f"**Location**: `{file_path}:{start}-{end}` (confidence: {confidence:.0%})\n")
    lines.append(body)
    lines.append("")
    return "\n".join(lines)


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """
    Parse PR/MR URL to extract owner, repo, and PR/MR number.

    Examples:
        GitHub: https://github.com/owner/repo/pull/123 -> ('owner', 'repo', 123)
        GitLab: https://gitlab.com/owner/repo/-/merge_requests/123 -> ('owner', 'repo', 123)
    """
    parsed = urlparse(pr_url)
    path_parts = [p for p in parsed.path.split("/") if p]

    # GitHub format: /owner/repo/pull/123
    if len(path_parts) >= 4 and path_parts[2] == "pull":
        owner = path_parts[0]
        repo = path_parts[1]
        pr_number = int(path_parts[3])
        return owner, repo, pr_number

    # GitLab format: /group/subgroup/repo/-/merge_requests/123
    elif "merge_requests" in path_parts:
        mr_index = path_parts.index("merge_requests")
        if mr_index < 2 or mr_index + 1 >= len(path_parts):
            raise ValueError(f"Invalid GitLab MR URL format: {pr_url}. Expected .../-/merge_requests/<number>")
        if path_parts[mr_index - 1] != "-":
            raise ValueError(f"Invalid GitLab MR URL format: {pr_url}. Missing '/-/' segment before merge_requests.")

        repo = path_parts[mr_index - 2]
        owner_parts = path_parts[: mr_index - 2]
        owner = "/".join(owner_parts) if owner_parts else path_parts[0]
        pr_number = int(path_parts[mr_index + 1])
        return owner, repo, pr_number

    else:
        raise ValueError(
            f"Invalid PR/MR URL format: {pr_url}. Expected GitHub pull request or GitLab merge request URL."
        )


def execute_tools_parallel(
    tool_calls: list,
    platform: Platform,
    token: str | None,
    gitlab_url: str | None = None,
    max_workers: int = 15,
    review_context: ReviewContext | None = None,
) -> list[dict[str, Any]]:
    """
    Execute multiple tool calls in parallel using ThreadPoolExecutor.

    Args:
        tool_calls: List of tool call objects from LLM
        platform: Platform (github or gitlab)
        token: API authentication token
        gitlab_url: GitLab instance URL (for self-hosted GitLab)
        max_workers: Maximum number of parallel workers

    Returns:
        List of tool result dictionaries
    """

    def execute_single_tool(tool_call) -> dict[str, Any]:
        """Execute a single tool call and return the result."""
        try:
            arguments = json.loads(tool_call.function.arguments)
            tool_name = tool_call.function.name

            if review_context:
                default_args = {
                    "owner": review_context.owner,
                    "repo": review_context.repo,
                    "pr_number": review_context.pr_number,
                }
                if tool_name in {"search_tests", "fetch_file_content", "list_repo_tree"}:
                    for key, value in default_args.items():
                        arguments.setdefault(key, value)
                elif tool_name in {
                    "fetch_pr_metadata",
                    "fetch_pr_files",
                    "fetch_file_diff",
                    "fetch_pr_commits",
                    "fetch_ci_status",
                }:
                    arguments.setdefault("owner", review_context.owner)
                    arguments.setdefault("repo", review_context.repo)
                    arguments.setdefault("pr_number", review_context.pr_number)

            if review_context and tool_name == "fetch_file_diff":
                file_path = arguments.get("file_path")
                if file_path and not review_context.is_known_file(file_path):
                    missing = review_context.missing_files()
                    hint = ", ".join(sorted(list(missing))[:5]) if missing else "unknown"
                    warning = {
                        "error": (
                            f"File {file_path} is not part of the tracked PR files. "
                            f"Known files include: {hint}. Re-run fetch_pr_files if needed."
                        )
                    }
                    return {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(warning, indent=2)}

            result = execute_tool(tool_call.function.name, arguments, platform, token, gitlab_url)

            if review_context and isinstance(result, dict):
                review_context.process_tool_result(tool_call.function.name, arguments, result)

            return {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, indent=2)}
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function.name}: {str(e)}")
            return {"role": "tool", "tool_call_id": tool_call.id, "content": f"Error executing tool: {str(e)}"}

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_single_tool, tc): tc for tc in tool_calls}

        results = []
        for future in as_completed(futures):
            results.append(future.result())

    return results


def post_review_comment(
    pr_url: str, review_text: str, token: str | None = None, model: str | None = None
) -> dict[str, Any]:
    """
    Post a review comment on a GitHub PR or GitLab MR.

    Args:
        pr_url: URL of the pull request or merge request
        review_text: The review text to post as a comment
        token: API token for authentication
        model: LLM model used for the review (optional, for transparency)

    Returns:
        Dictionary with comment posting result
    """
    # Detect platform and parse URL
    platform = detect_platform(pr_url)
    logger.info(f"Posting comment to {platform} PR/MR: {pr_url}")

    # Extract GitLab URL for self-hosted instances
    gitlab_url = None
    if platform == "gitlab":
        from urllib.parse import urlparse

        parsed = urlparse(pr_url)
        gitlab_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.info(f"GitLab URL: {gitlab_url}")

    try:
        owner, repo, pr_number = parse_pr_url(pr_url)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    # Append model information to review text for transparency
    if model:
        review_text_with_footer = f"{review_text}\n\n---\n\n*Review generated by Hodor using `{model}`*"
    else:
        review_text_with_footer = review_text

    # Call appropriate comment function based on platform
    try:
        if platform == "github":
            result = github_tools.post_pr_comment(
                owner=owner, repo=repo, pr_number=pr_number, comment_body=review_text_with_footer, github_token=token
            )
        elif platform == "gitlab":
            from .tools import gitlab_tools

            result = gitlab_tools.post_mr_comment(
                owner=owner,
                repo=repo,
                mr_number=pr_number,
                comment_body=review_text_with_footer,
                github_token=token,
                gitlab_url=gitlab_url,
            )
        else:
            return {"success": False, "error": f"Unsupported platform: {platform}"}

        return result

    except Exception as e:
        logger.error(f"Error posting comment: {str(e)}")
        return {"success": False, "error": str(e)}


def review_pr(
    pr_url: str,
    max_iterations: int = 20,
    max_workers: int = 15,
    token: str | None = None,
    custom_prompt: str | None = None,
    prompt_file: str | None = None,
    reasoning_effort: str | None = None,
    **litellm_config,
) -> str:
    """
    Review a GitHub or GitLab pull request using AI.

    Args:
        pr_url: URL of the pull request (e.g., https://github.com/owner/repo/pull/123)
        max_iterations: Maximum number of agentic loop iterations (default: 20)
        max_workers: Maximum number of parallel tool calls (default: 15)
        token: API token for authentication
               If not provided, will use GITHUB_TOKEN or GITLAB_TOKEN environment variable
        custom_prompt: Custom inline prompt text (overrides default prompt)
        prompt_file: Path to file containing custom prompt (overrides default prompt)
        reasoning_effort: Reasoning effort level for supported models ('low', 'medium', 'high')
                         Default is 'high' if not specified
        **litellm_config: Additional configuration for litellm.completion()
                         (e.g., model, temperature, max_tokens, etc.)

    Returns:
        Markdown-formatted review text
    """
    # Detect platform and parse URL
    platform = detect_platform(pr_url)
    logger.info(f"Detected platform: {platform}")

    # Extract GitLab URL for self-hosted instances
    gitlab_url = None
    if platform == "gitlab":
        from urllib.parse import urlparse

        parsed = urlparse(pr_url)
        gitlab_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.info(f"GitLab URL: {gitlab_url}")

    try:
        owner, repo, pr_number = parse_pr_url(pr_url)
        logger.info(f"Reviewing PR: {owner}/{repo}/pull/{pr_number}")
    except ValueError as e:
        return f"Error: {str(e)}"

    review_context = ReviewContext(owner, repo, pr_number)

    # Warm up file list so we can track coverage and cache responses.
    initial_files_args = {"owner": owner, "repo": repo, "pr_number": pr_number}
    try:
        initial_files = execute_tool("fetch_pr_files", initial_files_args, platform, token, gitlab_url)
        if isinstance(initial_files, dict):
            review_context.process_tool_result("fetch_pr_files", initial_files_args, initial_files)
    except Exception as exc:
        logger.warning(f"Prefetching PR files failed: {exc}")

    # Set default litellm config and apply user overrides first.
    llm_params = {
        "model": "openai/responses/gpt-5",
        "tools": TOOLS,
        "parallel_tool_calls": True,
    }
    user_overrides = dict(litellm_config)
    llm_params.update(user_overrides)

    # Normalize model names so GPT-5/o3 use the Responses API path
    model = llm_params.get("model", "openai/responses/gpt-5")
    if not model.startswith("openai/responses/"):
        model_lower = model.lower()
        if "gpt-5" in model_lower or "o3-mini" in model_lower or model_lower == "o3":
            logger.info(f"Using Responses API for model: {model}")
            model = f"openai/responses/{model}"
    llm_params["model"] = model

    # Add deterministic temperature only when user hasn't supplied one
    if (
        "temperature" not in user_overrides
        and "gpt-5" not in model.lower()
        and "o3" not in model.lower()
        and "temperature" not in llm_params
    ):
        llm_params["temperature"] = 0.0

    # Enable reasoning for supported models (respect user overrides)
    try:
        if reasoning_effort:
            llm_params["reasoning_effort"] = reasoning_effort
        elif "reasoning_effort" not in llm_params and litellm.supports_reasoning(model):
            logger.info(f"Enabling high reasoning effort for model {model}")
            llm_params["reasoning_effort"] = "high"
    except Exception as e:
        logger.debug(f"Could not check reasoning support: {e}")

    # Load system prompt (priority: custom_prompt > prompt_file > default)
    if custom_prompt:
        logger.info("Using custom inline prompt")
        system_prompt = custom_prompt
    elif prompt_file:
        logger.info(f"Loading prompt from file: {prompt_file}")
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            return f"Error loading prompt file: {e}"
    else:
        # Default system prompt
        system_prompt = f"""You are an expert code reviewer analyzing PR {owner}/{repo}/pull/{pr_number}. Your goal is to find legitimate bugs and issues that would cause problems in production.

# Available Tools
- `fetch_pr_metadata`: Get PR title, description, author
- `fetch_pr_files`: List all changed files
- `fetch_file_diff`: Get unified diff for a file (USE THIS TO READ ACTUAL CODE)
- `fetch_file_content`: Fetch full file contents from the PR head or a provided ref
- `list_repo_tree`: Inspect repository layout (optionally scoped to a directory)
- `fetch_pr_commits`: Get commit history
- `fetch_ci_status`: Check test status
- `search_tests`: Pass `owner`, `repo`, `pr_number`, and `file_path` to locate related tests

# Review Process
1. Call `fetch_pr_metadata` and `fetch_pr_files` in parallel
2. Call `fetch_file_diff` for EVERY code file to see actual changes. Use up to {max_workers} parallel calls.
3. **Read every line of code carefully**. Look for subtle bugs, edge cases, and logic errors.
4. Think critically: What could go wrong? What inputs would break this? What concurrency issues exist?

# What Qualifies as a Bug
A bug must meet ALL these criteria:
   - Meaningfully impacts accuracy, performance, security, or maintainability
   - Discrete and actionable (not general codebase issues)
   - Introduced in this commit (not pre-existing)
   - Author would likely fix if made aware
   - Does not rely on unstated assumptions
   - Not just an intentional change by the author

# Bug Categories - Be Thorough
**Critical (P0/P1)**:
- Race conditions, null/nil derefs, off-by-one errors
- Resource leaks (unclosed files, connections, goroutines, db transactions)
- SQL injection, XSS, command injection, path traversal
- Auth/authz bypasses, session fixation, data exposure
- Incorrect error handling (ignored errors, wrong error types, panic potential)
- Logic errors that cause incorrect behavior

**Important (P2)**:
- N+1 queries, missing database indexes
- Inefficient algorithms (O(n²) where O(n) possible)
- Missing input validation, missing bounds checks
- Deadlock potential, blocking operations in async code
- Memory leaks, unbounded growth
- Incorrect assumptions about data format/structure

**Low (P3)**:
- Code smells impacting maintainability
- Inconsistent error messages
- Magic numbers without explanation
- Overly complex logic that should be simplified

# Priority Levels
- **P0**: Blocks release/operations. Universal issues not dependent on assumptions.
- **P1**: Urgent. Should be addressed in next cycle.
- **P2**: Normal. To be fixed eventually.
- **P3**: Low. Nice to have.

# Critical Instructions
- **DO NOT skip files**. Review every changed code file thoroughly.
- **Look for edge cases**: empty inputs, null/nil values, boundary conditions, concurrent access
- **Think about error paths**: What happens when things fail? Are errors handled properly?
- **Consider security**: Could user input cause problems? Are credentials exposed?
- **Be skeptical**: Don't assume the code is correct. Look for issues.

# Output Format
Return ONLY valid JSON (no markdown fences, no extra prose):

```
{{
  "findings": [
    {{
      "title": "[P0] Brief imperative description (≤80 chars)",
      "body": "One paragraph explaining WHY this is a problem. Reference file:line. Be matter-of-fact, not accusatory. No flattery. Max 3-line code snippets in markdown.",
      "confidence_score": 0.85,
      "priority": 0,
      "code_location": {{
        "absolute_file_path": "path/to/file.ext",
        "line_range": {{"start": 45, "end": 47}}
      }}
    }}
  ],
  "overall_correctness": "patch is correct",
  "overall_explanation": "1-3 sentences justifying the verdict. Ignore non-blocking issues like style/typos.",
  "overall_confidence_score": 0.9
}}
```

# Guidelines
- Output ALL qualifying findings. If nothing qualifies, return empty findings array.
- Comments must be brief (1 paragraph max), clear, and immediately graspable.
- Communicate severity accurately. State specific scenarios/inputs that trigger the bug.
- Line ranges should be 5-10 lines max, pinpointing the problem.
- Assume competent developer. Don't invent problems. If code is good, say so.
- No excessive flattery, no "Great job", no "Thanks for".

Begin."""

    # Initialize conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Review pull request {pr_url}"},
    ]

    # Agent loop
    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration + 1}/{max_iterations}")

        try:
            # Call LLM
            response = completion(messages=messages, **llm_params)
            message = response.choices[0].message

            # Build assistant message
            assistant_msg = {"role": "assistant", "content": []}

            # Handle content
            if hasattr(message, "content") and message.content:
                if isinstance(message.content, str):
                    assistant_msg["content"].append({"type": "text", "text": message.content})
                elif isinstance(message.content, list):
                    assistant_msg["content"].extend(message.content)

            # Handle tool calls
            if hasattr(message, "tool_calls") and message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
                messages.append(assistant_msg)

                logger.info(f"Executing {len(message.tool_calls)} tool calls in parallel...")

                # Execute tools in parallel
                tool_results = execute_tools_parallel(
                    message.tool_calls, platform, token, gitlab_url, max_workers, review_context
                )

                # Add tool results to messages
                messages.extend(tool_results)

            else:
                # No tool calls - final answer
                messages.append(assistant_msg)

                # Extract text response
                response_text = ""
                if isinstance(message.content, str):
                    response_text = message.content
                elif isinstance(message.content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in message.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    response_text = "\n".join(text_parts)

                if not response_text:
                    return "Review completed (no content)"

                if review_context and review_context.files_to_cover:
                    missing = review_context.missing_files()
                    if missing:
                        sample_missing = ", ".join(sorted(list(missing))[:5])
                        if len(missing) > 5:
                            sample_missing += ", ..."
                        reminder_text = (
                            "You must inspect the diff for every changed code file before finishing. "
                            f"Still missing {len(missing)} file(s): {sample_missing}. "
                            "Call `fetch_file_diff` for each remaining file."
                        )
                        messages.append({"role": "user", "content": reminder_text})
                        continue

                # Try to parse as JSON and format as markdown
                try:
                    # Remove markdown fences if present
                    clean_text = response_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.startswith("```"):
                        clean_text = clean_text[3:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]

                    review_json = json.loads(clean_text.strip())
                    return format_review_as_markdown(review_json, owner, repo, pr_number)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, returning raw text")
                    return response_text

        except Exception as e:
            logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
            return f"Error during review: {str(e)}"

    return f"Maximum iterations ({max_iterations}) reached. Review incomplete."
