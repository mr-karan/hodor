"""Command-line interface for Hodor PR Review Agent."""

import logging
import os
import subprocess
import sys
from pathlib import Path

from . import _tty as _terminal_safety  # noqa: F401

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent import detect_platform, parse_pr_url, post_review_comment, review_pr

console = Console()


def parse_llm_args(ctx, param, value):
    """Parse --llm arguments into a dictionary.

    Supports formats like:
    - --llm key=value
    - --llm flag  (sets to True)
    """
    if not value:
        return {}

    config = {}
    for arg in value:
        if "=" in arg:
            key, val = arg.split("=", 1)
            # Try to convert to appropriate type
            if val.lower() == "true":
                config[key] = True
            elif val.lower() == "false":
                config[key] = False
            elif val.replace(".", "", 1).replace("-", "", 1).isdigit():
                config[key] = float(val) if "." in val else int(val)
            else:
                config[key] = val
        else:
            config[arg] = True

    return config


@click.command()
@click.argument("pr_url")
@click.option(
    "--model",
    default="anthropic/claude-sonnet-4-5-20250929",
    help="LLM model to use. Recommended: anthropic/claude-sonnet-4-20250514, anthropic/claude-sonnet-4-5-20250929 (default), openai/gpt-5-2025-08-07, gemini/gemini-2.5-pro, deepseek/deepseek-chat, moonshot/kimi-k2-0711-preview. Supports any LiteLLM model (https://docs.litellm.ai/docs/providers).",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="LLM temperature (0.0-2.0). Auto-selected if not specified based on model capabilities.",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default=None,
    help="Reasoning effort level for models that support extended thinking (e.g., Claude, GPT-5)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging (shows OpenHands agent activity)",
)
@click.option(
    "--llm",
    multiple=True,
    callback=parse_llm_args,
    help="Additional LLM parameters in key=value format (can be specified multiple times)",
)
@click.option(
    "--post/--no-post",
    default=False,
    help="Post the review directly to the PR/MR as a comment (useful for CI/CD). Default: no-post (print to stdout)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output structured JSON format instead of markdown (useful for CI/CD automation and parsing)",
)
@click.option(
    "--prompt",
    default=None,
    help="Custom inline prompt text (overrides default and any prompt file)",
)
@click.option(
    "--prompt-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to file containing custom prompt instructions",
)
@click.option(
    "--workspace",
    default=None,
    type=click.Path(),
    help="Workspace directory to use (creates temp dir if not specified). Reuses workspace if same repo.",
)
@click.option(
    "--max-iterations",
    default=500,
    type=int,
    help="Maximum number of agent iterations/steps (default: 500, use -1 for unlimited)",
)
@click.option(
    "--ultrathink",
    is_flag=True,
    help="Enable maximum reasoning effort with extended thinking budget (shortcut for --reasoning-effort high)",
)
def main(
    pr_url: str,
    model: str,
    temperature: float | None,
    reasoning_effort: str | None,
    verbose: bool,
    llm: dict,
    post: bool,
    output_json: bool,
    prompt: str | None,
    prompt_file: str | None,
    workspace: str | None,
    max_iterations: int,
    ultrathink: bool,
):
    """
    Review a GitHub pull request or GitLab merge request using AI.

    Hodor uses OpenHands SDK to run an AI agent that clones the repository,
    checks out the PR branch, and analyzes the code using bash tools (gh, git,
    glab) for metadata fetching and comment posting.

    \b
    Examples:
        # Review GitHub PR (output to console)
        hodor https://github.com/owner/repo/pull/123

        # Review and post directly to PR
        hodor https://github.com/owner/repo/pull/123 --post

        # Review GitLab MR (self-hosted)
        export GITLAB_HOST=gitlab.example.com
        hodor https://gitlab.example.com/owner/project/-/merge_requests/456

        # Custom model with reasoning
        hodor URL --model anthropic/claude-opus-4 --reasoning-effort high

        # Custom prompt
        hodor URL --prompt-file prompts/security-focused.txt

        # Additional LLM params
        hodor URL --llm max_tokens=8000 --llm stop="```"

    \b
    Environment Variables:
        LLM_API_KEY or ANTHROPIC_API_KEY or OPENAI_API_KEY - LLM API key (required)
        LLM_BASE_URL - Custom LLM endpoint (optional)
        GITHUB_TOKEN - GitHub API token (for gh CLI authentication)
        GITLAB_TOKEN / GITLAB_PRIVATE_TOKEN / CI_JOB_TOKEN - GitLab API tokens for glab CLI
        GITLAB_HOST - GitLab host for self-hosted instances (default: gitlab.com)

    \b
    Authentication:
        - GitHub: gh auth login  or set GITHUB_TOKEN
        - GitLab: provide a token with api scope via GITLAB_TOKEN (or CI_JOB_TOKEN in CI)
    """
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Check platform and token availability
    platform = detect_platform(pr_url)
    github_token = os.getenv("GITHUB_TOKEN")
    gitlab_token = os.getenv("GITLAB_TOKEN") or os.getenv("GITLAB_PRIVATE_TOKEN") or os.getenv("CI_JOB_TOKEN")

    if platform == "github" and not github_token:
        console.print(
            "[yellow]⚠️  Warning: GITHUB_TOKEN not set. You may encounter rate limits or authentication issues.[/yellow]"
        )
        console.print("[dim]   Set GITHUB_TOKEN environment variable or run: gh auth login[/dim]\n")
    elif platform == "gitlab" and not gitlab_token:
        console.print(
            "[yellow]⚠️  Warning: No GitLab token detected. Set GITLAB_TOKEN (api scope) or rely on CI_JOB_TOKEN for "
            "CI environments.[/yellow]"
        )
        console.print("[dim]   Export GITLAB_TOKEN and optionally GITLAB_HOST for self-hosted instances.[/dim]\n")

    # Parse prompt file path
    prompt_file_path = Path(prompt_file) if prompt_file else None

    # Handle ultrathink flag
    if ultrathink:
        reasoning_effort = "high"
        # Ensure extended_thinking_budget is high if not already set
        if "extended_thinking_budget" not in llm:
            llm = {**llm, "extended_thinking_budget": 500000}

    console.print("\n[bold cyan]🚪 Hodor - AI Code Review Agent[/bold cyan]")
    console.print(f"[dim]Platform: {platform.upper()}[/dim]")
    console.print(f"[dim]PR URL: {pr_url}[/dim]")
    console.print(f"[dim]Model: {model}[/dim]")
    if reasoning_effort:
        console.print(f"[dim]Reasoning Effort: {reasoning_effort}[/dim]")
    if max_iterations == -1:
        console.print(f"[dim]Max Iterations: Unlimited[/dim]")
    else:
        console.print(f"[dim]Max Iterations: {max_iterations}[/dim]")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Setting up workspace and running review...", total=None)

            # Run the review
            workspace_path = Path(workspace) if workspace else None
            review_output = review_pr(
                pr_url=pr_url,
                model=model,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                custom_prompt=prompt,
                prompt_file=prompt_file_path,
                user_llm_params=llm,
                verbose=verbose,
                cleanup=workspace is None,  # Only cleanup if using temp dir
                workspace_dir=workspace_path,
                output_format="json" if output_json else "markdown",
                max_iterations=max_iterations,
            )

            progress.update(task, description="Review complete!")
            progress.stop()

        # Display result
        if post:
            # Post to PR/MR (always as markdown, never raw JSON)
            console.print("\n[cyan]📤 Posting review to PR/MR...[/cyan]\n")
            try:
                # If output is JSON, we need to format it as markdown for posting
                if output_json:
                    from .review_parser import parse_review_output, format_review_markdown

                    parsed = parse_review_output(review_output)
                    review_text = format_review_markdown(parsed)
                else:
                    review_text = review_output

                result = post_review_comment(
                    pr_url=pr_url,
                    review_text=review_text,
                    model=model,
                )

                if result.get("success"):
                    console.print("[bold green]✅ Review posted successfully![/bold green]")
                    if platform == "github":
                        console.print(f"[dim]   PR: {pr_url}[/dim]")
                    else:
                        console.print(f"[dim]   MR: {pr_url}[/dim]")
                else:
                    console.print(f"[bold red]❌ Failed to post review:[/bold red] {result.get('error')}")
                    console.print("\n[yellow]Review output:[/yellow]\n")
                    if output_json:
                        console.print(review_output)
                    else:
                        console.print(Markdown(review_output))

            except Exception as e:
                console.print(f"[bold red]❌ Error posting review:[/bold red] {str(e)}")
                console.print("\n[yellow]Review output:[/yellow]\n")
                if output_json:
                    console.print(review_output)
                else:
                    console.print(Markdown(review_output))

        else:
            # Print to console
            console.print("[bold green]✅ Review Complete[/bold green]\n")
            if output_json:
                # Output raw JSON (no markdown rendering)
                console.print(review_output)
            else:
                console.print(Markdown(review_output))
                console.print("\n[dim]💡 Tip: Use --post to automatically post this review to the PR/MR[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Review cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
