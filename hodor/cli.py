"""Command-line interface for PR Review Agent."""

import logging
import os
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent import review_pr, post_review_comment, detect_platform, configure_litellm_logging

console = Console()


def parse_llm_args(ctx, param, value):
    """Parse --llm-* arguments into a dictionary."""
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
@click.option("--max-iterations", default=20, type=int, help="Maximum number of agentic loop iterations")
@click.option("--max-workers", default=15, type=int, help="Maximum number of parallel tool calls")
@click.option("--token", default=None, help="GitHub/GitLab API token (or set GITHUB_TOKEN/GITLAB_TOKEN env var)")
@click.option("--model", default="gpt-5", help="LLM model to use")
@click.option("--temperature", default=0.0, type=float, help="LLM temperature (0.0-2.0)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--llm", multiple=True, help="Additional LLM parameters (format: key=value)")
@click.option("--post-comment", is_flag=True, help="Post the review as a comment on the PR/MR (useful for CI/CD)")
@click.option("--prompt", default=None, help="Custom inline prompt text (overrides default)")
@click.option("--prompt-file", default=None, type=click.Path(exists=True), help="Path to file containing custom prompt")
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default=None,
    help="Reasoning effort level (default: high)",
)
def main(
    pr_url,
    max_iterations,
    max_workers,
    token,
    model,
    temperature,
    verbose,
    llm,
    post_comment,
    prompt,
    prompt_file,
    reasoning_effort,
):
    """
    Review a GitHub pull request or GitLab merge request using AI.

    Examples:

        \b
        # Review GitHub PR
        hodor https://github.com/owner/repo/pull/123

        \b
        # Review GitLab MR
        hodor https://gitlab.com/owner/project/-/merge_requests/456

        \b
        # With custom model
        hodor https://github.com/owner/repo/pull/123 --model claude-sonnet-4-5

        \b
        # With private repo token (GitHub)
        hodor https://github.com/owner/repo/pull/123 --token ghp_xxxxx

        \b
        # With private repo token (GitLab)
        hodor https://gitlab.com/owner/project/-/merge_requests/456 --token glpat-xxxxx

        \b
        # Verbose mode
        hodor https://github.com/owner/repo/pull/123 -v
    """
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    configure_litellm_logging(verbose)

    # Auto-detect token from environment variables if not provided
    if token is None:
        token = os.getenv("GITLAB_TOKEN") or os.getenv("GITHUB_TOKEN")

    # Warn if using wrong token for platform
    if token:
        platform = detect_platform(pr_url)
        gitlab_token = os.getenv("GITLAB_TOKEN")
        github_token = os.getenv("GITHUB_TOKEN")

        if platform == "gitlab" and github_token and not gitlab_token:
            console.print(
                "[yellow]⚠️  Warning: Detected GitLab URL but only GITHUB_TOKEN is set. You may need to set GITLAB_TOKEN instead.[/yellow]"
            )
        elif platform == "github" and gitlab_token and not github_token:
            console.print(
                "[yellow]⚠️  Warning: Detected GitHub URL but only GITLAB_TOKEN is set. You may need to set GITHUB_TOKEN instead.[/yellow]"
            )

    # Parse additional LLM arguments
    llm_config = parse_llm_args(None, None, llm)
    llm_config["model"] = model
    llm_config["temperature"] = temperature

    console.print("\n[bold cyan]🚪 Hodor[/bold cyan]")
    console.print(f"[dim]Reviewing: {pr_url}[/dim]")
    console.print(f"[dim]Model: {model}[/dim]\n")

    try:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True
        ) as progress:
            _task = progress.add_task("Analyzing PR...", total=None)

            # Run the review
            result = review_pr(
                pr_url=pr_url,
                max_iterations=max_iterations,
                max_workers=max_workers,
                token=token,
                custom_prompt=prompt,
                prompt_file=prompt_file,
                reasoning_effort=reasoning_effort,
                **llm_config,
            )

            progress.stop()

        # Display result as markdown
        console.print("\n[bold green]✅ Review Complete[/bold green]\n")
        md = Markdown(result)
        console.print(md)

        # Post comment if requested
        if post_comment:
            console.print("\n[cyan]Posting review as comment...[/cyan]")
            try:
                comment_result = post_review_comment(pr_url=pr_url, review_text=result, token=token, model=model)

                if comment_result.get("success"):
                    console.print("[bold green]✅ Comment posted successfully![/bold green]")
                    console.print(f"[dim]Comment URL: {comment_result.get('comment_url')}[/dim]")
                else:
                    console.print(f"[bold red]❌ Failed to post comment:[/bold red] {comment_result.get('error')}")
            except Exception as e:
                console.print(f"[bold red]❌ Error posting comment:[/bold red] {str(e)}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Review cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
