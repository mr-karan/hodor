#!/usr/bin/env node

import { Command } from "commander";
import chalk from "chalk";
import "dotenv/config";

import { detectPlatform, postReviewComment, reviewPr } from "./agent.js";
import type { AgentProgressEvent } from "./agent.js";
import { renderMarkdown } from "./render.js";
import { pushMetrics } from "./metrics.js";
import { setLogLevel } from "./utils/logger.js";

const program = new Command();

program
  .name("hodor")
  .description(
    "AI-powered code review agent for GitHub PRs, GitLab MRs, Gitea/Forgejo PRs, and local diffs.\n\n" +
      "Hodor uses an AI agent that clones the repository, checks out the PR branch,\n" +
      "and analyzes the code using tools (gh, git, glab) for metadata fetching and comment posting.\n\n" +
      "For local reviews, use --local with --diff-against to review changes in your current git repository.",
  )
  .version("0.6.0")
  .argument("[pr-url]", "URL of the GitHub PR, GitLab MR, or Gitea/Forgejo PR to review (optional with --local)")
  .option(
    "--model <model>",
    "LLM model to use as provider/model-id (e.g., anthropic/claude-sonnet-4-5-20250929, openrouter/moonshotai/kimi-k2.6)",
    "anthropic/claude-sonnet-4-5-20250929",
  )
  .option(
    "--reasoning-effort <level>",
    "Reasoning effort level: low, medium, high, xhigh",
  )
  .option("-v, --verbose", "Enable verbose logging", false)
  .option(
    "--post",
    "Post the review directly to the PR/MR as a comment",
    false,
  )
  .option("--prompt <text>", "Custom inline prompt text")
  .option(
    "--prompt-file <path>",
    "Path to file containing custom prompt instructions",
  )
  .option(
    "--workspace <dir>",
    "Workspace directory (creates temp dir if not specified)",
  )
  .option(
    "--ultrathink",
    "Enable maximum reasoning effort with extended thinking budget",
    false,
  )
  .option(
    "--bedrock-tags <json>",
    "JSON object of cost allocation tags for Bedrock requests (e.g., '{\"team\":\"platform\"}')",
  )
  .option(
    "--prometheus-push <url>",
    "Push review metrics to a Prometheus Pushgateway URL",
  )
  .option(
    "--local",
    "Review local changes in the current directory (no PR URL required)",
    false,
  )
  .option(
    "--diff-against <ref>",
    "Git ref to diff against in local mode (e.g., origin/main, HEAD~1)",
    "origin/main",
  )
  .action(async (prUrl: string | undefined, cmdOpts: Record<string, unknown>) => {
    const verbose = cmdOpts.verbose as boolean;
    const post = cmdOpts.post as boolean;
    const model = cmdOpts.model as string;
    let reasoningEffort = cmdOpts.reasoningEffort as string | undefined;
    const prompt = cmdOpts.prompt as string | undefined;
    const promptFile = cmdOpts.promptFile as string | undefined;
    const workspace = cmdOpts.workspace as string | undefined;
    const ultrathink = cmdOpts.ultrathink as boolean;
    const bedrockTagsRaw = cmdOpts.bedrockTags as string | undefined;
    const prometheusPush = cmdOpts.prometheusPush as string | undefined;
    const localMode = cmdOpts.local as boolean;
    const diffAgainst = cmdOpts.diffAgainst as string;

    if (!localMode && !prUrl) {
      console.error(chalk.red("Error: pr-url is required unless --local is specified"));
      process.exit(1);
    }
    if (localMode && post) {
      console.error(chalk.red("Error: --post is not supported in --local mode (no remote to post to)"));
      process.exit(1);
    }

    // Auto-detect CI environment
    const isCI = !!(process.env.CI || process.env.GITLAB_CI || process.env.GITHUB_ACTIONS || process.env.GITEA_ACTIONS || process.env.FORGEJO_ACTIONS);

    if (verbose) setLogLevel("debug");
    else if (isCI) setLogLevel("info");

    // Handle ultrathink
    if (ultrathink) {
      reasoningEffort = "high";
    }

    // Parse Bedrock cost allocation tags
    let bedrockTags: Record<string, string> | null = null;
    if (bedrockTagsRaw) {
      try {
        bedrockTags = JSON.parse(bedrockTagsRaw) as Record<string, string>;
      } catch {
        console.error(chalk.red("Error: --bedrock-tags must be valid JSON"));
        process.exit(1);
      }
    }

    const log = console.log;
    const logStream = process.stdout;

    const toolIcons: Record<string, string> = {
      bash: "$",
      read: "cat",
      grep: "grep",
      find: "find",
      ls: "ls",
    };

    /** Write a line to the log stream */
    function streamLog(msg: string): void {
      logStream.write(`${msg}\n`);
    }

    /** Write inline text (no newline) for streaming deltas */
    function streamWrite(text: string): void {
      process.stderr.write(text);
    }

    function handleEvent(event: AgentProgressEvent): void {
      switch (event.type) {
        case "agent_start":
          streamLog(chalk.dim("▶ Agent started"));
          break;
        case "turn_start":
          streamLog(chalk.dim(`\n── Turn ${event.turnIndex ?? "?"} ──`));
          break;
        case "tool_start": {
          const icon = toolIcons[event.toolName ?? ""] ?? event.toolName;
          const preview = event.toolArgs ? ` ${event.toolArgs}` : "";
          const maxLen = 160;
          const truncated = preview.length > maxLen ? preview.slice(0, maxLen) + "…" : preview;
          streamLog(chalk.green(`  ${icon}${truncated}`));
          break;
        }
        case "tool_end": {
          if (event.isError) {
            streamLog(chalk.red(`  ✗ error`));
          }
          if (event.result) {
            const lines = event.result.split("\n");
            const maxLines = verbose ? 15 : 6;
            const maxChars = verbose ? 400 : 200;
            let chars = 0;
            for (let i = 0; i < Math.min(lines.length, maxLines); i++) {
              const line = lines[i];
              if (chars + line.length > maxChars) {
                streamLog(chalk.dim(`    …(${lines.length - i} more lines)`));
                break;
              }
              streamLog(chalk.dim(`    ${line}`));
              chars += line.length;
            }
          }
          break;
        }
        case "text_delta":
          if (verbose && event.delta) {
            streamWrite(event.delta);
          }
          break;
        case "thinking_delta":
          // Only show reasoning in verbose mode
          if (verbose && event.delta) {
            streamWrite(chalk.dim(event.delta));
          }
          break;
        case "agent_end":
          streamLog(chalk.dim("\n▶ Extracting review..."));
          break;
      }
    }

    try {
      // Detect platform and warn about missing tokens
      let platform: string = "local";
      if (!localMode && prUrl) {
        platform = detectPlatform(prUrl);
        const githubToken = process.env.GITHUB_TOKEN;
        const gitlabToken =
          process.env.GITLAB_TOKEN ??
          process.env.GITLAB_PRIVATE_TOKEN ??
          process.env.CI_JOB_TOKEN;

        if (platform === "github" && !githubToken) {
          console.error(chalk.yellow("Warning: GITHUB_TOKEN not set. You may encounter rate limits."));
          console.error(chalk.dim("  Set GITHUB_TOKEN or run: gh auth login\n"));
        } else if (platform === "gitlab" && !gitlabToken) {
          console.error(chalk.yellow("Warning: No GitLab token detected. Set GITLAB_TOKEN (api scope)."));
          console.error(chalk.dim("  Export GITLAB_TOKEN and optionally GITLAB_HOST.\n"));
        } else if (platform === "gitea") {
          const giteaToken = process.env.GITEA_TOKEN ?? process.env.FORGEJO_TOKEN;
          if (!giteaToken) {
            console.error(chalk.yellow("Warning: No Gitea/Forgejo token detected. Set GITEA_TOKEN for authentication."));
            console.error(chalk.dim("  Export GITEA_TOKEN (or FORGEJO_TOKEN) for API access.\n"));
          }
        }
      }

      log(`\n${chalk.bold.cyan("Hodor - AI Code Review Agent")}`);
      if (localMode) {
        log(chalk.dim(`Mode: Local diff review`));
        log(chalk.dim(`Diff against: ${diffAgainst}`));
        log(chalk.dim(`Workspace: ${workspace ?? process.cwd()}`));
      } else {
        log(chalk.dim(`Platform: ${platform.toUpperCase()}`));
        log(chalk.dim(`PR URL: ${prUrl}`));
      }
      log(chalk.dim(`Model: ${model}`));
      if (reasoningEffort) {
        log(chalk.dim(`Reasoning Effort: ${reasoningEffort}`));
      }
      log();

      streamLog(chalk.dim("▶ Setting up workspace..."));
      const { review, metricsFooter, headSha, metrics } = await reviewPr({
        prUrl: localMode ? undefined : prUrl,
        model,
        reasoningEffort,
        customPrompt: prompt,
        promptFile,
        cleanup: !workspace,
        workspaceDir: workspace,
        includeMetricsFooter: post && !localMode,
        onEvent: handleEvent,
        bedrockTags,
        localMode,
        diffAgainst,
      });
      const reviewText = renderMarkdown(review);

      streamLog(chalk.green("✔ Review complete!"));

      if (post && prUrl) {
        log(chalk.cyan("\nPosting review to PR/MR..."));

        const result = await postReviewComment({
          prUrl,
          reviewText,
          model,
          metricsFooter,
          headSha,
        });

        if (result.success) {
          log(chalk.bold.green("Review posted successfully!"));
          log(chalk.dim(`  ${platform === "gitlab" ? "MR" : "PR"}: ${prUrl}`));
        } else {
          log(chalk.bold.red(`Failed to post review: ${result.error}`));
          log(chalk.yellow("\nReview output:\n"));
          console.log(reviewText);
        }
      } else {
        log(chalk.bold.green("Review Complete\n"));
        console.log(reviewText);
        if (!localMode) {
          log(chalk.dim("\nTip: Use --post to automatically post this review to the PR/MR"));
        }
      }

      // Push metrics to Prometheus Pushgateway (best-effort, never fails the run)
      if (prometheusPush) {
        await pushMetrics({
          pushgatewayUrl: prometheusPush,
          metrics,
          labels: {
            platform,
            model,
            verdict: review.overall_correctness === "patch is correct" ? "correct" : "incorrect",
          },
        });
      }
    } catch (err) {
      streamLog(chalk.red("✗ Review failed"));
      console.error(
        chalk.bold.red(`\nError: ${err instanceof Error ? err.message : err}`),
      );
      if (verbose && err instanceof Error && err.stack) {
        console.error(chalk.dim(err.stack));
      }
      process.exit(1);
    }
  });

program.parse();
