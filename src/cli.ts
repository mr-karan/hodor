#!/usr/bin/env node

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { Command } from "commander";
import chalk from "chalk";
import "dotenv/config";

import { detectPlatform, postReviewComment, reviewPr } from "./agent.js";
import type { AgentProgressEvent } from "./agent.js";
import { summarizeToolOutput } from "./logging.js";
import type { ConsoleLogMode } from "./logging.js";
import { renderMarkdown } from "./render.js";
import { setLogLevel } from "./utils/logger.js";

const program = new Command();

program
  .name("hodor")
  .description(
    "AI-powered code review agent for GitHub PRs and GitLab MRs.\n\n" +
      "Hodor uses an AI agent that clones the repository, checks out the PR branch,\n" +
      "and analyzes the code using tools (gh, git, glab) for metadata fetching and comment posting.",
  )
  .version("0.3.4")
  .argument("<pr-url>", "URL of the GitHub PR or GitLab MR to review")
  .option(
    "--model <model>",
    "LLM model to use (e.g., anthropic/claude-sonnet-4-5-20250929, openai/gpt-5, openai-codex/gpt-5.4)",
    "anthropic/claude-sonnet-4-5-20250929",
  )
  .option(
    "--reasoning-effort <level>",
    "Reasoning effort level: low, medium, high, xhigh",
  )
  .option(
    "--progress",
    "Show progress logging with colored assistant updates and summarized tool output",
    false,
  )
  .option(
    "-v, --verbose",
    "Enable verbose logging with completed thinking blobs and expanded tool previews",
    false,
  )
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
    "--session-log-file <path>",
    "Write a detailed session log with full tool outputs to a file",
  )
  .action(async (prUrl: string, cmdOpts: Record<string, unknown>) => {
    const progress = cmdOpts.progress as boolean;
    const verbose = cmdOpts.verbose as boolean;
    const post = cmdOpts.post as boolean;
    const model = cmdOpts.model as string;
    let reasoningEffort = cmdOpts.reasoningEffort as string | undefined;
    const prompt = cmdOpts.prompt as string | undefined;
    const promptFile = cmdOpts.promptFile as string | undefined;
    const workspace = cmdOpts.workspace as string | undefined;
    const ultrathink = cmdOpts.ultrathink as boolean;
    const sessionLogFile = cmdOpts.sessionLogFile as string | undefined;
    const consoleLogMode: ConsoleLogMode = verbose ? "verbose" : progress ? "progress" : "quiet";
    let currentTurn = 0;

    // Auto-detect CI environment
    const isCI = !!(process.env.CI || process.env.GITLAB_CI || process.env.GITHUB_ACTIONS);

    if (verbose) setLogLevel("debug");
    else if (isCI) setLogLevel("info");

    // Handle ultrathink
    if (ultrathink) {
      reasoningEffort = "xhigh";
    }

    const log = console.log;
    const logStream = process.stdout;
    const assistantLineColor = chalk.hex("#2FC2FF");
    const thinkingLineColor = chalk.hex("#7F9CB5");

    function streamLog(msg: string): void {
      logStream.write(`${msg}\n`);
    }

    function streamBlock(prefix: string, text: string, colorize?: (value: string) => string): void {
      const render = colorize ?? ((value: string) => value);
      const normalized = text
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n")
        .trim();
      if (!normalized) return;
      for (const line of normalized.split("\n")) {
        streamLog(render(`${prefix}${line}`));
      }
    }

    function initializeSessionLog(): void {
      if (!sessionLogFile) return;
      mkdirSync(dirname(sessionLogFile), { recursive: true });
      writeFileSync(sessionLogFile, "", "utf8");
    }

    function writeSessionLog(text: string): void {
      if (!sessionLogFile) return;
      appendFileSync(sessionLogFile, text, "utf8");
    }

    function writeSessionLine(label: string, text?: string): void {
      const suffix = text ? ` ${text}` : "";
      writeSessionLog(`[${new Date().toISOString()}] ${label}${suffix}\n`);
    }

    function writeSessionBlock(label: string, text: string): void {
      const normalized = text
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n")
        .trimEnd();
      if (!normalized) return;
      writeSessionLine(`${label} begin`);
      writeSessionLog(normalized.endsWith("\n") ? normalized : `${normalized}\n`);
      writeSessionLine(`${label} end`);
    }

    function loadToolOutputForSessionLog(event: AgentProgressEvent): string {
      const fallback = event.result?.trimEnd() ?? "";
      if (!event.fullOutputPath || !existsSync(event.fullOutputPath)) {
        return fallback;
      }

      try {
        const fullOutput = readFileSync(event.fullOutputPath, "utf8").trimEnd();
        return fullOutput || fallback;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return [fallback, `[Unable to read ${event.fullOutputPath}: ${message}]`]
          .filter((part) => part && part.length > 0)
          .join("\n")
          .trimEnd();
      }
    }

    function handleEvent(event: AgentProgressEvent): void {
      switch (event.type) {
        case "agent_start":
          streamLog(chalk.dim("▶ Agent started"));
          writeSessionLine("[agent]", "started");
          break;
        case "turn_start":
          currentTurn = event.turnIndex ?? currentTurn;
          streamLog(chalk.dim(`\n── Turn ${currentTurn || "?"} ──`));
          writeSessionLine(`[turn ${currentTurn || "?"}]`, "start");
          break;
        case "tool_start": {
          const toolLabel = event.toolName ?? "tool";
          const preview = event.toolArgs ? ` ${event.toolArgs}` : "";
          const maxLen = consoleLogMode === "verbose" ? 220 : 160;
          const truncated = preview.length > maxLen ? preview.slice(0, maxLen) + "…" : preview;
          streamLog(`  tool> ${toolLabel}${truncated}`);
          writeSessionLine(`[turn ${currentTurn || "?"}] [tool:start]`, `${toolLabel}${preview}`);
          break;
        }
        case "tool_end": {
          const toolLabel = event.toolName ?? "tool";
          if (event.isError) {
            streamLog(chalk.red(`  tool! ${toolLabel} failed`));
          }
          if (consoleLogMode !== "quiet" && event.result) {
            const previewMode = consoleLogMode === "verbose" ? "verbose" : "progress";
            for (const line of summarizeToolOutput(event.result, previewMode)) {
              streamLog(chalk.dim(`    out> ${line}`));
            }
          }
          writeSessionLine(
            `[turn ${currentTurn || "?"}] [tool:end]`,
            `${toolLabel} ${event.isError ? "error" : "ok"}${event.truncated ? " truncated=true" : ""}${event.fullOutputPath ? ` full_output_path=${event.fullOutputPath}` : ""}`,
          );
          const fullOutput = loadToolOutputForSessionLog(event);
          if (fullOutput) {
            writeSessionBlock(`[turn ${currentTurn || "?"}] [tool:output] ${toolLabel}`, fullOutput);
          }
          break;
        }
        case "assistant_text":
          if (event.message) {
            streamBlock(
              "  agent> ",
              event.message,
              consoleLogMode === "quiet" ? undefined : assistantLineColor,
            );
            writeSessionBlock(`[turn ${currentTurn || "?"}] [assistant:text]`, event.message);
          }
          break;
        case "assistant_thinking":
          if (event.message) {
            if (consoleLogMode !== "quiet") {
              streamBlock("  think> ", event.message, thinkingLineColor);
            }
            writeSessionBlock(`[turn ${currentTurn || "?"}] [assistant:thinking]`, event.message);
          }
          break;
        case "turn_end":
          if (event.message) {
            const turnLabel = (event.turnIndex ?? currentTurn) || "?";
            streamLog(chalk.dim(`  turn> ${event.message}`));
            writeSessionLine(`[turn ${turnLabel}]`, `end ${event.message}`);
          }
          break;
        case "agent_end":
          streamLog(chalk.dim("\n▶ Extracting review..."));
          writeSessionLine("[agent]", "extracting_review");
          break;
      }
    }

    try {
      // Validate URL and detect platform (inside try so errors are caught)
      const platform = detectPlatform(prUrl);
      initializeSessionLog();
      const githubToken = process.env.GITHUB_TOKEN;
      const gitlabToken =
        process.env.GITLAB_TOKEN ??
        process.env.GITLAB_PRIVATE_TOKEN ??
        process.env.CI_JOB_TOKEN;

      if (platform === "github" && !githubToken) {
        console.error(
          chalk.yellow(
            "Warning: GITHUB_TOKEN not set. You may encounter rate limits or authentication issues.",
          ),
        );
        console.error(
          chalk.dim("  Set GITHUB_TOKEN environment variable or run: gh auth login\n"),
        );
      } else if (platform === "gitlab" && !gitlabToken) {
        console.error(
          chalk.yellow(
            "Warning: No GitLab token detected. Set GITLAB_TOKEN (api scope) for authentication.",
          ),
        );
        console.error(
          chalk.dim(
            "  Export GITLAB_TOKEN and optionally GITLAB_HOST for self-hosted instances.\n",
          ),
        );
      }

      log(
        `\n${chalk.bold.cyan("Hodor - AI Code Review Agent")}`,
      );
      log(chalk.dim(`Platform: ${platform.toUpperCase()}`));
      log(chalk.dim(`PR URL: ${prUrl}`));
      log(chalk.dim(`Model: ${model}`));
      if (reasoningEffort) {
        log(chalk.dim(`Reasoning Effort: ${reasoningEffort}`));
      }
      log();
      writeSessionLine("[session]", `platform=${platform} pr=${prUrl} model=${model}${reasoningEffort ? ` reasoning=${reasoningEffort}` : ""} console_mode=${consoleLogMode}`);

      streamLog(chalk.dim("▶ Setting up workspace..."));
      const { review, metricsFooter } = await reviewPr({
        prUrl,
        model,
        reasoningEffort,
        customPrompt: prompt,
        promptFile,
        cleanup: !workspace,
        workspaceDir: workspace,
        includeMetricsFooter: post,
        onEvent: handleEvent,
      });
      const reviewText = renderMarkdown(review);
      writeSessionBlock("[review]", reviewText);

      streamLog(chalk.green("✔ Review complete!"));

      if (post) {
        log(chalk.cyan("\nPosting review to PR/MR..."));

        const result = await postReviewComment({
          prUrl,
          reviewText,
          model,
          metricsFooter,
        });

        if (result.success) {
          writeSessionLine("[post]", `success ${prUrl}`);
          log(chalk.bold.green("Review posted successfully!"));
          log(chalk.dim(`  ${platform === "github" ? "PR" : "MR"}: ${prUrl}`));
        } else {
          writeSessionLine("[post]", `failure ${result.error ?? "unknown error"}`);
          log(
            chalk.bold.red(`Failed to post review: ${result.error}`),
          );
          log(chalk.yellow("\nReview output:\n"));
          console.log(reviewText);
        }
      } else {
        log(chalk.bold.green("Review Complete\n"));
        console.log(reviewText);
        log(
          chalk.dim(
            "\nTip: Use --post to automatically post this review to the PR/MR",
          ),
        );
      }
    } catch (err) {
      streamLog(chalk.red("✗ Review failed"));
      writeSessionLine("[error]", err instanceof Error ? err.message : String(err));
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
