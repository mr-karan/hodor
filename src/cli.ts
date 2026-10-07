#!/usr/bin/env node

import { writeFileSync } from "node:fs";
import { Command } from "commander";
import chalk from "chalk";
import "dotenv/config";
import packageJson from "../package.json" with { type: "json" };

import { detectPlatform, parsePrUrl, postReviewComment, postReviewStructured, reviewPr } from "./agent.js";
import {
  createTraceRenderer,
  formatDiagnostics,
  formatReviewSummary,
  formatStartLine,
  type Delivery,
  type ReviewTarget,
} from "./cli-output.js";
import { formatCodeQualityReport } from "./codequality.js";
import { fetchGitlabPublisherIdentity, listHodorDiscussions } from "./gitlab.js";
import { mergeReviewStateFindings } from "./review-state.js";
import type { Platform, PostCommentResult, ReviewStateFinding } from "./types.js";
import { renderMarkdown } from "./render.js";
import { pushMetrics } from "./metrics.js";
import {
  hasBlockingFinding,
  parseFailOnPriority,
  type FailOnPriority,
} from "./review-policy.js";
import { loadReviewInstructionsFile } from "./review-instructions.js";
import { drainBufferedLogs, getWarnings, logger, setLogBuffering, setLogLevel } from "./utils/logger.js";

const program = new Command();

program
  .name("hodor")
  .description(
    "AI-powered code review agent for GitHub PRs, GitLab MRs, Gitea/Forgejo PRs, and local diffs.\n\n" +
      "Hodor uses an AI agent that clones the repository, checks out the PR branch,\n" +
      "and analyzes the code using tools (gh, git, glab) for metadata fetching and comment posting.\n\n" +
      "For local reviews, use --local with --diff-against to review changes in your current git repository.",
  )
  .version(packageJson.version)
  .argument("[pr-url]", "URL of the GitHub PR, GitLab MR, or Gitea/Forgejo PR to review (optional with --local)")
  .option(
    "--model <model>",
    "LLM model to use as provider/model-id (e.g., anthropic/claude-opus-5-5, openrouter/moonshotai/kimi-k2.6)",
    "anthropic/claude-opus-5-5",
  )
  .option(
    "--reasoning-effort <level>",
    "Reasoning effort level: minimal, low, medium, high, xhigh",
  )
  .option("-v, --verbose", "Enable verbose logging", false)
  .option(
    "--post",
    "Post the review directly to the PR/MR as a comment",
    false,
  )
  .option(
    "--focus <text>",
    "Focus this review, optionally narrowing the kinds of findings to report",
  )
  .option(
    "--instructions <path>",
    "Path to additive review instructions (repeatable, later files win conflicts)",
    (path: string, previous: string[]) => [...previous, path],
    [],
  )
  .option(
    "--workspace <dir>",
    "Workspace directory (creates temp dir if not specified)",
  )
  .option(
    "--review-style <style>",
    "How to post reviews on GitLab: summary (each review posts a new summary note and collapses older ones), inline diff comments, or hybrid (both). Default: hybrid.",
    "hybrid",
  )
  .option(
    "--code-quality <path>",
    "Write a cumulative GitLab Code Quality report to this path",
  )
  .option(
    "--commit-status",
    "Post a pass/fail status from all unresolved Hodor findings",
    false,
  )
  .option(
    "--require-delivery",
    "Exit non-zero if requested comments, statuses, or artifacts are not delivered",
    false,
  )
  .option(
    "--fail-on-priority <priority>",
    "Exit non-zero when findings at or above this severity exist: P0, P1, P2, or P3",
  )
  .option(
    "--ultrathink",
    "Enable maximum reasoning effort with extended thinking budget",
    false,
  )
  .option(
    "--bedrock-tags <json>",
    "JSON object of Bedrock requestMetadata for filtering model invocation logs (e.g., '{\"team\":\"platform\"}'). " +
      "NOT billing tags: AWS ignores this for cost allocation. For per-team cost in Cost Explorer, " +
      "pass a tagged application inference profile ARN via --model instead.",
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
  .option(
    "--full",
    "Force a full review of the entire source-vs-target diff, ignoring any previous hodor reviews on the MR/PR (disables incremental mode)",
    false,
  )
  .option(
    "--target-branch <ref>",
    "Override the target branch to diff against for a full review (default: the MR/PR's target branch). Only used with --full.",
  )
  .option(
    "--tiny-diff-fast-path",
    "For tiny, low-risk, fully embedded diffs, expose only submit_review so the review completes in one turn (cheaper; no repository exploration)",
    false,
  )
  .option(
    "--codemode",
    "Let the agent batch its read-only tool calls in Pi's codemode sandbox (cheaper on large reviews)",
    false,
  )
  .action(async (prUrl: string | undefined, cmdOpts: Record<string, unknown>) => {
    const verbose = cmdOpts.verbose as boolean;
    const post = cmdOpts.post as boolean;
    const model = cmdOpts.model as string;
    let reasoningEffort = cmdOpts.reasoningEffort as string | undefined;
    const focus = typeof cmdOpts.focus === "string" ? cmdOpts.focus : undefined;
    const instructionPaths = Array.isArray(cmdOpts.instructions)
      ? cmdOpts.instructions.filter((path): path is string => typeof path === "string")
      : [];
    const workspace = cmdOpts.workspace as string | undefined;
    const reviewStyle = cmdOpts.reviewStyle as "summary" | "inline" | "hybrid" | undefined;
    const codeQuality = cmdOpts.codeQuality as string | undefined;
    const commitStatus = cmdOpts.commitStatus as boolean;
    const requireDelivery = cmdOpts.requireDelivery as boolean;
    const failOnPriorityRaw = cmdOpts.failOnPriority as string | undefined;
    const ultrathink = cmdOpts.ultrathink as boolean;
    const bedrockTagsRaw = cmdOpts.bedrockTags as string | undefined;
    const prometheusPush = cmdOpts.prometheusPush as string | undefined;
    const localMode = cmdOpts.local as boolean;
    const diffAgainst = cmdOpts.diffAgainst as string;
    const full = cmdOpts.full as boolean;
    const targetBranchOverride = cmdOpts.targetBranch as string | undefined;
    const tinyDiffFastPath = cmdOpts.tinyDiffFastPath as boolean;
    const codemode = cmdOpts.codemode as boolean;

    if (!localMode && !prUrl) {
      console.error(chalk.red("Error: pr-url is required unless --local is specified"));
      process.exit(1);
    }
    if (localMode && post) {
      console.error(chalk.red("Error: --post is not supported in --local mode (no remote to post to)"));
      process.exit(1);
    }
    if (!["summary", "inline", "hybrid"].includes(reviewStyle ?? "hybrid")) {
      console.error(chalk.red("Error: --review-style must be one of: summary, inline, hybrid"));
      process.exit(1);
    }
    if (targetBranchOverride && !full) {
      console.error(chalk.yellow("Warning: --target-branch is only used with --full; ignoring it."));
    }
    if (full && localMode) {
      console.error(chalk.yellow("Warning: --full has no effect in --local mode (local reviews are always full)."));
    }
    if (requireDelivery && !post && !codeQuality) {
      console.error(chalk.red("Error: --require-delivery requires --post or --code-quality"));
      process.exit(1);
    }

    let failOnPriority: FailOnPriority | undefined;
    if (failOnPriorityRaw) {
      try {
        failOnPriority = parseFailOnPriority(failOnPriorityRaw.toUpperCase());
      } catch (error) {
        console.error(chalk.red(`Error: ${error instanceof Error ? error.message : error}`));
        process.exit(1);
      }
    }

    // Auto-detect CI environment
    const isCI = !!(process.env.CI || process.env.GITLAB_CI || process.env.GITHUB_ACTIONS || process.env.GITEA_ACTIONS || process.env.FORGEJO_ACTIONS);

    if (verbose) setLogLevel("debug");
    else if (isCI) setLogLevel("info");

    // Handle ultrathink
    if (ultrathink) {
      reasoningEffort = "xhigh";
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

    const gitlabCi = process.env.GITLAB_CI === "true";
    if (!verbose) setLogBuffering(true);
    const writeLog = (text: string): void => {
      process.stderr.write(text);
    };
    const trace = createTraceRenderer({ verbose, gitlabCi, write: writeLog });
    const printDiagnostics = (): void => {
      writeLog(formatDiagnostics(drainBufferedLogs(), gitlabCi, Math.floor(Date.now() / 1000)));
    };

    try {
      const instructions = instructionPaths.map((path) => loadReviewInstructionsFile(path));
      // Detect platform and warn about missing tokens
      let platform: Platform | "local" = "local";
      let target: ReviewTarget = { kind: "local", ref: diffAgainst };
      let metricsProject: string | undefined;
      let metricsOutcome = "reviewed";
      let requestedExitCode = 0;
      if (!localMode && prUrl) {
        platform = detectPlatform(prUrl);
        const parsedPr = parsePrUrl(prUrl);
        metricsProject = `${parsedPr.owner}/${parsedPr.repo}`;
        target = { kind: "remote", platform, project: metricsProject, number: parsedPr.prNumber };
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

      writeLog(`${formatStartLine({ version: packageJson.version, target, model, reasoningEffort, codemode })}\n`);
      for (const path of instructionPaths) logger.info(`Explicit instructions: ${path}`);
      if (focus) logger.info("Review focus: supplied");

      let reviewResult: Awaited<ReturnType<typeof reviewPr>>;
      try {
        reviewResult = await reviewPr({
          prUrl: localMode ? undefined : prUrl,
          model,
          reasoningEffort,
          instructions,
          focus,
          cleanup: !workspace,
          workspaceDir: workspace,
          includeMetricsFooter: post && !localMode,
          onEvent: (event) => trace.handle(event),
          bedrockTags,
          localMode,
          diffAgainst,
          full,
          targetBranchOverride,
          tinyDiffFastPath,
          codemode,
        });
      } finally {
        trace.close();
      }
      const {
        review,
        metricsFooter,
        headSha,
        metrics,
        workspacePath,
        cacheMarker,
        reusedReview,
        range,
        context,
      } = reviewResult;
      const reviewText = renderMarkdown(review);
      if (reusedReview) {
        logger.info("Reused the existing review for this HEAD; no LLM request was made.");
      }

      let reviewFindings: ReviewStateFinding[] = mergeReviewStateFindings(
        review.findings,
        [],
        process.env.CI_PROJECT_DIR ?? workspacePath,
      ).open;
      let gitlabReviewStateLoaded = false;
      let codeQualityWritten = false;

      let delivery: Delivery = localMode ? { kind: "local" } : { kind: "not-posted" };
      if (post && prUrl) {
        const useStructured = platform === "gitlab";

        let result: PostCommentResult;
        if (useStructured) {
          result = await postReviewStructured({
            prUrl,
            review,
            model,
            metricsFooter,
            reviewStyle: reviewStyle ?? "hybrid",
            commitStatus,
            headSha,
            workspacePath,
            cacheMarker,
            skipSummary: reusedReview,
            skipInline: reusedReview,
            reviewMode: metrics.reviewMode,
          });
        } else if (reusedReview) {
          result = {
            success: true,
            platform: detectPlatform(prUrl),
            summaryPosted: true,
          };
        } else {
          result = await postReviewComment({
            prUrl,
            reviewText,
            model,
            metricsFooter,
            headSha,
            cacheMarker,
          });
        }
        delivery = { kind: "posted", result };

        if (result.reviewFindings) {
          reviewFindings = result.reviewFindings;
          gitlabReviewStateLoaded = result.reviewStateComplete === true;
        }

        if (!result.success) {
          metricsOutcome = "delivery_failed";
          if (requireDelivery) requestedExitCode = 1;
        }
      }

      if (codeQuality) {
        try {
          if (platform === "gitlab" && prUrl && !gitlabReviewStateLoaded) {
            const parsed = parsePrUrl(prUrl);
            const discussions = await listHodorDiscussions(
              parsed.owner,
              parsed.repo,
              parsed.prNumber,
              parsed.host,
              await fetchGitlabPublisherIdentity(parsed.host),
            );
            reviewFindings = mergeReviewStateFindings(
              review.findings,
              discussions,
              process.env.CI_PROJECT_DIR ?? workspacePath,
              {
                suppressResolvedCurrent: reusedReview,
                resolvedFindingIds: review.resolved_findings,
              },
            ).open;
          }

          writeFileSync(codeQuality, formatCodeQualityReport(reviewFindings), "utf-8");
          codeQualityWritten = true;
          logger.info(`Wrote code quality report to ${codeQuality}`);
        } catch (err) {
          logger.warn(`Failed to write code quality report: ${err}`);
          metricsOutcome = "delivery_failed";
          if (requireDelivery) requestedExitCode = 1;
        }
      }

      if (requireDelivery && codeQuality && !codeQualityWritten) {
        requestedExitCode = 1;
      }
      let policyFailure: string | null = null;
      if (failOnPriority && hasBlockingFinding(review, failOnPriority)) {
        const maximumPriority = Number(failOnPriority.slice(1));
        const blocking = review.findings.filter(
          (finding) => finding.priority <= maximumPriority,
        ).length;
        policyFailure = `Review policy failed: ${blocking} finding(s) at ${failOnPriority} or higher`;
        metricsOutcome = "policy_failed";
        requestedExitCode = 1;
      }

      // Push metrics to Prometheus Pushgateway (best-effort, never fails the run)
      if (prometheusPush) {
        const labels: Record<string, string> = {
          platform,
          model,
          verdict: review.overall_correctness === "patch is correct" ? "correct" : "incorrect",
          outcome: metricsOutcome,
          review_mode: metrics.reviewMode ?? "unknown",
          reasoning_effort: metrics.reasoningEffort ?? reasoningEffort ?? "none",
          reused: metrics.reused ? "true" : "false",
          fast_path: metrics.fastPath ? "true" : "false",
        };
        if (metricsProject) labels.project = metricsProject;

        await pushMetrics({
          pushgatewayUrl: prometheusPush,
          metrics,
          // Reused reviews are delivery/cache events, not newly discovered
          // findings. Keep them observable without double-counting findings.
          findings: metrics.reused ? [] : review.findings,
          labels,
        });
      }

      printDiagnostics();
      writeLog(formatReviewSummary({
        platform,
        range,
        metrics,
        context,
        review,
        workspacePath,
        delivery,
        warnings: getWarnings(),
      }));
      if (policyFailure) writeLog(`${chalk.bold.red(policyFailure)}\n`);
      // The review markdown goes to stdout whenever it was not delivered.
      if (delivery.kind !== "posted" || !delivery.result.success) {
        console.log(`\n${reviewText}`);
      }
      if (requestedExitCode !== 0) process.exitCode = requestedExitCode;
    } catch (err) {
      let failurePlatform = "local";
      let failureProject: string | undefined;
      if (!localMode && prUrl) {
        try {
          failurePlatform = detectPlatform(prUrl);
          const parsed = parsePrUrl(prUrl);
          failureProject = `${parsed.owner}/${parsed.repo}`;
        } catch {
          // The original validation error is more useful than metrics-label parsing.
        }
      }

      // Failures are the expensive outlier case: a review can burn a full
      // budget of turns and then die before submit_review. Emit the same
      // telemetry shape as a successful review so they aren't invisible.
      logger.info(`Review telemetry: ${JSON.stringify({
        project: failureProject ?? null,
        mr: null,
        headSha: null,
        model,
        outcome: "review_failed",
        reviewMode: null,
        reasoningEffort: reasoningEffort ?? "auto",
        fastPath: null,
        reused: false,
        error: err instanceof Error ? err.message : String(err),
      })}`);
      printDiagnostics();
      console.error(chalk.bold.red(`Review failed: ${err instanceof Error ? err.message : err}`));
      if (verbose && err instanceof Error && err.stack) {
        console.error(chalk.dim(err.stack));
      }

      if (prometheusPush) {
        const labels: Record<string, string> = {
          platform: failurePlatform,
          model,
          verdict: "unknown",
          outcome: "review_failed",
        };
        if (failureProject) labels.project = failureProject;
        await pushMetrics({
          pushgatewayUrl: prometheusPush,
          metrics: {
            inputTokens: 0,
            outputTokens: 0,
            cacheReadTokens: 0,
            cacheWriteTokens: 0,
            totalTokens: 0,
            cost: 0,
            turns: 0,
            toolCalls: 0,
            durationSeconds: 0,
          },
          labels,
        });
      }
      process.exitCode = 1;
    }
  });

program.configureOutput({
  outputError: (message, write) => {
    if (/unknown option '--review-instructions(?:=|')/.test(message)) {
      write("error: --review-instructions was removed. Use --instructions <path>; files now add to the baseline review instead of replacing it.\n");
    } else if (/unknown option '--additional-instructions(?:=|')/.test(message)) {
      write("error: --additional-instructions was removed. Use --focus <text>.\n");
    } else {
      write(message);
    }
  },
});
program.parse();
