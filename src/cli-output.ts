import chalk from "chalk";
import type { AgentProgressEvent } from "./agent.js";
import { formatDuration, formatTokenCount } from "./metrics.js";
import { displayModel } from "./publisher.js";
import type {
  Platform,
  PostCommentResult,
  ReviewContextManifest,
  ReviewMetrics,
  ReviewOutput,
  ReviewRange,
} from "./types.js";
import { relativizeWorkspacePath } from "./utils/path.js";

const MAX_SUMMARY_WARNINGS = 5;

const TRACE_SECTION = "hodor_trace";
const DIAGNOSTICS_SECTION = "hodor_diagnostics";
const MAX_TRACE_ARGS = 160;
const LABEL_WIDTH = 11;
const TRACE_TURN_WIDTH = 9;
const TRACE_TOOL_WIDTH = 7;

export type ReviewTarget =
  | { kind: "remote"; platform: Platform; project: string; number: number }
  | { kind: "local"; ref: string };

/** `Hodor 0.10.0 · acme/app !42 · claude-opus-5-5 (high) · codemode` */
export function formatStartLine(opts: {
  version: string;
  target: ReviewTarget;
  model: string;
  reasoningEffort?: string;
  codemode: boolean;
}): string {
  const target = opts.target.kind === "local"
    ? `local diff vs ${opts.target.ref}`
    : `${opts.target.project} ${opts.target.platform === "gitlab" ? "!" : "#"}${opts.target.number}`;
  const model = `${displayModel(opts.model)} (${opts.reasoningEffort ?? "adaptive"})`;
  return [`Hodor ${opts.version}`, target, model, ...(opts.codemode ? ["codemode"] : [])].join(" · ");
}

/**
 * Open a log section. In GitLab CI this is a collapsed section marker;
 * elsewhere it is a plain header line.
 */
export function formatSectionStart(name: string, header: string, gitlabCi: boolean, nowSeconds: number): string {
  return gitlabCi
    ? `\x1b[0Ksection_start:${nowSeconds}:${name}[collapsed=true]\r\x1b[0K${header}\n`
    : `${chalk.bold(header)}\n`;
}

/** Close a log section. Empty outside GitLab CI. */
export function formatSectionEnd(name: string, gitlabCi: boolean, nowSeconds: number): string {
  return gitlabCi ? `\x1b[0Ksection_end:${nowSeconds}:${name}\r\x1b[0K\n` : "";
}

function unixSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

export interface TraceRenderer {
  handle(event: AgentProgressEvent): void;
  /** Close the trace section, if one is open. */
  close(): void;
}

/**
 * Render agent progress events. The default mode prints one line per tool
 * call inside an "Agent trace" section. Verbose mode prints result previews,
 * reasoning, and text deltas live, with no section.
 */
export function createTraceRenderer(opts: {
  verbose: boolean;
  gitlabCi: boolean;
  write: (text: string) => void;
  now?: () => number;
}): TraceRenderer {
  const { verbose, gitlabCi, write, now = unixSeconds } = opts;
  return verbose ? createVerboseTrace(write) : createCompactTrace(gitlabCi, write, now);
}

function createCompactTrace(
  gitlabCi: boolean,
  write: (text: string) => void,
  now: () => number,
): TraceRenderer {
  let open = false;
  let turn = 0;
  const line = (text: string): void => {
    if (!open) {
      write(formatSectionStart(TRACE_SECTION, "Agent trace", gitlabCi, now()));
      open = true;
    }
    write(`${text}\n`);
  };
  const callPrefix = (event: AgentProgressEvent): string =>
    event.parentToolCallId
      ? `${" ".repeat(TRACE_TURN_WIDTH)}  ↳ `
      : `turn ${turn}`.padEnd(TRACE_TURN_WIDTH);

  return {
    handle(event) {
      switch (event.type) {
        case "turn_start":
          turn = event.turnIndex ?? turn + 1;
          break;
        case "tool_start": {
          const name = (event.toolName ?? "tool").padEnd(TRACE_TOOL_WIDTH - 1);
          const args = event.toolArgs ? truncate(event.toolArgs, MAX_TRACE_ARGS) : "";
          line(`${callPrefix(event)}${name} ${args}`.trimEnd());
          break;
        }
        case "tool_end":
          if (event.isError) {
            const firstLine = (event.result ?? "").split("\n")[0]?.trim() || "error";
            const indent = event.parentToolCallId ? TRACE_TURN_WIDTH + 4 : TRACE_TURN_WIDTH;
            line(chalk.red(`${" ".repeat(indent)}✗ ${event.toolName ?? "tool"} failed: ${truncate(firstLine, MAX_TRACE_ARGS)}`));
          }
          break;
        case "retry":
          line(formatRetryLine(event));
          break;
        case "compaction":
          line(formatCompactionLine(event));
          break;
      }
    },
    close() {
      if (!open) return;
      write(formatSectionEnd(TRACE_SECTION, gitlabCi, now()));
      open = false;
    },
  };
}

const VERBOSE_TOOL_LABELS: Record<string, string> = {
  git_diff: "git diff",
  read: "cat",
  grep: "grep",
  find: "find",
  ls: "ls",
};

function createVerboseTrace(write: (text: string) => void): TraceRenderer {
  const line = (text: string): void => write(`${text}\n`);
  return {
    handle(event) {
      switch (event.type) {
        case "agent_start":
          line(chalk.dim("▶ Agent started"));
          break;
        case "turn_start":
          line(chalk.dim(`\n── Turn ${event.turnIndex ?? "?"} ──`));
          break;
        case "tool_start": {
          const label = VERBOSE_TOOL_LABELS[event.toolName ?? ""] ?? event.toolName;
          const nested = event.parentToolCallId ? "  ↳ " : "";
          const args = event.toolArgs ? ` ${truncate(event.toolArgs, MAX_TRACE_ARGS)}` : "";
          line(chalk.green(`  ${nested}${label}${args}`));
          break;
        }
        case "tool_end": {
          if (event.isError) line(chalk.red("  ✗ error"));
          if (!event.result) break;
          const lines = event.result.split("\n");
          let chars = 0;
          for (let i = 0; i < Math.min(lines.length, 15); i++) {
            if (chars + lines[i].length > 400) {
              line(chalk.dim(`    …(${lines.length - i} more lines)`));
              break;
            }
            line(chalk.dim(`    ${lines[i]}`));
            chars += lines[i].length;
          }
          break;
        }
        case "text_delta":
          if (event.delta) write(event.delta);
          break;
        case "thinking_delta":
          if (event.delta) write(chalk.dim(event.delta));
          break;
        case "retry":
          line(formatRetryLine(event));
          break;
        case "compaction":
          line(formatCompactionLine(event));
          break;
        case "agent_end":
          line(chalk.dim("\n▶ Extracting review..."));
          break;
      }
    },
    close() {},
  };
}

function formatRetryLine(event: AgentProgressEvent): string {
  return event.phase === "start"
    ? chalk.yellow(`  ↻ Retry ${event.attempt}/${event.maxAttempts} in ${event.delayMs}ms: ${event.reason}`)
    : chalk.dim(`  ↻ Retry ${event.success ? "succeeded" : "failed"} (attempt ${event.attempt})`);
}

function formatCompactionLine(event: AgentProgressEvent): string {
  return chalk.dim(`  ⧉ Compaction ${event.phase === "start" ? "started" : "finished"} (${event.reason})`);
}

/** The buffered log lines in a "Diagnostics" section. Empty when there are none. */
export function formatDiagnostics(lines: readonly string[], gitlabCi: boolean, nowSeconds: number): string {
  if (lines.length === 0) return "";
  return (
    formatSectionStart(DIAGNOSTICS_SECTION, "Diagnostics", gitlabCi, nowSeconds) +
    lines.map((line) => `${line}\n`).join("") +
    formatSectionEnd(DIAGNOSTICS_SECTION, gitlabCi, nowSeconds)
  );
}

export type Delivery =
  | { kind: "local" }
  | { kind: "not-posted" }
  | { kind: "posted"; result: PostCommentResult };

export interface ReviewSummary {
  platform: Platform | "local";
  range: ReviewRange;
  metrics: ReviewMetrics;
  context: ReviewContextManifest | null;
  review: ReviewOutput;
  workspacePath: string;
  delivery: Delivery;
  /** Warning and error messages from this run, shown in full outside any collapsed section. */
  warnings: readonly string[];
}

function plural(count: number, singular: string, pluralForm = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

function describeMode(mode: ReviewMetrics["reviewMode"], platform: Platform | "local"): string {
  const change = platform === "gitlab" ? "MR" : "PR";
  switch (mode) {
    case "incremental":
      return "incremental since the last review";
    case "snapshot":
      return platform === "gitlab"
        ? `full ${change} diff after a rebase`
        : "snapshot delta since the last review, history rewritten";
    case "local":
      return "local diff";
    case "reused":
      return "reused review for this HEAD";
    default:
      return `full ${change} diff`;
  }
}

function formatReviewing(summary: ReviewSummary): string {
  const { range, metrics, platform } = summary;
  const head = range.headSha ? range.headSha.slice(0, 8) : "working tree";
  const base = range.baseSha ? `${range.targetBranch}@${range.baseSha.slice(0, 8)}` : range.targetBranch;
  return `${head} against ${base} (${describeMode(metrics.reviewMode, platform)})`;
}

function formatDiff(metrics: ReviewMetrics): string {
  const source = metrics.diffEmbedded ? "embedded" : "via git_diff";
  return `${plural(metrics.diffFiles ?? 0, "file")}, +${metrics.diffAdditions ?? 0} −${metrics.diffDeletions ?? 0} (${source})`;
}

function formatContext(context: ReviewContextManifest): string[] {
  const { open, fixedWaiting, resolved, droppedByLimit } = context.hodorThreads;
  const total = open + fixedWaiting + resolved + droppedByLimit;
  const threads = total === 0
    ? "No Hodor threads"
    : `${plural(total, "Hodor thread")}: ${open} open, ${fixedWaiting} fixed waiting, ${resolved} resolved` +
      (droppedByLimit > 0 ? `, ${droppedByLimit} not shown` : "");
  const { included, droppedByBudget } = context.humanComments;
  const comments =
    `${plural(included, "human comment")} included` +
    (droppedByBudget > 0 ? `, ${droppedByBudget} dropped (budget)` : "");
  return [threads, comments];
}

function formatResult(review: ReviewOutput, workspacePath: string): string[] {
  const count = review.findings.length;
  const verified = review.resolved_findings?.length ?? 0;
  const headline =
    (count === 0 ? "No new findings" : plural(count, "new finding")) +
    (verified > 0 ? ` · ${plural(verified, "earlier finding")} confirmed fixed` : "");
  const findings = review.findings.map((finding) => {
    const title = finding.title.replace(/^\[P[0-3]\]\s*/, "");
    const path = relativizeWorkspacePath(finding.code_location.absolute_file_path, workspacePath);
    return `[P${finding.priority}] ${title}  ${path}:${finding.code_location.line_range.start}`;
  });
  return [headline, ...findings];
}

function formatPosted(delivery: Delivery, reused: boolean): string | null {
  if (delivery.kind === "local") return null;
  if (delivery.kind === "not-posted") return "not posted (use --post)";
  const { result } = delivery;
  if (!result.success) return chalk.red(`failed: ${result.error ?? "review delivery was incomplete"}`);
  if (result.platform !== "gitlab") {
    const number = result.prNumber ?? result.mrNumber;
    return reused ? "nothing new (review reused)" : `review comment${number ? ` on PR #${number}` : ""}`;
  }
  const parts: string[] = [];
  if (result.summaryPosted) parts.push(result.summaryUrl ? `summary ${result.summaryUrl}` : "summary");
  if (result.inlineCreated) parts.push(`${result.inlineCreated} inline`);
  if (result.fixedReplies) parts.push(plural(result.fixedReplies, "fixed reply", "fixed replies"));
  if (parts.length === 0) return reused ? "nothing new (review reused)" : "nothing new";
  return parts.join(" · ");
}

function formatCost(metrics: ReviewMetrics): string {
  if (metrics.reused) return `$${metrics.cost.toFixed(4)} · no LLM request (review reused)`;
  const input = metrics.inputTokens + metrics.cacheReadTokens;
  const cached = metrics.cacheReadTokens > 0 ? ` (${formatTokenCount(metrics.cacheReadTokens)} cached)` : "";
  const nested = metrics.nestedToolCalls !== undefined ? ` (${metrics.nestedToolCalls} in codemode)` : "";
  return [
    `$${metrics.cost.toFixed(4)}`,
    formatDuration(metrics.durationSeconds),
    `${formatTokenCount(input)} in${cached}, ${formatTokenCount(metrics.outputTokens)} out`,
    `${plural(metrics.turns, "turn")}, ${plural(metrics.toolCalls, "tool call")}${nested}`,
  ].join(" · ");
}

/** The end-of-run summary: one labelled row per aspect, continuation lines aligned. */
export function formatReviewSummary(summary: ReviewSummary): string {
  const rows: Array<[string, string[]]> = [["Reviewing", [formatReviewing(summary)]]];
  if (!summary.metrics.reused) rows.push(["Diff", [formatDiff(summary.metrics)]]);
  if (summary.context) rows.push(["Context", formatContext(summary.context)]);
  rows.push(["Result", formatResult(summary.review, summary.workspacePath)]);
  const posted = formatPosted(summary.delivery, summary.metrics.reused === true);
  if (posted) rows.push(["Posted", [posted]]);
  rows.push(["Cost", [formatCost(summary.metrics)]]);
  if (summary.warnings.length > 0) {
    const shown = summary.warnings.slice(0, MAX_SUMMARY_WARNINGS).map((warning) =>
      chalk.yellow(warning.length > 200 ? `${warning.slice(0, 199)}…` : warning));
    const more = summary.warnings.length - shown.length;
    rows.push(["Warnings", more > 0 ? [...shown, chalk.yellow(`…and ${more} more in the log above`)] : shown]);
  }

  const lines: string[] = [];
  for (const [label, values] of rows) {
    values.forEach((value, index) => {
      lines.push(`${(index === 0 ? label : "").padEnd(LABEL_WIDTH)}${value}`);
    });
  }
  return `${lines.join("\n")}\n`;
}
