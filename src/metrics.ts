import chalk from "chalk";
import type { ReviewMetrics } from "./types.js";

function tok(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function formatDuration(seconds: number): string {
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }
  return `${seconds}s`;
}

export function formatMetricsMarkdown(metrics: ReviewMetrics): string {
  const totalInput = metrics.inputTokens + metrics.cacheReadTokens;
  const parts = [`in \`${tok(totalInput)}\``];
  if (metrics.cacheReadTokens > 0) {
    parts.push(`cached \`${tok(metrics.cacheReadTokens)}\``);
  }
  parts.push(`out \`${tok(metrics.outputTokens)}\``);

  const lines = [
    `**Review Metrics** — ${metrics.turns} turns, ${metrics.toolCalls} tool calls, ${formatDuration(metrics.durationSeconds)}`,
    `- Tokens: ${parts.join(" | ")} (total \`${tok(metrics.totalTokens)}\`)`,
  ];
  if (metrics.cost > 0) {
    lines.push(`- Cost: \`$${metrics.cost.toFixed(4)}\``);
  }
  return lines.join("\n");
}

export function printMetrics(metrics: ReviewMetrics, stream: NodeJS.WritableStream = process.stderr): void {
  const dim = chalk.dim;
  const bold = chalk.bold;
  const cyan = chalk.cyan;

  const write = (line: string) => stream.write(line + "\n");

  write("");
  write(dim("─".repeat(50)));

  // Tokens — inputTokens may be only fresh tokens (SDK reports cache hits separately)
  const totalInput = metrics.inputTokens + metrics.cacheReadTokens;
  let tokenLine = `${dim("Tokens:")}  ${bold(tok(totalInput))} in`;
  if (metrics.cacheReadTokens > 0) {
    const hitPct = ((metrics.cacheReadTokens / totalInput) * 100).toFixed(0);
    tokenLine += dim(` (${tok(metrics.cacheReadTokens)} cached ${hitPct}% · ${tok(metrics.inputTokens)} fresh)`);
  }
  tokenLine += `  ${bold(tok(metrics.outputTokens))} out`;
  tokenLine += dim(`  (${tok(metrics.totalTokens)} total)`);
  write(tokenLine);

  // Agent work
  write(
    `${dim("Agent:")}   ${bold(String(metrics.turns))} turns  ${bold(String(metrics.toolCalls))} tool calls  ${cyan(formatDuration(metrics.durationSeconds))}`,
  );

  // Cost
  if (metrics.cost > 0) {
    write(`${dim("Cost:")}    ${bold("$" + metrics.cost.toFixed(4))}`);
  }

  write(dim("─".repeat(50)));
}
