import chalk from "chalk";
import { logger } from "./utils/logger.js";
import type { ReviewMetrics } from "./types.js";

type FindingPriority = { priority: number };

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
  const work = `${metrics.turns} turns · ${metrics.toolCalls} tool calls · ${formatDuration(metrics.durationSeconds)}`;
  const cost = metrics.cost > 0 ? ` · \`$${metrics.cost.toFixed(4)}\`` : "";
  const tokens = [`\`${tok(totalInput)}\` input`];
  if (metrics.cacheReadTokens > 0) {
    const hitPct = ((metrics.cacheReadTokens / totalInput) * 100).toFixed(0);
    tokens.push(`\`${tok(metrics.cacheReadTokens)}\` cached (${hitPct}%)`);
  }
  tokens.push(`\`${tok(metrics.outputTokens)}\` output`);
  if (metrics.cacheWriteTokens > 0) {
    tokens.push(`\`${tok(metrics.cacheWriteTokens)}\` cache write`);
  }

  return `**Review metrics:** ${work}${cost}\n- Tokens: ${tokens.join(" · ")}`;
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
  if (metrics.cacheWriteTokens > 0) {
    tokenLine += dim(` · ${tok(metrics.cacheWriteTokens)} cache write`);
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

/**
 * Push review metrics to a Prometheus Pushgateway.
 * Failures are logged as warnings and never thrown.
 */
export async function pushMetrics(opts: {
  pushgatewayUrl: string;
  metrics: ReviewMetrics;
  findings?: FindingPriority[];
  labels?: Record<string, string>;
}): Promise<void> {
  const { pushgatewayUrl, metrics, findings = [], labels = {} } = opts;

  // Build label string for all metrics
  const formatLabels = (extraLabels: Record<string, string> = {}) => {
    const labelPairs = Object.entries({ ...labels, ...extraLabels })
      .map(([k, v]) => `${k}="${v.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/"/g, '\\"')}"`)
      .join(",");
    return labelPairs ? `{${labelPairs}}` : "";
  };
  const labelSuffix = formatLabels();

  const totalInput = metrics.inputTokens + metrics.cacheReadTokens;
  const cacheHitRatio = totalInput > 0 ? metrics.cacheReadTokens / totalInput : 0;
  const priorityCounts = [0, 1, 2, 3].map((priority) => ({
    priority,
    count: findings.filter((finding) => finding.priority === priority).length,
  }));
  const lines = [
    `# HELP hodor_review_input_tokens_total Total input tokens (fresh + cached)`,
    `# TYPE hodor_review_input_tokens_total gauge`,
    `hodor_review_input_tokens_total${labelSuffix} ${totalInput}`,
    `# HELP hodor_review_output_tokens_total Total output tokens`,
    `# TYPE hodor_review_output_tokens_total gauge`,
    `hodor_review_output_tokens_total${labelSuffix} ${metrics.outputTokens}`,
    `# HELP hodor_review_cache_read_tokens_total Tokens served from prompt cache`,
    `# TYPE hodor_review_cache_read_tokens_total gauge`,
    `hodor_review_cache_read_tokens_total${labelSuffix} ${metrics.cacheReadTokens}`,
    `# HELP hodor_review_cache_write_tokens_total Tokens written to prompt cache`,
    `# TYPE hodor_review_cache_write_tokens_total gauge`,
    `hodor_review_cache_write_tokens_total${labelSuffix} ${metrics.cacheWriteTokens}`,
    `# HELP hodor_review_cache_hit_ratio Fraction of input tokens served from cache (0-1)`,
    `# TYPE hodor_review_cache_hit_ratio gauge`,
    `hodor_review_cache_hit_ratio${labelSuffix} ${cacheHitRatio}`,
    `# HELP hodor_review_findings_total Number of findings at each priority level`,
    `# TYPE hodor_review_findings_total gauge`,
    ...priorityCounts.map(({ priority, count }) =>
      `hodor_review_findings_total${formatLabels({ priority: `P${priority}` })} ${count}`,
    ),
    `# HELP hodor_review_cost_dollars Cost of the review in USD`,
    `# TYPE hodor_review_cost_dollars gauge`,
    `hodor_review_cost_dollars${labelSuffix} ${metrics.cost}`,
    `# HELP hodor_review_turns_total Number of agent turns`,
    `# TYPE hodor_review_turns_total gauge`,
    `hodor_review_turns_total${labelSuffix} ${metrics.turns}`,
    `# HELP hodor_review_tool_calls_total Number of tool calls`,
    `# TYPE hodor_review_tool_calls_total gauge`,
    `hodor_review_tool_calls_total${labelSuffix} ${metrics.toolCalls}`,
    `# HELP hodor_review_duration_seconds Review duration in seconds`,
    `# TYPE hodor_review_duration_seconds gauge`,
    `hodor_review_duration_seconds${labelSuffix} ${metrics.durationSeconds}`,
    `# HELP hodor_review_diff_files Number of files included in the reviewed diff`,
    `# TYPE hodor_review_diff_files gauge`,
    `hodor_review_diff_files${labelSuffix} ${metrics.diffFiles ?? 0}`,
    `# HELP hodor_review_diff_additions Added lines included in the reviewed diff`,
    `# TYPE hodor_review_diff_additions gauge`,
    `hodor_review_diff_additions${labelSuffix} ${metrics.diffAdditions ?? 0}`,
    `# HELP hodor_review_diff_deletions Deleted lines included in the reviewed diff`,
    `# TYPE hodor_review_diff_deletions gauge`,
    `hodor_review_diff_deletions${labelSuffix} ${metrics.diffDeletions ?? 0}`,
    `# HELP hodor_review_diff_bytes Size of the reviewed diff in bytes`,
    `# TYPE hodor_review_diff_bytes gauge`,
    `hodor_review_diff_bytes${labelSuffix} ${metrics.diffBytes ?? 0}`,
    `# HELP hodor_review_reused Whether an existing identical-HEAD review was reused`,
    `# TYPE hodor_review_reused gauge`,
    `hodor_review_reused${labelSuffix} ${metrics.reused ? 1 : 0}`,
    `# HELP hodor_review_fast_path Whether the single-turn tiny-diff fast path gated tools to submit_review`,
    `# TYPE hodor_review_fast_path gauge`,
    `hodor_review_fast_path${labelSuffix} ${metrics.fastPath ? 1 : 0}`,
    "",
  ];
  const body = lines.join("\n");

  // POST to either a Prometheus Pushgateway base URL or a direct Prometheus text import endpoint.
  const baseUrl = pushgatewayUrl.replace(/\/+$/, "");
  const url = baseUrl.endsWith("/api/v1/import/prometheus")
    ? baseUrl
    : `${baseUrl}/metrics/job/hodor`;

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body,
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      logger.warn(`Metrics endpoint returned ${res.status}: ${text.slice(0, 200)}`);
    } else {
      logger.info("Metrics pushed successfully");
    }
  } catch (err) {
    logger.warn(`Failed to push metrics: ${err instanceof Error ? err.message : err}`);
  }
}
