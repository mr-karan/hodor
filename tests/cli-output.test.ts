import chalk from "chalk";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentProgressEvent } from "../src/agent.js";
import {
  createTraceRenderer,
  formatDiagnostics,
  formatReviewSummary,
  formatStartLine,
  type ReviewSummary,
} from "../src/cli-output.js";
import type { ReviewFinding, ReviewMetrics } from "../src/types.js";
import {
  drainBufferedLogs,
  getWarnings,
  logger,
  setLogBuffering,
  setLogLevel,
} from "../src/utils/logger.js";

const HEAD = "0123456789abcdef0123456789abcdef01234567";
const BASE = "fedcba9876543210fedcba9876543210fedcba98";
const WORKSPACE = "/builds/acme/app";
const NOW = 1_790_000_000;

beforeAll(() => {
  chalk.level = 0;
});

function finding(title: string, priority: 0 | 1 | 2 | 3, path: string, line: number): ReviewFinding {
  return {
    title,
    body: "Body.",
    priority,
    code_location: { absolute_file_path: `${WORKSPACE}/${path}`, line_range: { start: line, end: line } },
  };
}

const CODEMODE_METRICS: ReviewMetrics = {
  inputTokens: 150_000,
  outputTokens: 12_300,
  cacheReadTokens: 1_100_000,
  cacheWriteTokens: 40_000,
  totalTokens: 1_302_300,
  cost: 0.4213,
  turns: 24,
  toolCalls: 61,
  nestedToolCalls: 18,
  codemodeCalls: 4,
  durationSeconds: 192,
  reviewMode: "incremental",
  reasoningEffort: "high",
  diffFiles: 12,
  diffAdditions: 340,
  diffDeletions: 25,
  diffBytes: 40_000,
  diffEmbedded: true,
  reused: false,
  fastPath: false,
};

function gitlabPostSummary(): ReviewSummary {
  return {
    platform: "gitlab",
    range: { headSha: HEAD, targetBranch: "main", baseSha: BASE },
    metrics: CODEMODE_METRICS,
    context: {
      hodorThreads: { open: 3, fixedWaiting: 1, resolved: 2, droppedByLimit: 1 },
      humanComments: { included: 12, droppedByBudget: 2 },
      priorHodorReviews: 3,
    },
    review: {
      findings: [
        finding("[P1] Missing authorization check", 1, "src/foo.py", 42),
        finding("Cache key ignores tenant", 2, "src/cache.py", 7),
      ],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Two issues.",
      resolved_findings: ["57a2a375"],
    },
    workspacePath: WORKSPACE,
    delivery: {
      kind: "posted",
      result: {
        success: true,
        platform: "gitlab",
        mrNumber: 42,
        summaryPosted: true,
        summaryUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42#note_900",
        inlineCreated: 2,
        fixedReplies: 1,
        fixedAwaiting: 2,
      },
    },
    warnings: ["WARN Invalid submit_review payload: findings[0].priority must be 0-3", "WARN Failed to reply on thread 84608: 500"],
  };
}

describe("formatStartLine", () => {
  it("names the project, MR, model, reasoning, and codemode", () => {
    expect(formatStartLine({
      version: "0.10.0",
      target: { kind: "remote", platform: "gitlab", project: "acme/app", number: 42 },
      model: "amazon-bedrock/arn:aws:bedrock:ap-south-1:123:application-inference-profile/abc@global.anthropic.claude-opus-5",
      reasoningEffort: "high",
      codemode: true,
    })).toBe("Hodor 0.10.0 · acme/app !42 · global.anthropic.claude-opus-5 (high) · codemode");
  });

  it("describes a local review", () => {
    expect(formatStartLine({
      version: "0.10.0",
      target: { kind: "local", ref: "HEAD~1" },
      model: "anthropic/claude-opus-5-5",
      codemode: false,
    })).toBe("Hodor 0.10.0 · local diff vs HEAD~1 · anthropic/claude-opus-5-5 (adaptive)");
  });
});

describe("formatReviewSummary", () => {
  it("summarizes a GitLab post with findings, fixes, codemode, and warnings", () => {
    expect(formatReviewSummary(gitlabPostSummary())).toBe([
      "Reviewing  01234567 against main@fedcba98 (incremental since the last review)",
      "Diff       12 files, +340 −25 (embedded)",
      "Context    7 Hodor threads: 3 open, 1 fixed waiting, 2 resolved, 1 not shown",
      "           12 human comments included, 2 dropped (budget)",
      "Result     2 new findings · 1 earlier finding confirmed fixed",
      "           [P1] Missing authorization check  src/foo.py:42",
      "           [P2] Cache key ignores tenant  src/cache.py:7",
      "Posted     summary https://gitlab.example.com/acme/app/-/merge_requests/42#note_900 · 2 inline · 1 fixed reply",
      "Cost       $0.4213 · 3m 12s · 1.25M in (1.10M cached), 12.3K out · 24 turns, 61 tool calls (18 in codemode)",
      "Warnings   WARN Invalid submit_review payload: findings[0].priority must be 0-3",
      "           WARN Failed to reply on thread 84608: 500",
      "",
    ].join("\n"));
  });

  it("summarizes a local review that was not posted", () => {
    const summary = formatReviewSummary({
      platform: "local",
      range: { headSha: null, targetBranch: "HEAD~1", baseSha: null },
      metrics: {
        ...CODEMODE_METRICS,
        nestedToolCalls: undefined,
        cacheReadTokens: 0,
        inputTokens: 8_000,
        reviewMode: "local",
        diffFiles: 1,
        diffAdditions: 3,
        diffDeletions: 1,
        diffEmbedded: false,
        turns: 1,
        toolCalls: 1,
        durationSeconds: 9,
      },
      context: null,
      review: { findings: [], overall_correctness: "patch is correct", overall_explanation: "Fine." },
      workspacePath: WORKSPACE,
      delivery: { kind: "local" },
      warnings: [],
    });
    expect(summary).toBe([
      "Reviewing  working tree against HEAD~1 (local diff)",
      "Diff       1 file, +3 −1 (via git_diff)",
      "Result     No new findings",
      "Cost       $0.4213 · 9s · 8.0K in, 12.3K out · 1 turn, 1 tool call",
      "",
    ].join("\n"));
  });

  it("reports a failed post and an unposted remote review", () => {
    const failed = formatReviewSummary({
      ...gitlabPostSummary(),
      delivery: { kind: "posted", result: { success: false, platform: "gitlab", error: "summary comment: 403 Forbidden" } },
      warnings: [],
    });
    expect(failed).toContain("\nPosted     failed: summary comment: 403 Forbidden\n");
    expect(failed).not.toContain("Warnings");

    const unposted = formatReviewSummary({ ...gitlabPostSummary(), delivery: { kind: "not-posted" } });
    expect(unposted).toContain("\nPosted     not posted (use --post)\n");
  });

  it("omits the diff and token detail for a reused review", () => {
    const summary = formatReviewSummary({
      ...gitlabPostSummary(),
      metrics: { ...CODEMODE_METRICS, reused: true, reviewMode: "reused", cost: 0 },
      context: null,
      delivery: { kind: "posted", result: { success: true, platform: "gitlab", summaryPosted: false, inlineCreated: 0 } },
    });
    expect(summary).not.toContain("Diff ");
    expect(summary).toContain("(reused review for this HEAD)");
    expect(summary).toContain("Posted     nothing new (review reused)");
    expect(summary).toContain("Cost       $0.0000 · no LLM request (review reused)");
  });
});

describe("createTraceRenderer", () => {
  const events: AgentProgressEvent[] = [
    { type: "agent_start" },
    { type: "turn_start", turnIndex: 12 },
    { type: "tool_start", toolName: "read", toolArgs: "src/foo.py", toolCallId: "a" },
    { type: "tool_end", toolName: "read", result: "line one\nline two", toolCallId: "a" },
    { type: "turn_start", turnIndex: 13 },
    { type: "tool_start", toolName: "codemode", toolArgs: "6-line script", toolCallId: "b" },
    { type: "tool_start", toolName: "grep", toolArgs: '"has_role\\(" in src', toolCallId: "c", parentToolCallId: "b" },
    { type: "tool_end", toolName: "grep", result: "src/auth.py:3: has_role(", toolCallId: "c", parentToolCallId: "b" },
    { type: "tool_start", toolName: "read", toolArgs: "vendor/x.js", toolCallId: "d", parentToolCallId: "b" },
    { type: "tool_end", toolName: "read", isError: true, result: "Path is not tracked: vendor/x.js\nmore detail", toolCallId: "d", parentToolCallId: "b" },
    { type: "tool_end", toolName: "codemode", result: "done", toolCallId: "b" },
    { type: "thinking_delta", delta: "hmm" },
    { type: "text_delta", delta: "Reviewing." },
    { type: "retry", phase: "start", attempt: 1, maxAttempts: 3, delayMs: 2000, reason: "429 overloaded" },
    { type: "agent_end" },
  ];

  function render(verbose: boolean, gitlabCi: boolean): string {
    let out = "";
    const trace = createTraceRenderer({ verbose, gitlabCi, write: (text) => { out += text; }, now: () => NOW });
    for (const event of events) trace.handle(event);
    trace.close();
    return out;
  }

  it("prints one line per call in a collapsed GitLab section, with nested calls indented", () => {
    expect(render(false, true)).toBe([
      `\x1b[0Ksection_start:${NOW}:hodor_trace[collapsed=true]\r\x1b[0KAgent trace`,
      "turn 12  read   src/foo.py",
      "turn 13  codemode 6-line script",
      '           ↳ grep   "has_role\\(" in src',
      "           ↳ read   vendor/x.js",
      "             ✗ read failed: Path is not tracked: vendor/x.js",
      "  ↻ Retry 1/3 in 2000ms: 429 overloaded",
      `\x1b[0Ksection_end:${NOW}:hodor_trace\r\x1b[0K`,
      "",
    ].join("\n"));
  });

  it("prints a plain header and no section markers outside GitLab CI", () => {
    const out = render(false, false);
    expect(out.startsWith("Agent trace\nturn 12  read   src/foo.py\n")).toBe(true);
    expect(out).not.toContain("section_");
    expect(out).not.toContain("line one");
    expect(out).not.toContain("hmm");
  });

  it("keeps previews and deltas in verbose mode", () => {
    const out = render(true, true);
    expect(out).not.toContain("section_");
    expect(out).toContain("  cat src/foo.py\n    line one\n    line two\n");
    expect(out).toContain("    ↳ grep \"has_role\\(\" in src");
    expect(out).toContain("hmmReviewing.");
    expect(out).toContain("▶ Extracting review...");
  });

  it("opens no section when no trace event arrives", () => {
    let out = "";
    const trace = createTraceRenderer({ verbose: false, gitlabCi: true, write: (text) => { out += text; } });
    trace.handle({ type: "agent_start" });
    trace.close();
    expect(out).toBe("");
  });
});

describe("logger buffering", () => {
  let stderr: string;

  beforeEach(() => {
    stderr = "";
    vi.spyOn(process.stderr, "write").mockImplementation((chunk: string | Uint8Array) => {
      stderr += String(chunk);
      return true;
    });
    setLogLevel("info");
    setLogBuffering(true);
    drainBufferedLogs();
  });

  afterEach(() => {
    setLogBuffering(false);
    setLogLevel("warn");
    vi.restoreAllMocks();
  });

  it("buffers info for Diagnostics and prints warnings live, counting them", () => {
    const warningsBefore = getWarnings().length;
    logger.info("Embedding diff in prompt (4000 bytes, raw: 4000 bytes)");
    logger.info('Review telemetry: {"outcome":"reviewed"}');
    logger.warn("Failed to list Hodor finding threads for the prompt: 500");
    logger.debug("hidden at info level");

    expect(stderr).toContain("WARN  Failed to list Hodor finding threads");
    expect(stderr).not.toContain("Embedding diff");
    expect(getWarnings().slice(warningsBefore)).toEqual(["WARN Failed to list Hodor finding threads for the prompt: 500"]);

    const diagnostics = formatDiagnostics(drainBufferedLogs(), true, NOW);
    expect(diagnostics.split("\n")).toEqual([
      `\x1b[0Ksection_start:${NOW}:hodor_diagnostics[collapsed=true]\r\x1b[0KDiagnostics`,
      expect.stringMatching(/ INFO  Embedding diff in prompt/),
      expect.stringMatching(/ INFO  Review telemetry: \{"outcome":"reviewed"\}$/),
      `\x1b[0Ksection_end:${NOW}:hodor_diagnostics\r\x1b[0K`,
      "",
    ]);
    expect(drainBufferedLogs()).toEqual([]);
  });

  it("prints everything live when buffering is off", () => {
    setLogBuffering(false);
    logger.info("live info line");
    expect(stderr).toContain("INFO  live info line");
    expect(drainBufferedLogs()).toEqual([]);
  });

  it("renders no Diagnostics section when nothing was buffered", () => {
    expect(formatDiagnostics([], true, NOW)).toBe("");
  });
});

describe("sample CI job log", () => {
  beforeEach(() => {
    vi.useFakeTimers({ now: new Date(NOW * 1000) });
  });

  afterEach(() => {
    vi.useRealTimers();
    setLogBuffering(false);
    setLogLevel("warn");
    vi.restoreAllMocks();
  });

  /** The full default-mode log of a GitLab CI review, from fixed inputs. */
  function renderSampleLog(): string {
    let log = "";
    vi.spyOn(process.stderr, "write").mockImplementation((chunk: string | Uint8Array) => {
      log += String(chunk);
      return true;
    });
    const write = (text: string): void => {
      process.stderr.write(text);
    };
    setLogLevel("info");
    setLogBuffering(true);
    drainBufferedLogs();
    const warningsBefore = getWarnings().length;

    write(`${formatStartLine({
      version: "0.10.0",
      target: { kind: "remote", platform: "gitlab", project: "acme/app", number: 42 },
      model: "anthropic/claude-opus-5-5",
      reasoningEffort: "high",
      codemode: true,
    })}\n`);
    logger.info("Review instructions: bundled default");
    logger.info("Incremental mode: previous review at fedcba98");
    logger.info("Embedding diff in prompt (40000 bytes, raw: 41200 bytes)");
    logger.info('Review context: {"hodorThreads":{"open":3,"fixedWaiting":1,"resolved":2,"droppedByLimit":1},"humanComments":{"included":12,"droppedByBudget":2},"priorHodorReviews":3}');

    const trace = createTraceRenderer({ verbose: false, gitlabCi: true, write, now: () => NOW });
    const events: AgentProgressEvent[] = [
      { type: "turn_start", turnIndex: 1 },
      { type: "tool_start", toolName: "codemode", toolArgs: "9-line script", toolCallId: "s1" },
      { type: "tool_start", toolName: "read", toolArgs: "src/foo.py", toolCallId: "n1", parentToolCallId: "s1" },
      { type: "tool_start", toolName: "grep", toolArgs: '"has_role\\(" in src', toolCallId: "n2", parentToolCallId: "s1" },
      { type: "turn_start", turnIndex: 2 },
      { type: "tool_start", toolName: "read", toolArgs: "src/cache.py", toolCallId: "t2" },
      { type: "tool_end", toolName: "read", isError: true, result: "Line offset 900 is past the end of src/cache.py (120 lines)", toolCallId: "t2" },
      { type: "retry", phase: "start", attempt: 1, maxAttempts: 3, delayMs: 2000, reason: "429 overloaded" },
      { type: "retry", phase: "end", attempt: 1, success: true },
      { type: "turn_start", turnIndex: 3 },
      { type: "tool_start", toolName: "submit_review", toolArgs: "", toolCallId: "t3" },
    ];
    for (const event of events) trace.handle(event);
    logger.warn("Invalid submit_review payload: findings[0].priority must be 0-3");
    trace.close();

    logger.info("Created 2 inline draft note(s)");
    logger.info("Posted summary note 900; collapsed 1 older summary note(s)");
    logger.info('Review telemetry: {"project":"acme/app","mr":42,"outcome":"reviewed","findings":2}');
    write(formatDiagnostics(drainBufferedLogs(), true, NOW));
    write(formatReviewSummary({ ...gitlabPostSummary(), warnings: getWarnings().slice(warningsBefore) }));
    return log;
  }

  it("renders a short default log with collapsed trace and diagnostics", () => {
    const visible = renderSampleLog().replaceAll("\x1b", "\\e").replaceAll("\r", "\\r");
    expect(visible).toMatchInlineSnapshot(`
      "Hodor 0.10.0 · acme/app !42 · anthropic/claude-opus-5-5 (high) · codemode
      \\e[0Ksection_start:1790000000:hodor_trace[collapsed=true]\\r\\e[0KAgent trace
      turn 1   codemode 9-line script
                 ↳ read   src/foo.py
                 ↳ grep   "has_role\\(" in src
      turn 2   read   src/cache.py
               ✗ read failed: Line offset 900 is past the end of src/cache.py (120 lines)
        ↻ Retry 1/3 in 2000ms: 429 overloaded
        ↻ Retry succeeded (attempt 1)
      turn 3   submit_review
      2026-09-21T14:13:20.000Z WARN  Invalid submit_review payload: findings[0].priority must be 0-3
      \\e[0Ksection_end:1790000000:hodor_trace\\r\\e[0K
      \\e[0Ksection_start:1790000000:hodor_diagnostics[collapsed=true]\\r\\e[0KDiagnostics
      2026-09-21T14:13:20.000Z INFO  Review instructions: bundled default
      2026-09-21T14:13:20.000Z INFO  Incremental mode: previous review at fedcba98
      2026-09-21T14:13:20.000Z INFO  Embedding diff in prompt (40000 bytes, raw: 41200 bytes)
      2026-09-21T14:13:20.000Z INFO  Review context: {"hodorThreads":{"open":3,"fixedWaiting":1,"resolved":2,"droppedByLimit":1},"humanComments":{"included":12,"droppedByBudget":2},"priorHodorReviews":3}
      2026-09-21T14:13:20.000Z INFO  Created 2 inline draft note(s)
      2026-09-21T14:13:20.000Z INFO  Posted summary note 900; collapsed 1 older summary note(s)
      2026-09-21T14:13:20.000Z INFO  Review telemetry: {"project":"acme/app","mr":42,"outcome":"reviewed","findings":2}
      \\e[0Ksection_end:1790000000:hodor_diagnostics\\r\\e[0K
      Reviewing  01234567 against main@fedcba98 (incremental since the last review)
      Diff       12 files, +340 −25 (embedded)
      Context    7 Hodor threads: 3 open, 1 fixed waiting, 2 resolved, 1 not shown
                 12 human comments included, 2 dropped (budget)
      Result     2 new findings · 1 earlier finding confirmed fixed
                 [P1] Missing authorization check  src/foo.py:42
                 [P2] Cache key ignores tenant  src/cache.py:7
      Posted     summary https://gitlab.example.com/acme/app/-/merge_requests/42#note_900 · 2 inline · 1 fixed reply
      Cost       $0.4213 · 3m 12s · 1.25M in (1.10M cached), 12.3K out · 24 turns, 61 tool calls (18 in codemode)
      Warnings   WARN Invalid submit_review payload: findings[0].priority must be 0-3
      "
    `);
  });
});
