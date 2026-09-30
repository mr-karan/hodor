import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { reviewPr, type AgentProgressEvent } from "../src/agent.js";
import { logger } from "../src/utils/logger.js";

const mocks = vi.hoisted(() => ({
  createAgentSession: vi.fn(),
  exec: vi.fn(),
  execJson: vi.fn(),
  prompts: [] as string[],
  hiddenUsage: 0,
  settingsOptions: [] as unknown[],
  extraEvents: [] as Array<Record<string, unknown>>,
  resourceLoaderOptions: [] as Array<{
    systemPromptOverride?: () => string;
    appendSystemPromptOverride?: () => string[];
  }>,
  promptResponses: [] as Array<
    | { kind: "text"; text: string }
    | { kind: "tool"; args?: Record<string, unknown> }
  >,
}));

const VALID_REVIEW_TEXT = JSON.stringify({
  findings: [],
  overall_correctness: "patch is correct",
  overall_explanation: "No production issues were found.",
});

const INVALID_REVIEW_TEXT = JSON.stringify({
  findings: [
    {
      title: "[P1] Missing null guard",
      body: "This crashes when the API returns a null payload.",
      priority: 1,
      code_location: {
        absolute_file_path: "/tmp/hodor-recovery/src/example.ts",
        line_range: { start: "1", end: 1 },
      },
    },
  ],
  overall_correctness: "patch is incorrect",
  overall_explanation: "The change introduces a crash on a valid error path.",
});

vi.mock("../src/utils/exec.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../src/utils/exec.js")>()),
  exec: mocks.exec,
  execJson: mocks.execJson,
}));

// The confined review tools are built from Pi's real tool factories; only the
// session, runtime, and loader are faked.
vi.mock("@earendil-works/pi-coding-agent", async (importOriginal) => {
  class MockResourceLoader {
    constructor(opts: {
      systemPromptOverride?: () => string;
      appendSystemPromptOverride?: () => string[];
    }) {
      mocks.resourceLoaderOptions.push(opts);
    }

    async reload(): Promise<void> {}
    getExtensions(): { extensions: unknown[]; errors: unknown[] } {
      return { extensions: [], errors: [] };
    }
    getSkills(): { skills: unknown[]; diagnostics: unknown[] } {
      return { skills: [], diagnostics: [] };
    }
  }

  return {
    ...(await importOriginal<typeof import("@earendil-works/pi-coding-agent")>()),
    createAgentSession: mocks.createAgentSession,
    DefaultResourceLoader: MockResourceLoader,
    getAgentDir: () => "/tmp/pi-agent",
    ModelRuntime: {
      create: async () => ({
        setRuntimeApiKey: vi.fn(),
        getModel: () => ({
          id: "test-model",
          name: "test-model",
          provider: "anthropic",
          api: "anthropic",
          reasoning: true,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 200000,
          maxTokens: 8192,
        }),
        getAuth: async () => ({ apiKey: "test-key" }),
      }),
    },
    SessionManager: {
      inMemory: () => ({}),
    },
    SettingsManager: {
      inMemory: (options: unknown) => {
        mocks.settingsOptions.push(options);
        return {};
      },
    },
  };
});

// The review tools list tracked files with real git, so the workspace is a
// real repository.
let workspaceDir = "";

beforeAll(() => {
  workspaceDir = mkdtempSync(join(tmpdir(), "hodor-recovery-"));
  mkdirSync(join(workspaceDir, "src"));
  writeFileSync(join(workspaceDir, "src", "example.ts"), "const value = 2;\n");
  execFileSync("git", ["init", "-q"], { cwd: workspaceDir });
  execFileSync("git", ["add", "."], { cwd: workspaceDir });
});

afterAll(() => {
  rmSync(workspaceDir, { recursive: true, force: true });
});

describe("reviewPr submit_review recovery", () => {
  beforeEach(() => {
    mocks.prompts.length = 0;
    mocks.hiddenUsage = 0;
    mocks.settingsOptions.length = 0;
    mocks.extraEvents.length = 0;
    mocks.resourceLoaderOptions.length = 0;
    mocks.promptResponses = [
      { kind: "text", text: "I found no issues." },
      { kind: "tool" },
    ];
    mocks.exec.mockReset();
    mocks.execJson.mockReset();
    mocks.execJson.mockResolvedValue({});
    mocks.createAgentSession.mockReset();

    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("--show-toplevel")) {
        return { stdout: `${workspaceDir}\n`, stderr: "" };
      }
      if (args.includes("diff")) {
        return {
          stdout: [
            "diff --git a/src/example.ts b/src/example.ts",
            "index 1111111..2222222 100644",
            "--- a/src/example.ts",
            "+++ b/src/example.ts",
            "@@ -1 +1 @@",
            "-const value = 1;",
            "+const value = 2;",
          ].join("\n"),
          stderr: "",
        };
      }
      return { stdout: "", stderr: "" };
    });

    mocks.createAgentSession.mockImplementation(async (opts: {
      customTools: Array<{
        name: string;
        execute: (toolCallId: string, params: unknown) => Promise<{ content: unknown; details: unknown }>;
      }>;
    }) => {
      const { customTools } = opts;
      const messages: Array<Record<string, unknown>> = [];
      const hiddenUsage = mocks.hiddenUsage;
      const subscribers: Array<(event: Record<string, unknown>) => void> = [];

      const emit = (event: Record<string, unknown>): void => {
        for (const subscriber of subscribers) {
          subscriber(event);
        }
      };

      const getLastAssistantText = (): string => {
        const assistant = [...messages].reverse().find((msg) => msg.role === "assistant");
        const content = assistant?.content;
        if (!Array.isArray(content)) return "";
        return content
          .map((item) => {
            const block = item as { type?: string; text?: string };
            return block.type === "text" ? block.text ?? "" : "";
          })
          .join("");
      };

      return {
        session: {
          messages,
          getSessionStats: () => {
            const totals = { input: hiddenUsage, output: 0, cacheRead: 0, cacheWrite: 0, cost: hiddenUsage * 0.01 };
            for (const msg of messages) {
              const usage = msg.usage as
                | { input: number; output: number; cacheRead: number; cacheWrite: number; cost: { total: number } }
                | undefined;
              if (msg.role !== "assistant" || !usage) continue;
              totals.input += usage.input;
              totals.output += usage.output;
              totals.cacheRead += usage.cacheRead;
              totals.cacheWrite += usage.cacheWrite;
              totals.cost += usage.cost.total;
            }
            return {
              tokens: {
                input: totals.input,
                output: totals.output,
                cacheRead: totals.cacheRead,
                cacheWrite: totals.cacheWrite,
                total: totals.input + totals.output + totals.cacheRead + totals.cacheWrite,
              },
              cost: totals.cost,
            };
          },
          state: {},
          subscribe: (subscriber: (event: Record<string, unknown>) => void) => {
            subscribers.push(subscriber);
            return () => {};
          },
          dispose: vi.fn(),
          getLastAssistantText,
          prompt: vi.fn(async (prompt: string) => {
            mocks.prompts.push(prompt);
            emit({ type: "agent_start" });
            emit({ type: "turn_start" });
            for (const extra of mocks.extraEvents) emit(extra);

            const response = mocks.promptResponses[mocks.prompts.length - 1] ?? {
              kind: "text",
              text: "I found no issues.",
            };

            if (response.kind === "text") {
              messages.push({
                role: "assistant",
                stopReason: "stop",
                content: [{ type: "text", text: response.text }],
                usage: {
                  input: 1,
                  output: 1,
                  cacheRead: 0,
                  cacheWrite: 0,
                  totalTokens: 2,
                  cost: { total: 0 },
                },
              });
            } else {
              const submitReview = customTools.find((tool) => tool.name === "submit_review");
              if (!submitReview) {
                throw new Error("submit_review tool was not registered");
              }
              const result = await submitReview.execute("tool-1", response.args ?? {
                findings: [],
                overall_correctness: "patch is correct",
                overall_explanation: "No production issues were found.",
              });
              messages.push({
                role: "assistant",
                stopReason: "tool_use",
                content: [{ type: "toolCall", name: "submit_review", arguments: {} }],
                usage: {
                  input: 1,
                  output: 1,
                  cacheRead: 0,
                  cacheWrite: 0,
                  totalTokens: 2,
                  cost: { total: 0 },
                },
              });
              messages.push({
                role: "toolResult",
                toolCallId: "tool-1",
                toolName: "submit_review",
                content: result.content,
                details: result.details,
              });
            }

            emit({ type: "turn_end" });
            emit({ type: "agent_end" });
          }),
        },
      };
    });
  });

  it("passes selected instructions as the system prompt without leaking them into the user task", async () => {
    const customProfile = "# Custom profile\nReview tenant-boundary regressions.";
    const additionalInstructions = "Prioritize authorization checks.";

    await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
      reviewInstructions: customProfile,
      additionalInstructions,
    });

    const loader = mocks.resourceLoaderOptions[mocks.resourceLoaderOptions.length - 1];
    const systemPrompt = loader.systemPromptOverride?.() ?? "";

    expect(systemPrompt).toContain(customProfile);
    expect(systemPrompt).toContain(additionalInstructions);
    expect(systemPrompt.indexOf(customProfile)).toBeLessThan(systemPrompt.indexOf(additionalInstructions));
    expect(systemPrompt.indexOf(additionalInstructions)).toBeLessThan(
      systemPrompt.indexOf("<HODOR_REVIEW_PROTOCOL>"),
    );
    expect(loader.appendSystemPromptOverride?.()).toEqual([]);
    expect(mocks.prompts[0]).not.toContain(customProfile);
    expect(mocks.prompts[0]).not.toContain(additionalInstructions);
  });

  it("asks the same session to recover when the first run ends without submit_review", async () => {
    const result = await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
    });

    expect(result.review).toEqual({
      findings: [],
      overall_correctness: "patch is correct",
      overall_explanation: "No production issues were found.",
    });
    expect(mocks.prompts).toHaveLength(2);
    expect(mocks.prompts[1]).toContain("without a valid `submit_review` tool call");
  });

  it("recovers valid review JSON emitted as assistant text without retrying", async () => {
    mocks.promptResponses = [
      { kind: "text", text: `\`\`\`json\n${VALID_REVIEW_TEXT}\n\`\`\`` },
    ];

    const result = await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
    });

    expect(result.review).toEqual({
      findings: [],
      overall_correctness: "patch is correct",
      overall_explanation: "No production issues were found.",
    });
    expect(mocks.prompts).toHaveLength(1);
  });

  it("ignores schema-invalid review JSON emitted as text and retries", async () => {
    mocks.promptResponses = [
      { kind: "text", text: `\`\`\`json\n${INVALID_REVIEW_TEXT}\n\`\`\`` },
      { kind: "tool" },
    ];

    const result = await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
    });

    expect(result.review.overall_correctness).toBe("patch is correct");
    expect(mocks.prompts).toHaveLength(2);
    expect(mocks.prompts[1]).toContain("without a valid `submit_review` tool call");
  });

  it("fails with assistant diagnostics after all recovery attempts are exhausted", async () => {
    mocks.promptResponses = [
      { kind: "text", text: "I found no issues." },
      { kind: "text", text: "Still no tool." },
      { kind: "text", text: "Still no tool." },
    ];

    await expect(reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
    })).rejects.toThrow(
      /Agent did not call submit_review after 2 recovery attempt\(s\): stopReason=stop, content=\[text\], text="Still no tool\."/,
    );
    expect(mocks.prompts).toHaveLength(3);
  });

  it("reports usage from session stats, including attempts missing from session.messages", async () => {
    mocks.hiddenUsage = 100;

    const result = await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
    });

    // Two visible assistant messages contribute 1 input + 1 output each; 100 input tokens
    // and cost 1.0 come only from usage outside session.messages.
    expect(result.metrics).toMatchObject({
      inputTokens: 102,
      outputTokens: 2,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      totalTokens: 104,
      cost: 1,
    });
  });

  it("bounds SDK retry waits in the session settings", async () => {
    await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
    });

    expect(mocks.settingsOptions[0]).toMatchObject({
      compaction: { enabled: true },
      cacheWarming: "off",
      retry: { maxRetries: 3, maxAgentDelayMs: 30_000 },
    });
  });

  it("reports SDK retry and compaction events as progress events and log lines", async () => {
    mocks.promptResponses = [{ kind: "tool" }];
    mocks.extraEvents = [
      { type: "auto_retry_start", attempt: 1, maxAttempts: 3, delayMs: 2000, errorMessage: "429 overloaded" },
      { type: "auto_retry_end", success: true, attempt: 1 },
      { type: "compaction_start", reason: "threshold" },
      { type: "compaction_end", reason: "threshold", result: undefined, aborted: false, willRetry: false },
    ];
    const infoSpy = vi.spyOn(logger, "info");
    const events: AgentProgressEvent[] = [];

    await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
      onEvent: (event) => events.push(event),
    });

    expect(events.filter((e) => e.type === "retry" || e.type === "compaction")).toEqual([
      { type: "retry", phase: "start", attempt: 1, maxAttempts: 3, delayMs: 2000, reason: "429 overloaded" },
      { type: "retry", phase: "end", attempt: 1, success: true, reason: undefined },
      { type: "compaction", phase: "start", reason: "threshold" },
      { type: "compaction", phase: "end", reason: "threshold", success: true },
    ]);
    const lines = infoSpy.mock.calls.map(([msg]) => msg);
    expect(lines).toContain("Retrying LLM request (attempt 1/3) in 2000ms: 429 overloaded");
    expect(lines).toContain("Compacting context (reason: threshold)");
    infoSpy.mockRestore();
  });

  it("passes tool call ids and codemode parents through progress events", async () => {
    mocks.promptResponses = [{ kind: "tool" }];
    mocks.extraEvents = [
      { type: "tool_execution_start", toolCallId: "c1", toolName: "codemode", args: { code: "a();\nb();\n" } },
      { type: "tool_execution_start", toolCallId: "c2", parentToolCallId: "c1", toolName: "grep", args: { pattern: "has_role\\(", path: "src" } },
      { type: "tool_execution_end", toolCallId: "c2", parentToolCallId: "c1", toolName: "grep", result: { content: [] }, isError: false },
    ];
    const events: AgentProgressEvent[] = [];

    const result = await reviewPr({
      localMode: true,
      workspaceDir,
      cleanup: false,
      model: "anthropic/test-model",
      onEvent: (event) => events.push(event),
    });

    expect(events.filter((e) => e.type === "tool_start" || e.type === "tool_end")).toEqual([
      { type: "tool_start", toolName: "codemode", toolArgs: "2-line script", toolCallId: "c1" },
      { type: "tool_start", toolName: "grep", toolArgs: '"has_role\\(" in src', toolCallId: "c2", parentToolCallId: "c1" },
      { type: "tool_end", toolName: "grep", isError: false, result: "", toolCallId: "c2", parentToolCallId: "c1" },
    ]);
    expect(result.context).toBeNull();
  });

  describe("verified fixes on GitLab", () => {
    const HEAD = "0123456789abcdef0123456789abcdef01234567";
    const ENV_KEYS = ["GITLAB_CI", "CI_PROJECT_DIR", "CI_PROJECT_PATH", "CI_MERGE_REQUEST_TARGET_BRANCH_NAME"];
    const savedEnv: Record<string, string | undefined> = {};
    const inDiff = `57a2a375${"1".repeat(56)}`;
    const outsideDiff = `9f00aa11${"2".repeat(56)}`;

    function thread(id: string, fingerprint: string, title: string, path: string, extraNotes: unknown[] = []) {
      return {
        id,
        notes: [
          {
            id: 1,
            body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\n**${title}**\n\nBody.`,
            resolvable: true,
            resolved: false,
            author: { id: 7, username: "hodor-bot" },
            position: { new_path: path, new_line: 1 },
          },
          ...extraNotes,
        ],
      };
    }

    beforeEach(() => {
      for (const key of ENV_KEYS) savedEnv[key] = process.env[key];
      process.env.GITLAB_CI = "true";
      process.env.CI_PROJECT_DIR = workspaceDir;
      process.env.CI_PROJECT_PATH = "acme/app";
      process.env.CI_MERGE_REQUEST_TARGET_BRANCH_NAME = "main";

      mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) =>
        args.includes("user") ? { id: 7, username: "hodor-bot" } : { title: "Fix archive" },
      );
      const fallback = mocks.exec.getMockImplementation();
      mocks.exec.mockImplementation(async (cmd: string, args: string[], opts?: unknown) => {
        if (args.includes("get-url")) return { stdout: "https://gitlab.example.com/acme/app.git\n", stderr: "" };
        if (args.includes("merge-base")) return { stdout: `${"a".repeat(40)}\n`, stderr: "" };
        if (args.includes("rev-parse") && args.includes("HEAD")) return { stdout: `${HEAD}\n`, stderr: "" };
        if (args.some((arg) => arg.includes("/discussions?"))) {
          return {
            stdout: JSON.stringify([
              thread("in-diff", inDiff, "[P2] Keep the value in range", "src/example.ts", [
                { id: 2, body: "not a bug", author: { id: 8, username: "alice" } },
              ]),
              thread("outside", outsideDiff, "[P3] Rename the helper", "src/other.ts"),
            ]),
            stderr: "",
          };
        }
        if (args.some((arg) => arg.includes("/notes"))) return { stdout: "[]", stderr: "" };
        if (!fallback) throw new Error("exec fallback missing");
        return fallback(cmd, args, opts);
      });
    });

    afterEach(() => {
      for (const key of ENV_KEYS) {
        const value = savedEnv[key];
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    });

    it("shows open threads with ids and keeps only verified resolved_findings", async () => {
      mocks.promptResponses = [{
        kind: "tool",
        args: {
          findings: [],
          overall_correctness: "patch is correct",
          overall_explanation: "The earlier range issue is fixed.",
          resolved_findings: ["57a2a375", "9f00aa11", "deadbeef"],
        },
      }];

      const result = await reviewPr({
        prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
        cleanup: false,
        model: "anthropic/test-model",
      });

      expect(mocks.prompts[0]).toContain("## Hodor Finding Threads");
      expect(mocks.prompts[0]).toContain("- 57a2a375 [P2] Keep the value in range (src/example.ts): open\n  - @alice: not a bug");
      // Its file is not in the diff, so the model gets no id for it.
      expect(mocks.prompts[0]).toContain("- [P3] Rename the helper (src/other.ts): open");
      expect(result.review.resolved_findings).toEqual(["57a2a375"]);
      expect(result.cacheMarker).not.toBeNull();
    });

    it("records the context manifest and shows in-thread replies only with their thread", async () => {
      const fallback = mocks.exec.getMockImplementation();
      mocks.exec.mockImplementation(async (cmd: string, args: string[], opts?: unknown) => {
        if (args.some((arg) => arg.endsWith("/notes"))) {
          return {
            stdout: JSON.stringify([
              { id: 2, body: "not a bug", author: { id: 8, username: "alice" }, created_at: "2026-09-02T10:00:00Z" },
              { id: 10, body: "false positive", author: { id: 9, username: "bob" }, created_at: "2026-09-03T10:00:00Z" },
              { id: 11, body: "lgtm", author: { id: 9, username: "bob" }, created_at: "2026-09-03T11:00:00Z" },
            ]),
            stderr: "",
          };
        }
        if (!fallback) throw new Error("exec fallback missing");
        return fallback(cmd, args, opts);
      });
      mocks.promptResponses = [{ kind: "tool" }];

      const result = await reviewPr({
        prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
        cleanup: false,
        model: "anthropic/test-model",
      });

      expect(mocks.prompts[0]).toContain("## Existing Human MR Notes");
      expect(mocks.prompts[0]).toContain("@bob:\n  false positive");
      expect(mocks.prompts[0].split("not a bug")).toHaveLength(2);
      expect(result.context).toEqual({
        hodorThreads: { open: 2, fixedWaiting: 0, resolved: 0, droppedByLimit: 0 },
        humanComments: { included: 1, droppedByBudget: 0 },
        priorHodorReviews: 0,
      });
      expect(result.range).toEqual({ headSha: HEAD, targetBranch: "main", baseSha: "a".repeat(40) });
      expect(result.metrics.diffEmbedded).toBe(true);
    });

    it("drops an id whose finding the review reports again", async () => {
      const { getFindingFingerprint } = await import("../src/review-state.js");
      const finding = {
        title: "[P2] Keep the value in range",
        body: "Still out of range.",
        priority: 2,
        code_location: { absolute_file_path: join(workspaceDir, "src", "example.ts"), line_range: { start: 1, end: 1 } },
      };
      const fingerprint = getFindingFingerprint(finding, workspaceDir);
      const fallback = mocks.exec.getMockImplementation();
      mocks.exec.mockImplementation(async (cmd: string, args: string[], opts?: unknown) => {
        if (args.some((arg) => arg.includes("/discussions?"))) {
          return {
            stdout: JSON.stringify([thread("same", fingerprint, finding.title, "src/example.ts")]),
            stderr: "",
          };
        }
        if (!fallback) throw new Error("exec fallback missing");
        return fallback(cmd, args, opts);
      });
      mocks.promptResponses = [{
        kind: "tool",
        args: {
          findings: [finding],
          overall_correctness: "patch is incorrect",
          overall_explanation: "The range issue remains.",
          resolved_findings: [fingerprint.slice(0, 8)],
        },
      }];

      const result = await reviewPr({
        prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
        cleanup: false,
        model: "anthropic/test-model",
      });

      expect(mocks.prompts[0]).toContain(`- ${fingerprint.slice(0, 8)} [P2] Keep the value in range`);
      expect(result.review.findings).toHaveLength(1);
      expect(result.review).not.toHaveProperty("resolved_findings");
    });
  });
});
