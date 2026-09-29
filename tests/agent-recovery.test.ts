import { beforeEach, describe, expect, it, vi } from "vitest";
import { reviewPr, type AgentProgressEvent } from "../src/agent.js";
import { logger } from "../src/utils/logger.js";

const mocks = vi.hoisted(() => ({
  createAgentSession: vi.fn(),
  exec: vi.fn(),
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
    | { kind: "tool" }
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

vi.mock("../src/utils/exec.js", () => ({
  exec: mocks.exec,
  execJson: vi.fn(async () => ({})),
  commandOnPath: vi.fn(() => true),
}));

vi.mock("@earendil-works/pi-coding-agent", () => {
  class MockResourceLoader {
    constructor(opts: {
      systemPromptOverride?: () => string;
      appendSystemPromptOverride?: () => string[];
    }) {
      mocks.resourceLoaderOptions.push(opts);
    }

    async reload(): Promise<void> {}
    getSkills(): { skills: unknown[]; diagnostics: unknown[] } {
      return { skills: [], diagnostics: [] };
    }
  }

  return {
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
    mocks.createAgentSession.mockReset();

    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("--show-toplevel")) {
        return { stdout: "/tmp/hodor-recovery\n", stderr: "" };
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
              const result = await submitReview.execute("tool-1", {
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
      workspaceDir: "/tmp/hodor-recovery",
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
});
