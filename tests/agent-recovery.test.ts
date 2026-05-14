import { beforeEach, describe, expect, it, vi } from "vitest";
import { reviewPr } from "../src/agent.js";

const mocks = vi.hoisted(() => ({
  createAgentSession: vi.fn(),
  exec: vi.fn(),
  prompts: [] as string[],
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
}));

vi.mock("@earendil-works/pi-coding-agent", () => {
  class MockResourceLoader {
    async reload(): Promise<void> {}
    getSkills(): { skills: unknown[]; diagnostics: unknown[] } {
      return { skills: [], diagnostics: [] };
    }
  }

  return {
    AuthStorage: {
      inMemory: () => ({ setRuntimeApiKey: vi.fn() }),
    },
    createAgentSession: mocks.createAgentSession,
    DefaultResourceLoader: MockResourceLoader,
    getAgentDir: () => "/tmp/pi-agent",
    ModelRegistry: {
      inMemory: () => ({
        find: () => ({
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
        getApiKeyForProvider: async () => "test-key",
      }),
    },
    SessionManager: {
      inMemory: () => ({}),
    },
    SettingsManager: {
      inMemory: () => ({}),
    },
  };
});

describe("reviewPr submit_review recovery", () => {
  beforeEach(() => {
    mocks.prompts.length = 0;
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
});
