import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  getApiKey,
  getDefaultReasoningEffortForModel,
  mapReasoningEffort,
  parseModelString,
  selectReasoningEffort,
} from "../src/model.js";

describe("parseModelString", () => {
  it.each([
    ["anthropic/claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"],
    ["openai/gpt-5-2025-08-07", "openai", "gpt-5-2025-08-07"],
    ["openrouter/moonshotai/kimi-k2.6", "openrouter", "moonshotai/kimi-k2.6"],
    ["mistral/mistral-large-latest", "mistral", "mistral-large-latest"],
    ["google/gemini-2.5-pro", "google", "gemini-2.5-pro"],
    ["bedrock/anthropic.claude-opus-4-6-v1", "amazon-bedrock", "anthropic.claude-opus-4-6-v1"],
    ["bedrock/converse/global.anthropic.claude-opus-5", "amazon-bedrock", "global.anthropic.claude-opus-5"],
    ["bedrock/converse/arn:aws:bedrock:ap-south-1:123:inference-profile/xyz", "amazon-bedrock", "arn:aws:bedrock:ap-south-1:123:inference-profile/xyz"],
    ["claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"],
    ["gpt-5", "openai", "gpt-5"],
    ["o3-mini", "openai", "o3-mini"],
  ] as const)("parses %s", (input, expectedProvider, expectedModelId) => {
    const result = parseModelString(input);
    expect(result.provider).toBe(expectedProvider);
    expect(result.modelId).toBe(expectedModelId);
  });

  it("throws on empty string", () => {
    expect(() => parseModelString("")).toThrow();
  });

  it("throws on unknown provider prefixes", () => {
    expect(() => parseModelString("unknown/foo")).toThrow(/Unsupported provider/);
  });
});

describe("mapReasoningEffort", () => {
  it.each([
    [undefined, undefined],
    ["minimal", "minimal"],
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["xhigh", "xhigh"],
    ["max", "max"],
  ] as const)("maps %s to %s", (input, expected) => {
    expect(mapReasoningEffort(input as string | undefined)).toBe(expected);
  });
});

describe("getDefaultReasoningEffortForModel", () => {
  it.each([
    ["global.anthropic.claude-opus-4-7", "Claude Opus 4.7"],
    ["global.anthropic.claude-opus-4-8", "Claude Opus 4.8"],
    ["global.anthropic.claude-opus-5", "Claude Opus 5"],
  ])("defaults %s to xhigh", (id, name) => {
    expect(getDefaultReasoningEffortForModel({ id, name })).toBe("xhigh");
  });

  it("does not default other Opus versions to xhigh", () => {
    expect(
      getDefaultReasoningEffortForModel({
        id: "global.anthropic.claude-opus-4-6-v1",
        name: "Claude Opus 4.6",
      }),
    ).toBeUndefined();
  });
});

describe("selectReasoningEffort", () => {
  const smallDiff = "diff --git a/src/a.ts b/src/a.ts\n+const enabled = true;\n";
  const stats = { files: 1, additions: 1, deletions: 0, bytes: smallDiff.length };

  it("uses high for routine incremental Opus reviews", () => {
    expect(selectReasoningEffort({
      modelDefault: "xhigh",
      mode: "incremental",
      diff: smallDiff,
      stats,
    })).toBe("high");
  });

  it("retains xhigh for risky diffs and explicit full reviews", () => {
    expect(selectReasoningEffort({
      modelDefault: "xhigh",
      mode: "snapshot",
      diff: "diff --git a/auth/token.ts b/auth/token.ts\n+authorize(token);\n",
      stats,
    })).toBe("xhigh");
    expect(selectReasoningEffort({
      modelDefault: "xhigh",
      mode: "full",
      forcedFull: true,
      diff: smallDiff,
      stats,
    })).toBe("xhigh");
  });

  it("always honors an explicit setting", () => {
    expect(selectReasoningEffort({
      requested: "medium",
      modelDefault: "xhigh",
      mode: "full",
      forcedFull: true,
    })).toBe("medium");
  });
});

describe("getApiKey", () => {
  beforeEach(() => {
    delete process.env.LLM_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.OPENAI_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.MISTRAL_API_KEY;
    delete process.env.GEMINI_API_KEY;
    delete process.env.AWS_ACCESS_KEY_ID;
    delete process.env.AWS_SECRET_ACCESS_KEY;
    delete process.env.AWS_REGION_NAME;
    delete process.env.AWS_PROFILE;
  });

  it("prefers LLM_API_KEY override", () => {
    process.env.LLM_API_KEY = "sk-universal";
    process.env.OPENAI_API_KEY = "sk-openai";
    process.env.ANTHROPIC_API_KEY = "sk-anthropic";
    expect(getApiKey("openai/gpt-4o")).toBe("sk-universal");
  });

  it("prefers OpenAI key for OpenAI models", () => {
    process.env.OPENAI_API_KEY = "sk-openai";
    process.env.ANTHROPIC_API_KEY = "sk-anthropic";
    expect(getApiKey("openai/gpt-4o")).toBe("sk-openai");
  });

  it("prefers Anthropic key for Anthropic models", () => {
    process.env.OPENAI_API_KEY = "sk-openai";
    process.env.ANTHROPIC_API_KEY = "sk-anthropic";
    expect(getApiKey("anthropic/claude-sonnet-4-5")).toBe("sk-anthropic");
  });

  it("prefers OpenRouter key for OpenRouter models", () => {
    process.env.OPENAI_API_KEY = "sk-openai";
    process.env.OPENROUTER_API_KEY = "sk-or";
    expect(getApiKey("openrouter/moonshotai/kimi-k2.6")).toBe("sk-or");
  });

  it("uses pi-ai provider env vars for registry-backed providers", () => {
    process.env.MISTRAL_API_KEY = "sk-mistral";
    expect(getApiKey("mistral/mistral-large-latest")).toBe("sk-mistral");
  });

  it("returns null for bedrock", () => {
    expect(getApiKey("bedrock/anthropic.claude-opus-4-6-v1")).toBeNull();
  });

  it("throws when no key available", () => {
    expect(() => getApiKey("openai/gpt-4o")).toThrow();
  });

  it("throws without a model instead of guessing a provider", () => {
    process.env.OPENAI_API_KEY = "sk-openai";
    process.env.ANTHROPIC_API_KEY = "sk-anthropic";
    expect(() => getApiKey()).toThrow();
  });
});
