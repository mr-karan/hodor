import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  addOpenAiBedrockReasoning,
  buildBedrockArnModel,
  extractBedrockArnRegion,
  getApiKey,
  getDefaultReasoningEffortForModel,
  isHighRiskDiff,
  isOpenAiBedrockModel,
  mapReasoningEffort,
  parseModelString,
  qualifiesForSingleTurnReview,
  selectReasoningEffort,
  stripBedrockRegionalPrefix,
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

  it("leaves baseModelId unset for plain bedrock models", () => {
    expect(parseModelString("bedrock/converse/global.anthropic.claude-opus-5").baseModelId)
      .toBeUndefined();
  });

  it("parses the base model backing an inference profile ARN", () => {
    const arn = "arn:aws:bedrock:ap-south-1:123:application-inference-profile/abc123";
    const result = parseModelString(
      `bedrock/converse/${arn}@global.anthropic.claude-opus-5`,
    );
    expect(result.provider).toBe("amazon-bedrock");
    expect(result.modelId).toBe(arn);
    expect(result.baseModelId).toBe("global.anthropic.claude-opus-5");
  });

  it("parses the base model backing a regional inference profile id", () => {
    const result = parseModelString(
      "bedrock/in.openai.gpt-5.6-terra@openai.gpt-5.6-terra",
    );
    expect(result.modelId).toBe("in.openai.gpt-5.6-terra");
    expect(result.baseModelId).toBe("openai.gpt-5.6-terra");
  });

  it.each([
    "bedrock/converse/arn:aws:bedrock:ap-south-1:123:application-inference-profile/abc@",
    "bedrock/converse/@global.anthropic.claude-opus-5",
  ])("rejects a half-specified base model hint (%s)", (input) => {
    expect(() => parseModelString(input)).toThrow(/Invalid bedrock model/);
  });
});

describe("stripBedrockRegionalPrefix", () => {
  it.each([
    ["in.openai.gpt-5.6-terra", "openai.gpt-5.6-terra"],
    ["global.anthropic.claude-opus-5", "anthropic.claude-opus-5"],
    ["apac.amazon.nova-pro-v1:0", "amazon.nova-pro-v1:0"],
  ])("strips the prefix from %s", (modelId, expected) => {
    expect(stripBedrockRegionalPrefix(modelId)).toBe(expected);
  });

  it.each([
    "openai.gpt-5.6-sol",
    "anthropic.claude-opus-5",
    "custom.example.model",
  ])("does not strip an unrecognized prefix from %s", (modelId) => {
    expect(stripBedrockRegionalPrefix(modelId)).toBeNull();
  });
});

describe("OpenAI Bedrock reasoning", () => {
  it("recognizes registry and regional OpenAI Bedrock models", () => {
    expect(isOpenAiBedrockModel({
      provider: "amazon-bedrock",
      id: "in.openai.gpt-5.6-terra",
    })).toBe(true);
    expect(isOpenAiBedrockModel({
      provider: "amazon-bedrock",
      id: "arn:aws:bedrock:ap-south-1:123:application-inference-profile/abc",
      name: "openai.gpt-5.6-sol",
    })).toBe(true);
    expect(isOpenAiBedrockModel({
      provider: "amazon-bedrock",
      id: "global.anthropic.claude-opus-5",
    })).toBe(false);
  });

  it("adds reasoning effort without discarding other provider fields", () => {
    expect(addOpenAiBedrockReasoning({
      modelId: "global.openai.gpt-5.6-sol",
      additionalModelRequestFields: { trace: true },
    }, "medium")).toEqual({
      modelId: "global.openai.gpt-5.6-sol",
      additionalModelRequestFields: {
        trace: true,
        reasoning: { effort: "medium" },
      },
    });
  });
});

describe("buildBedrockArnModel", () => {
  const arn = "arn:aws:bedrock:ap-south-1:123:application-inference-profile/abc123";
  // Shape mirrors the pi-ai registry entry for global.anthropic.claude-opus-5.
  const baseModel = {
    id: "global.anthropic.claude-opus-5",
    name: "Claude Opus 5 (Global)",
    api: "bedrock-converse-stream",
    provider: "amazon-bedrock",
    baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
    reasoning: true,
    input: ["text"],
    cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
    contextWindow: 1_000_000,
    maxTokens: 128_000,
  } as never;

  it("derives the region from the ARN", () => {
    expect(extractBedrockArnRegion(arn)).toBe("ap-south-1");
    expect(extractBedrockArnRegion("arn:aws:bedrock")).toBe("us-east-1");
  });

  it("inherits capabilities and cost from the base model", () => {
    const model = buildBedrockArnModel({ arn, baseModel }) as Record<string, unknown>;

    expect(model.id).toBe(arn);
    expect(model.reasoning).toBe(true);
    expect(model.cost).toEqual({ input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 });
    expect(model.contextWindow).toBe(1_000_000);
    expect(model.maxTokens).toBe(128_000);
    expect(model.baseUrl).toBe("https://bedrock-runtime.ap-south-1.amazonaws.com");
  });

  // pi-ai gates prompt caching, adaptive thinking and xhigh effort on substring
  // matches over the model id/name. The ARN matches none of them, so the name
  // must carry the registry id or those features silently switch off.
  it("keeps a name that satisfies pi-ai's capability matchers", () => {
    const model = buildBedrockArnModel({ arn, baseModel }) as Record<string, unknown>;
    const candidates = [model.id, model.name]
      .filter((value): value is string => typeof value === "string")
      .flatMap((value) => {
        const lower = value.toLowerCase();
        return [lower, lower.replace(/[\s_.:]+/g, "-")];
      });

    expect(candidates.some((c) => c.includes("claude"))).toBe(true);
    expect(candidates.some((c) => c.includes("opus-5"))).toBe(true);
    expect(candidates.some((c) => c.includes("anthropic.claude") || c.includes("anthropic-claude")))
      .toBe(true);
  });

  it("keeps adaptive reasoning defaults reachable through the ARN", () => {
    const model = buildBedrockArnModel({ arn, baseModel }) as { id?: string; name?: string };
    expect(getDefaultReasoningEffortForModel(model)).toBe("xhigh");
  });

  it("falls back to a capability-free descriptor without a base model", () => {
    const model = buildBedrockArnModel({ arn }) as Record<string, unknown>;
    expect(model.reasoning).toBe(false);
    expect(model.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    expect(model.name).toBe(arn);
  });
});

describe("isHighRiskDiff", () => {
  it("flags security-sensitive paths and semantics", () => {
    expect(isHighRiskDiff("diff --git a/auth/token.ts b/auth/token.ts\n+const x = 1;\n")).toBe(true);
    expect(isHighRiskDiff("diff --git a/db/migrations/001.sql b/db/migrations/001.sql\n+ALTER TABLE t;\n")).toBe(true);
    expect(isHighRiskDiff("diff --git a/src/ui.ts b/src/ui.ts\n+const permission = check();\n")).toBe(true);
  });

  it("does not flag routine changes", () => {
    expect(isHighRiskDiff("diff --git a/src/ui.ts b/src/ui.ts\n+const label = 'Save';\n")).toBe(false);
  });

  // Regex state leaking between calls would make the predicate alternate.
  it("is stable across repeated calls", () => {
    const diff = "diff --git a/auth/token.ts b/auth/token.ts\n+const x = 1;\n";
    expect([isHighRiskDiff(diff), isHighRiskDiff(diff), isHighRiskDiff(diff)])
      .toEqual([true, true, true]);
  });
});

describe("qualifiesForSingleTurnReview", () => {
  const tinyDiff = "diff --git a/src/ui.ts b/src/ui.ts\n+const label = 'Save';\n";
  const tinyStats = { files: 1, additions: 1, deletions: 0, bytes: tinyDiff.length };

  it("accepts a tiny embedded low-risk diff", () => {
    expect(qualifiesForSingleTurnReview({ diff: tinyDiff, stats: tinyStats, embedded: true }))
      .toBe(true);
  });

  it("requires the diff to be embedded", () => {
    expect(qualifiesForSingleTurnReview({ diff: tinyDiff, stats: tinyStats, embedded: false }))
      .toBe(false);
  });

  it("rejects high-risk diffs regardless of size", () => {
    const riskyDiff = "diff --git a/auth/token.ts b/auth/token.ts\n+const t = 1;\n";
    expect(qualifiesForSingleTurnReview({
      diff: riskyDiff,
      stats: { files: 1, additions: 1, deletions: 0, bytes: riskyDiff.length },
      embedded: true,
    })).toBe(false);
  });

  it("rejects diffs above the file or line budget", () => {
    expect(qualifiesForSingleTurnReview({
      diff: tinyDiff,
      stats: { files: 6, additions: 1, deletions: 0, bytes: 100 },
      embedded: true,
    })).toBe(false);
    expect(qualifiesForSingleTurnReview({
      diff: tinyDiff,
      stats: { files: 1, additions: 40, deletions: 11, bytes: 100 },
      embedded: true,
    })).toBe(false);
  });

  it("rejects an empty or unknown diff", () => {
    expect(qualifiesForSingleTurnReview({
      diff: "",
      stats: { files: 0, additions: 0, deletions: 0, bytes: 0 },
      embedded: true,
    })).toBe(false);
    expect(qualifiesForSingleTurnReview({ diff: tinyDiff, stats: null, embedded: true }))
      .toBe(false);
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
