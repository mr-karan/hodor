import { describe, it, expect, vi, beforeEach } from "vitest";
import { parseModelString, mapReasoningEffort, getApiKey } from "../src/model.js";

describe("parseModelString", () => {
  it.each([
    ["anthropic/claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"],
    ["openai/gpt-5-2025-08-07", "openai", "gpt-5-2025-08-07"],
    ["openai-codex/gpt-5.2-codex", "openai-codex", "gpt-5.2-codex"],
    ["bedrock/anthropic.claude-opus-4-6-v1", "amazon-bedrock", "anthropic.claude-opus-4-6-v1"],
    ["bedrock/converse/arn:aws:bedrock:ap-south-1:123:inference-profile/xyz", "amazon-bedrock", "arn:aws:bedrock:ap-south-1:123:inference-profile/xyz"],
    ["claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"],
    ["gpt-5", "openai", "gpt-5"],
    ["gpt-5.2-codex", "openai-codex", "gpt-5.2-codex"],
    ["o3-mini", "openai", "o3-mini"],
  ] as const)("parses %s", (input, expectedProvider, expectedModelId) => {
    const result = parseModelString(input);
    expect(result.provider).toBe(expectedProvider);
    expect(result.modelId).toBe(expectedModelId);
  });

  it("throws on empty string", () => {
    expect(() => parseModelString("")).toThrow();
  });
});

describe("mapReasoningEffort", () => {
  it.each([
    [undefined, undefined],
    ["low", "low"],
    ["medium", "medium"],
    ["high", "high"],
    ["xhigh", "xhigh"],
  ] as const)("maps %s to %s", (input, expected) => {
    expect(mapReasoningEffort(input as string | undefined)).toBe(expected);
  });
});

describe("getApiKey", () => {
  beforeEach(() => {
    delete process.env.LLM_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.OPENAI_API_KEY;
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

  it("falls back to Anthropic first without model", () => {
    process.env.OPENAI_API_KEY = "sk-openai";
    process.env.ANTHROPIC_API_KEY = "sk-anthropic";
    expect(getApiKey()).toBe("sk-anthropic");
  });

  it("returns null for bedrock", () => {
    expect(getApiKey("bedrock/anthropic.claude-opus-4-6-v1")).toBeNull();
  });

  it("does not reuse OPENAI_API_KEY for openai-codex models", () => {
    process.env.OPENAI_API_KEY = "sk-openai";
    expect(() => getApiKey("openai-codex/gpt-5.2-codex")).toThrow();
  });

  it("throws when no key available", () => {
    expect(() => getApiKey("openai/gpt-4o")).toThrow();
  });
});
