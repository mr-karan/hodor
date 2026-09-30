import { describe, it, expect } from "vitest";
import { validateToolArguments } from "@earendil-works/pi-ai";
import { parseReviewFromAssistantText } from "../src/review-recovery.js";
import { SUBMIT_REVIEW_SCHEMA, validateReviewOutput } from "../src/review.js";
import type { ReviewOutput } from "../src/types.js";

function finding(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    title: "[P1] Missing null guard",
    body: "This crashes when the API returns a null payload.",
    priority: 1,
    code_location: {
      absolute_file_path: "/workspace/src/api.ts",
      line_range: { start: 12, end: 14 },
    },
    ...overrides,
  };
}

function payload(findingOverrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    findings: [finding(findingOverrides)],
    overall_correctness: "patch is incorrect",
    overall_explanation: "The change introduces a crash on a valid error path.",
  };
}

function viaText(value: unknown): ReviewOutput | null {
  return parseReviewFromAssistantText(JSON.stringify(value));
}

// Mirrors the tool path: Pi's validateToolArguments, then hodor checks.
function viaTool(value: unknown): ReviewOutput | null {
  try {
    const args = validateToolArguments(
      { name: "submit_review", parameters: SUBMIT_REVIEW_SCHEMA } as never,
      { name: "submit_review", arguments: value } as never,
    );
    return validateReviewOutput(args as ReviewOutput);
  } catch {
    return null;
  }
}

describe("submit_review text fallback agrees with the tool path", () => {
  const accepted: Array<[string, Record<string, unknown>]> = [
    ["optional fields omitted", payload()],
    ["optional fields set", payload({ existing_code: "a()", suggestion: "b()" })],
    ["optional fields null", payload({ existing_code: null, suggestion: null })],
    ["only suggestion null", payload({ suggestion: null })],
    ["resolved_findings set", { ...payload(), resolved_findings: ["57a2a375"] }],
    ["resolved_findings null", { ...payload(), resolved_findings: null }],
    ["resolved_findings as one string", { ...payload(), resolved_findings: "57a2a375" }],
  ];
  const rejected: Array<[string, unknown]> = [
    ["missing required title", payload({ title: undefined })],
    ["missing code_location", payload({ code_location: undefined })],
    ["missing overall_explanation", { ...payload(), overall_explanation: undefined }],
    ["priority out of range", payload({ priority: 4 })],
    ["priority does not match title tag", payload({ priority: 2 })],
    ["priority null", payload({ priority: null })],
    ["title is an object", payload({ title: { text: "x" } })],
    ["findings is not an array", { ...payload(), findings: "none" }],
    ["unknown finding property", payload({ extra: true })],
    ["empty suggestion", payload({ suggestion: "" })],
    ["empty resolved_findings id", { ...payload(), resolved_findings: [""] }],
    ["resolved_findings is an object", { ...payload(), resolved_findings: { id: "57a2a375" } }],
    ["relative path", payload({
      code_location: { absolute_file_path: "src/api.ts", line_range: { start: 1, end: 2 } },
    })],
  ];

  it.each(accepted)("accepts %s on both paths", (_name, value) => {
    const text = viaText(value);
    expect(text).not.toBeNull();
    expect(viaTool(value)).toEqual(text);
  });

  it.each(rejected)("rejects %s on both paths", (_name, value) => {
    expect(viaText(value)).toBeNull();
    expect(viaTool(value)).toBeNull();
  });

  it("drops null optional fields instead of keeping them", () => {
    const review = viaText(payload({ existing_code: null, suggestion: null }));
    expect(review?.findings[0]).not.toHaveProperty("suggestion");
    expect(review?.findings[0]).not.toHaveProperty("existing_code");
  });

  it("keeps resolved_findings ids from assistant text", () => {
    expect(viaText({ ...payload(), resolved_findings: ["57a2a375"] })?.resolved_findings).toEqual(["57a2a375"]);
    expect(viaText({ ...payload(), resolved_findings: null })).not.toHaveProperty("resolved_findings");
  });
});
