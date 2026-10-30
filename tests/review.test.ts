import { describe, expect, test, vi } from "vitest";
import { Value } from "typebox/value";
import { createModelRuntime } from "../src/models-json.js";
import {
  SUBMIT_REVIEW_CONSTRAINED_SAMPLING,
  SUBMIT_REVIEW_SCHEMA,
  validateReviewOutput,
} from "../src/review.js";
import type { ReviewOutput } from "../src/types.js";
import { isRecord } from "../src/utils/json.js";

class PayloadCaptured extends Error {}

/** Build the real provider request for submit_review and stop before it is sent. */
async function captureSubmitReviewPayload(provider: string, modelId: string): Promise<unknown> {
  vi.stubEnv("AWS_REGION", "us-east-1");
  vi.stubEnv("AWS_ACCESS_KEY_ID", "offline-test");
  vi.stubEnv("AWS_SECRET_ACCESS_KEY", "offline-test");
  try {
    const runtime = await createModelRuntime(null);
    const model = runtime.getModel(provider, modelId);
    if (!model) throw new Error(`model ${provider}/${modelId} is not in the catalog`);
    let payload: unknown;
    const stream = runtime.streamSimple(
      model,
      {
        systemPrompt: "Review the change.",
        messages: [{ role: "user", content: "Review it.", timestamp: 0 }],
        tools: [
          {
            name: "submit_review",
            description: "Submit the review.",
            parameters: SUBMIT_REVIEW_SCHEMA,
            constrainedSampling: SUBMIT_REVIEW_CONSTRAINED_SAMPLING,
          },
        ],
      },
      {
        apiKey: "offline-test-key",
        onPayload: (params) => {
          payload = params;
          throw new PayloadCaptured();
        },
      },
    );
    await stream.result();
    if (payload === undefined) throw new Error("provider built no request payload");
    return payload;
  } finally {
    vi.unstubAllEnvs();
  }
}

function findTool(payload: unknown, path: string[], name: string): Record<string, unknown> {
  let tools: unknown = payload;
  for (const key of path) tools = isRecord(tools) ? tools[key] : undefined;
  if (!Array.isArray(tools)) throw new Error(`payload has no ${path.join(".")} list`);
  for (const tool of tools) {
    const spec = isRecord(tool) && isRecord(tool.toolSpec) ? tool.toolSpec : tool;
    if (isRecord(spec) && spec.name === name) return spec;
  }
  throw new Error(`payload does not declare ${name}`);
}

function makeReview(overrides: Partial<ReviewOutput> = {}): ReviewOutput {
  return {
    findings: [
      {
        title: "[P1] Missing null guard",
        body: "This crashes when the API returns a null payload.",
        priority: 1,
        code_location: {
          absolute_file_path: "/workspace/src/api.ts",
          line_range: { start: 12, end: 14 },
        },
      },
    ],
    overall_correctness: "patch is incorrect",
    overall_explanation: "The change introduces a crash on a valid error path.",
    ...overrides,
  };
}

describe("SUBMIT_REVIEW_SCHEMA", () => {
  test("accepts the optional existing_code field", () => {
    const review = makeReview();
    review.findings[0].existing_code = "function foo() {\n  return bar;\n}";
    expect(Value.Check(SUBMIT_REVIEW_SCHEMA, review)).toBe(true);
  });

  test("accepts an optional resolved_findings list", () => {
    expect(Value.Check(SUBMIT_REVIEW_SCHEMA, makeReview({ resolved_findings: ["57a2a375"] }))).toBe(true);
    expect(Value.Check(SUBMIT_REVIEW_SCHEMA, makeReview({ resolved_findings: [] }))).toBe(true);
  });

  test("accepts findings without existing_code (backward compatible)", () => {
    expect(Value.Check(SUBMIT_REVIEW_SCHEMA, makeReview())).toBe(true);
  });

  test("rejects a priority outside P0-P3", () => {
    const review = makeReview();
    expect(Value.Check(SUBMIT_REVIEW_SCHEMA, { ...review, findings: [{ ...review.findings[0], priority: 4 }] })).toBe(
      false,
    );
  });

  // Issue #32: Claude strict tool use rejected the whole request with a 400 when
  // submit_review carried minimum/maximum. Both Anthropic and Bedrock document
  // minLength/maxLength as unsupported too.
  test.each([
    ["anthropic", "claude-opus-5-5", ["tools"]],
    ["amazon-bedrock", "global.anthropic.claude-sonnet-4-6", ["toolConfig", "tools"]],
    ["amazon-bedrock", "global.anthropic.claude-sonnet-5-5", ["toolConfig", "tools"]],
    ["amazon-bedrock", "global.anthropic.claude-opus-5-5", ["toolConfig", "tools"]],
  ])("is sent to %s %s as a strict tool without length or numeric bounds", async (provider, modelId, path) => {
    const payload = await captureSubmitReviewPayload(provider, modelId);
    const tool = findTool(payload, path, "submit_review");
    expect(tool.strict).toBe(true);
    const schema = JSON.stringify(tool.input_schema ?? tool.inputSchema);
    for (const keyword of ["minimum", "maximum", "minLength", "maxLength"]) {
      expect(schema).not.toContain(`"${keyword}"`);
    }
  });
});

describe("validateReviewOutput", () => {
  test("accepts a valid structured review", () => {
    const review = makeReview();
    expect(validateReviewOutput(review)).toEqual(review);
  });

  test("auto-corrects findings with a wrong verdict to 'patch is incorrect'", () => {
    const review = makeReview({ overall_correctness: "patch is correct" });
    const result = validateReviewOutput(review);
    expect(result.overall_correctness).toBe("patch is incorrect");
    expect(result.findings).toEqual(review.findings);
  });

  test("auto-corrects empty findings with a wrong verdict to 'patch is correct'", () => {
    const review = makeReview({
      findings: [],
      overall_correctness: "patch is incorrect",
      overall_explanation: "No issues were found.",
    });
    const result = validateReviewOutput(review);
    expect(result.overall_correctness).toBe("patch is correct");
  });

  test("rejects mismatched title and numeric priority", () => {
    const review = makeReview({
      findings: [
        {
          title: "[P2] Missing null guard",
          body: "This crashes when the API returns a null payload.",
          priority: 1,
          code_location: {
            absolute_file_path: "/workspace/src/api.ts",
            line_range: { start: 12, end: 14 },
          },
        },
      ],
    });

    expect(() => validateReviewOutput(review)).toThrow(
      "submit_review finding 1 priority 1 does not match title tag 2",
    );
  });

  test("rejects relative file paths", () => {
    const review = makeReview({
      findings: [
        {
          title: "[P1] Missing null guard",
          body: "This crashes when the API returns a null payload.",
          priority: 1,
          code_location: {
            absolute_file_path: "src/api.ts",
            line_range: { start: 12, end: 14 },
          },
        },
      ],
    });

    expect(() => validateReviewOutput(review)).toThrow(
      "submit_review finding 1 code_location.absolute_file_path must be absolute",
    );
  });

  test("rejects inverted line ranges", () => {
    const review = makeReview({
      findings: [
        {
          title: "[P1] Missing null guard",
          body: "This crashes when the API returns a null payload.",
          priority: 1,
          code_location: {
            absolute_file_path: "/workspace/src/api.ts",
            line_range: { start: 14, end: 12 },
          },
        },
      ],
    });

    expect(() => validateReviewOutput(review)).toThrow(
      "submit_review finding 1 code_location line_range start must be <= end",
    );
  });

  test("rejects line numbers below 1", () => {
    const review = makeReview();
    const finding = review.findings[0];
    const zeroStart = makeReview({
      findings: [{ ...finding, code_location: { ...finding.code_location, line_range: { start: 0, end: 3 } } }],
    });

    expect(() => validateReviewOutput(zeroStart)).toThrow(
      "submit_review finding 1 code_location line_range start must be >= 1",
    );
  });

  test("rejects empty optional strings", () => {
    const finding = makeReview().findings[0];
    expect(() => validateReviewOutput(makeReview({ findings: [{ ...finding, existing_code: "" }] }))).toThrow(
      "submit_review finding 1 existing_code must be non-empty when present",
    );
    expect(() => validateReviewOutput(makeReview({ findings: [{ ...finding, suggestion: "" }] }))).toThrow(
      "submit_review finding 1 suggestion must be non-empty when present",
    );
    expect(() => validateReviewOutput(makeReview({ resolved_findings: ["57a2a375", ""] }))).toThrow(
      "submit_review resolved_findings entries must be non-empty",
    );
  });
});
