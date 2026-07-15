import { describe, expect, it } from "vitest";
import { hasBlockingFinding, parseFailOnPriority } from "../src/review-policy.js";
import type { ReviewOutput, ReviewPriority } from "../src/types.js";

function reviewWithPriorities(...priorities: ReviewPriority[]): ReviewOutput {
  return {
    findings: priorities.map((priority) => ({
      title: `[P${priority}] Finding`,
      body: "A concrete issue.",
      priority,
      code_location: {
        absolute_file_path: "/workspace/src/app.ts",
        line_range: { start: 1, end: 1 },
      },
    })),
    overall_correctness: priorities.length > 0 ? "patch is incorrect" : "patch is correct",
    overall_explanation: "Policy fixture.",
  };
}

describe("review policy", () => {
  it("parses supported thresholds", () => {
    expect(parseFailOnPriority("P0")).toBe("P0");
    expect(parseFailOnPriority("P3")).toBe("P3");
  });

  it("rejects invalid thresholds", () => {
    expect(() => parseFailOnPriority("critical")).toThrow(/P0, P1, P2, P3/);
  });

  it("fails only at or above the selected severity", () => {
    const review = reviewWithPriorities(1, 2);
    expect(hasBlockingFinding(review, "P0")).toBe(false);
    expect(hasBlockingFinding(review, "P1")).toBe(true);
    expect(hasBlockingFinding(review, "P2")).toBe(true);
  });
});
