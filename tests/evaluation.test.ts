import { describe, expect, it } from "vitest";
import { scoreReview } from "../src/evaluation.js";
import type { ReviewOutput } from "../src/types.js";

const review: ReviewOutput = {
  findings: [
    {
      title: "[P1] Guard the missing user",
      body: "The lookup can return null before this property access.",
      priority: 1,
      code_location: {
        absolute_file_path: "/tmp/eval/src/user.ts",
        line_range: { start: 4, end: 4 },
      },
    },
  ],
  overall_correctness: "patch is incorrect",
  overall_explanation: "A null dereference is possible.",
};

describe("review evaluation", () => {
  it("matches findings by path, keywords, and maximum priority", () => {
    const score = scoreReview(review, [
      { path: "src/user.ts", keywords: ["null"], maximumPriority: 1 },
    ]);

    expect(score).toMatchObject({ matched: 1, falsePositives: 0, recall: 1 });
  });

  it("reports misses and false positives independently", () => {
    const score = scoreReview(review, [
      { path: "src/auth.ts", keywords: ["authorization"], maximumPriority: 0 },
    ]);

    expect(score.missed).toHaveLength(1);
    expect(score.falsePositives).toBe(1);
    expect(score.recall).toBe(0);
  });
});
