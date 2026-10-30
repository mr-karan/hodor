import { describe, expect, it } from "vitest";
import { parseEvalCase, scoreReview, summarizeRuns } from "../src/evaluation.js";
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

describe("eval case parsing", () => {
  it("reads inline cases and repo cases", () => {
    expect(
      parseEvalCase(
        { id: "a", description: "d", baseFiles: {}, changedFiles: { "x.go": "package x" }, expectedFindings: [] },
        "a.json",
      ),
    ).toMatchObject({ kind: "inline", id: "a" });
    expect(
      parseEvalCase({ id: "b", description: "d", repo: "repos/x", base: "abc", head: "def", expectedFindings: [] }, "b.json"),
    ).toMatchObject({ kind: "repo", repo: "repos/x", base: "abc", head: "def" });
  });

  it("rejects malformed expected findings", () => {
    expect(() =>
      parseEvalCase(
        {
          id: "c",
          description: "d",
          baseFiles: {},
          changedFiles: {},
          expectedFindings: [{ path: "x.go", keywords: [], maximumPriority: 1 }],
        },
        "c.json",
      ),
    ).toThrow(/keywords/);
  });
});

describe("eval run summaries", () => {
  it("counts a failed review's expected findings as missed and reports per-trial spread", () => {
    const score = (matched: number, expected: number, falsePositives: number) => ({
      expected,
      matched,
      missed: [],
      falsePositives,
      recall: expected === 0 ? 1 : matched / expected,
    });
    const summary = summarizeRuns(
      [
        { caseId: "a", trial: 1, score: score(1, 1, 0), error: null, cost: 0.5, durationSeconds: 10 },
        { caseId: "b", trial: 1, score: score(0, 0, 2), error: null, cost: 0.25, durationSeconds: 30 },
        { caseId: "a", trial: 2, score: null, error: "timeout", cost: 0, durationSeconds: 20 },
        { caseId: "b", trial: 2, score: score(0, 0, 0), error: null, cost: 0.25, durationSeconds: 40 },
      ],
      new Map([
        ["a", 1],
        ["b", 0],
      ]),
    );
    expect(summary).toMatchObject({
      reviews: 4,
      failedReviews: 1,
      trials: 2,
      expected: 2,
      matched: 1,
      falsePositives: 2,
      recall: 0.5,
      recallByTrial: [1, 0],
      falsePositivesByTrial: [2, 0],
      cost: 1,
      medianDurationSeconds: 25,
    });
  });
});
