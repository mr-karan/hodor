import { describe, expect, it } from "vitest";
import {
  buildReviewCacheMarker,
  findCachedReview,
  getReviewCacheKey,
} from "../src/review-cache.js";
import type { ReviewOutput } from "../src/types.js";

const review: ReviewOutput = {
  findings: [{
    title: "[P1] Preserve the guard",
    body: "Removing this guard lets a null payload reach the decoder.",
    priority: 1,
    code_location: {
      absolute_file_path: "/builds/private/team/widget/src/api.ts",
      line_range: { start: 12, end: 13 },
    },
    existing_code: "decode(payload)",
  }],
  overall_correctness: "patch is incorrect",
  overall_explanation: "The null guard is required.",
};

describe("review cache", () => {
  it("round-trips a validated review without retaining workspace paths", () => {
    const key = getReviewCacheKey({
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      reviewInstructions: "default review profile",
    });
    const marker = buildReviewCacheMarker(key, review, "/builds/private/team/widget");
    const cached = findCachedReview([{
      body: `<!-- hodor:sha:${"a".repeat(40)} -->\n${marker}\n<!-- hodor-review -->`,
      created_at: "2026-07-16T00:00:00Z",
    }], key);

    expect(marker).not.toContain("private");
    expect(cached?.findings[0].code_location.absolute_file_path).toBe("/workspace/src/api.ts");
    expect(cached?.overall_correctness).toBe("patch is incorrect");
  });

  it("does not reuse a result with a different review identity", () => {
    const oldKey = getReviewCacheKey({
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      reviewInstructions: "default review profile",
    });
    const newKey = getReviewCacheKey({
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      requestedReasoningEffort: "high",
      reviewInstructions: "default review profile",
    });
    const marker = buildReviewCacheMarker(oldKey, review);

    expect(findCachedReview([{ body: `${marker}\n<!-- hodor-review -->` }], newKey)).toBeNull();
  });

  it("changes cache identity when the effective profile or additional instructions change", () => {
    const base = {
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      reviewInstructions: "Review authentication changes.",
    };

    const sameContent = getReviewCacheKey(base);
    const changedProfile = getReviewCacheKey({
      ...base,
      reviewInstructions: "Review authorization changes.",
    });
    const changedAdditionalInstructions = getReviewCacheKey({
      ...base,
      additionalInstructions: "Prioritize tenant isolation.",
    });

    expect(getReviewCacheKey({ ...base })).toBe(sameContent);
    expect(changedProfile).not.toBe(sameContent);
    expect(changedAdditionalInstructions).not.toBe(sameContent);
  });

  it("ignores malformed cache markers", () => {
    expect(findCachedReview([{
      body: "<!-- hodor:cache:v1:not-valid-gzip -->\n<!-- hodor-review -->",
    }], "key")).toBeNull();
  });
});
