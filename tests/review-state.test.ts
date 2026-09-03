import { describe, expect, it } from "vitest";
import type { HodorDiscussion } from "../src/gitlab.js";
import {
  getFindingFingerprint,
  mergeReviewStateFindings,
} from "../src/review-state.js";
import type { ReviewFinding } from "../src/types.js";

const currentFinding: ReviewFinding = {
  title: "[P1] Preserve authorization",
  body: "The new path skips the ownership check.",
  priority: 1,
  code_location: {
    absolute_file_path: "/workspace/src/app.ts",
    line_range: { start: 12, end: 13 },
  },
};

function discussion(
  finding: ReviewFinding,
  overrides: Partial<HodorDiscussion> = {},
): HodorDiscussion {
  const fingerprint = getFindingFingerprint(finding, "/workspace");
  return {
    discussionId: "discussion-1",
    noteId: 1,
    body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\n**${finding.title}**\n\n${finding.body}`,
    resolved: false,
    filePath: "src/app.ts",
    line: 12,
    ...overrides,
  };
}

describe("mergeReviewStateFindings", () => {
  it("normalizes current finding paths and assigns canonical fingerprints", () => {
    const [finding] = mergeReviewStateFindings(
      [currentFinding],
      [],
      "/workspace",
      { includeExisting: false },
    );

    expect(finding).toMatchObject({
      fingerprint: getFindingFingerprint(currentFinding, "/workspace"),
      filePath: "src/app.ts",
      lineRange: { start: 12, end: 13 },
    });
  });

  it("retains unresolved prior findings and excludes resolved threads", () => {
    const oldFinding: ReviewFinding = {
      ...currentFinding,
      title: "[P2] Keep the archive schema in sync",
      priority: 2,
    };
    const findings = mergeReviewStateFindings(
      [],
      [discussion(oldFinding), discussion(currentFinding, { resolved: true, noteId: 2 })],
      "/workspace",
    );

    expect(findings).toEqual([
      expect.objectContaining({
        title: oldFinding.title,
        priority: 2,
        filePath: "src/app.ts",
      }),
    ]);
  });

  it("prefers the current finding when an open thread has the same fingerprint", () => {
    const stale = discussion(currentFinding);
    stale.body = stale.body.replace(
      currentFinding.body,
      "Older explanation from the previous review.",
    );

    const findings = mergeReviewStateFindings(
      [currentFinding],
      [stale],
      "/workspace",
    );

    expect(findings).toHaveLength(1);
    expect(findings[0].body).toBe(currentFinding.body);
    expect(findings[0].lineRange).toEqual({ start: 12, end: 13 });
  });

  it("does not carry prior findings into a full review snapshot", () => {
    const findings = mergeReviewStateFindings(
      [],
      [discussion(currentFinding)],
      "/workspace",
      { includeExisting: false },
    );

    expect(findings).toEqual([]);
  });

  it("honors human resolution when an identical-head review is reused", () => {
    const findings = mergeReviewStateFindings(
      [currentFinding],
      [discussion(currentFinding, { resolved: true })],
      "/workspace",
      { suppressResolvedCurrent: true },
    );

    expect(findings).toEqual([]);
  });
});
