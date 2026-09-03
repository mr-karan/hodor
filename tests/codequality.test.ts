import { describe, expect, it } from "vitest";
import { formatCodeQualityReport } from "../src/codequality.js";
import type { ReviewPriority, ReviewStateFinding } from "../src/types.js";

function makeFinding(
  title: string,
  priority: ReviewPriority,
  fingerprint = String(priority).repeat(64),
): ReviewStateFinding {
  return {
    fingerprint,
    title,
    body: "Test body",
    priority,
    filePath: "src/foo.ts",
    lineRange: { start: 42, end: 42 },
  };
}

describe("formatCodeQualityReport", () => {
  it("returns an empty array for no findings", () => {
    expect(formatCodeQualityReport([])).toBe("[]");
  });

  it("maps priorities to Code Quality severities", () => {
    const issues = JSON.parse(
      formatCodeQualityReport([
        makeFinding("[P0] Critical bug", 0),
        makeFinding("[P1] High bug", 1),
        makeFinding("[P2] Medium issue", 2),
        makeFinding("[P3] Low nit", 3),
      ]),
    );

    expect(issues.map((issue: { severity: string }) => issue.severity)).toEqual([
      "critical",
      "major",
      "minor",
      "info",
    ]);
  });

  it("uses the canonical Hodor fingerprint and Code Quality format", () => {
    const finding = makeFinding("[P2] Missing validation", 2, "a".repeat(64));
    finding.filePath = "src/api.ts";
    finding.lineRange = { start: 100, end: 102 };

    const [issue] = JSON.parse(formatCodeQualityReport([finding]));
    expect(issue).toMatchObject({
      type: "issue",
      check_name: "hodor/P2",
      categories: ["Bug Risk"],
      fingerprint: "a".repeat(64),
      location: {
        path: "src/api.ts",
        lines: { begin: 100, end: 102 },
      },
    });
  });

  it("rejects findings without a reportable location", () => {
    const finding = makeFinding("[P1] Missing location", 1);
    delete finding.lineRange;

    expect(() => formatCodeQualityReport([finding])).toThrow(
      "Cannot report finding without a GitLab location",
    );
  });
});
