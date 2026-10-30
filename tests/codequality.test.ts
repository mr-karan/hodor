import { describe, expect, it } from "vitest";
import { formatCodeQualityReport } from "../src/codequality.js";
import { parseReviewKey } from "../src/review-key.js";
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

  it("keeps the report byte-identical without a review key", () => {
    const finding = makeFinding("[P1] Preserve authorization", 1, "b".repeat(64));

    expect(formatCodeQualityReport([finding])).toBe(JSON.stringify([{
      type: "issue",
      check_name: "hodor/P1",
      description: "[P1] Preserve authorization",
      content: { body: "Test body" },
      categories: ["Bug Risk"],
      severity: "major",
      location: { path: "src/foo.ts", lines: { begin: 42, end: 42 } },
      fingerprint: "b".repeat(64),
    }], null, 2));
  });

  it("keeps a keyed job's issues apart from the default job's for the same finding", () => {
    const finding = makeFinding("[P1] Escape the SQL parameter", 1, "c".repeat(64));

    const [defaultIssue] = JSON.parse(formatCodeQualityReport([finding]));
    const [securityIssue] = JSON.parse(formatCodeQualityReport([finding], parseReviewKey("security")));

    expect(defaultIssue).toMatchObject({ check_name: "hodor/P1", fingerprint: "c".repeat(64) });
    expect(securityIssue.check_name).toBe("hodor/security/P1");
    expect(securityIssue.fingerprint).toMatch(/^[a-f0-9]{64}$/);
    expect(securityIssue.fingerprint).not.toBe(defaultIssue.fingerprint);
    expect(JSON.parse(formatCodeQualityReport([finding], parseReviewKey("security")))[0].fingerprint)
      .toBe(securityIssue.fingerprint);
    expect(JSON.parse(formatCodeQualityReport([finding], parseReviewKey("perf")))[0].fingerprint)
      .not.toBe(securityIssue.fingerprint);
  });

  it("rejects findings without a reportable location", () => {
    const finding = makeFinding("[P1] Missing location", 1);
    delete finding.lineRange;

    expect(() => formatCodeQualityReport([finding])).toThrow(
      "Cannot report finding without a GitLab location",
    );
  });
});
