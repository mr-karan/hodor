import { describe, it, expect } from "vitest";
import { formatCodeQualityReport } from "../src/codequality.js";
import type { ReviewOutput, ReviewFinding } from "../src/types.js";

function makeFinding(
  title: string,
  priority: 0 | 1 | 2 | 3,
  path = "/workspace/src/foo.ts",
  line = 42,
): ReviewFinding {
  return {
    title,
    body: "Test body",
    priority,
    code_location: {
      absolute_file_path: path,
      line_range: { start: line, end: line },
    },
  };
}

describe("formatCodeQualityReport", () => {
  it("returns empty array for no findings", () => {
    const review: ReviewOutput = {
      findings: [],
      overall_correctness: "patch is correct",
      overall_explanation: "Clean.",
    };
    expect(formatCodeQualityReport(review)).toBe("[]");
  });

  it("maps priority to severity correctly", () => {
    const review: ReviewOutput = {
      findings: [
        makeFinding("[P0] Critical bug", 0),
        makeFinding("[P1] High bug", 1),
        makeFinding("[P2] Medium issue", 2),
        makeFinding("[P3] Low nit", 3),
      ],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Issues found.",
    };
    const issues = JSON.parse(formatCodeQualityReport(review));
    expect(issues).toHaveLength(4);
    expect(issues[0].severity).toBe("critical");
    expect(issues[1].severity).toBe("major");
    expect(issues[2].severity).toBe("minor");
    expect(issues[3].severity).toBe("info");
  });

  it("strips workspace prefix from paths", () => {
    const review: ReviewOutput = {
      findings: [makeFinding("[P1] Bug", 1, "/workspace/src/auth.ts")],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Bug.",
    };
    const issues = JSON.parse(formatCodeQualityReport(review, "/workspace"));
    expect(issues[0].location.path).toBe("src/auth.ts");
  });

  it("strips common workspace paths without an explicit prefix", () => {
    const review: ReviewOutput = {
      findings: [
        makeFinding("[P1] GitLab CI path", 1, "/builds/acme/app/src/api.ts"),
        makeFinding("[P2] Temp workspace path", 2, "/tmp/hodor-review-abc123/src/foo.ts"),
      ],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Bug.",
    };
    const issues = JSON.parse(formatCodeQualityReport(review));
    expect(issues[0].location.path).toBe("src/api.ts");
    expect(issues[1].location.path).toBe("src/foo.ts");
  });

  it("generates deterministic fingerprints", () => {
    const review: ReviewOutput = {
      findings: [makeFinding("[P1] Bug", 1)],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Bug.",
    };
    const result1 = JSON.parse(formatCodeQualityReport(review));
    const result2 = JSON.parse(formatCodeQualityReport(review));
    expect(result1[0].fingerprint).toBe(result2[0].fingerprint);
    expect(result1[0].fingerprint).toMatch(/^[a-f0-9]{32}$/);
  });

  it("uses CodeClimate format", () => {
    const review: ReviewOutput = {
      findings: [
        makeFinding("[P2] Missing validation", 2, "/builds/group/repo/src/api.ts", 100),
      ],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Bug.",
    };
    const issues = JSON.parse(formatCodeQualityReport(review));
    const issue = issues[0];
    expect(issue.type).toBe("issue");
    expect(issue.check_name).toBe("hodor/P2");
    expect(issue.categories).toEqual(["Bug Risk"]);
    expect(issue.location.lines.begin).toBe(100);
  });
});
