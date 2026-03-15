import { describe, test, expect, it } from "vitest";
import { renderMarkdown, renderSummaryMarkdown } from "../src/render.js";
import type { ReviewOutput, ReviewFinding } from "../src/types.js";

function makeFinding(title: string, priority: 0 | 1 | 2 | 3): ReviewFinding {
  return {
    title,
    body: "Test body",
    priority,
    code_location: {
      absolute_file_path: "/tmp/hodor-review-abc123/src/foo.ts",
      line_range: { start: 10, end: 15 },
    },
  };
}

describe("renderMarkdown", () => {
  test("renders empty findings", () => {
    const review: ReviewOutput = {
      findings: [],
      overall_correctness: "patch is correct",
      overall_explanation: "No issues found in the changes.",
    };
    const md = renderMarkdown(review);
    expect(md).toContain("### Issues Found");
    expect(md).toContain("No issues found.");
    expect(md).toContain("### Summary");
    expect(md).toContain("Total issues: 0 critical, 0 important, 0 minor.");
    expect(md).toContain("**Status**: Patch is correct");
  });

  test("renders findings grouped by priority", () => {
    const review: ReviewOutput = {
      findings: [
        {
          title: "[P0] SQL injection in login",
          body: "User input concatenated into query.",
          priority: 0,
          code_location: {
            absolute_file_path: "/builds/acme/app/src/db.ts",
            line_range: { start: 12, end: 15 },
          },
        },
        {
          title: "[P2] Missing index on user_id",
          body: "Full table scan on every request.",
          priority: 2,
          code_location: {
            absolute_file_path: "/builds/acme/app/src/models.ts",
            line_range: { start: 89, end: 89 },
          },
        },
        {
          title: "[P3] Magic number 42",
          body: "Should be a named constant.",
          priority: 3,
          code_location: {
            absolute_file_path: "/builds/acme/app/src/util.ts",
            line_range: { start: 7, end: 7 },
          },
        },
      ],
      overall_correctness: "patch is incorrect",
      overall_explanation: "SQL injection is a blocker.",
    };
    const md = renderMarkdown(review);
    expect(md).toContain("**Critical (P0/P1)**");
    expect(md).toContain("**Important (P2)**");
    expect(md).toContain("**Minor (P3)**");
    expect(md).toContain("Total issues: 1 critical, 1 important, 1 minor.");
    expect(md).toContain("**Status**: Patch has blocking issues");
    // Check path stripping: /builds/acme/app/src/db.ts → src/db.ts
    expect(md).toContain("`src/db.ts:12-15`");
    expect(md).toContain("`src/models.ts:89`");
  });

  it("includes hodor-review marker", () => {
    const review: ReviewOutput = {
      findings: [],
      overall_correctness: "patch is correct",
      overall_explanation: "Clean.",
    };
    expect(renderMarkdown(review)).toContain("<!-- hodor-review -->");
  });
});

describe("renderSummaryMarkdown", () => {
  it("includes hodor-review marker", () => {
    const review: ReviewOutput = {
      findings: [],
      overall_correctness: "patch is correct",
      overall_explanation: "All good.",
    };
    const result = renderSummaryMarkdown(review);
    expect(result).toContain("<!-- hodor-review -->");
  });

  it("shows correct counts by priority", () => {
    const review: ReviewOutput = {
      findings: [
        makeFinding("P0 bug", 0),
        makeFinding("P1 bug", 1),
        makeFinding("P2 issue", 2),
        makeFinding("P3 nit", 3),
      ],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Has blocking issues.",
    };
    const result = renderSummaryMarkdown(review);
    expect(result).toContain("| Critical (P0/P1) | 2 |");
    expect(result).toContain("| Important (P2) | 1 |");
    expect(result).toContain("| Minor (P3) | 1 |");
    expect(result).toContain("Patch has blocking issues");
  });

  it("includes findings table with locations", () => {
    const review: ReviewOutput = {
      findings: [makeFinding("[P1] Null check missing", 1)],
      overall_correctness: "patch is incorrect",
      overall_explanation: "Bug found.",
    };
    const result = renderSummaryMarkdown(review);
    expect(result).toContain("| Finding | Location | Priority |");
    expect(result).toContain("[P1] Null check missing");
    expect(result).toContain("P1");
  });
});
