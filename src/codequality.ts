import type { ReviewStateFinding, ReviewPriority } from "./types.js";

const PRIORITY_TO_SEVERITY: Record<ReviewPriority, string> = {
  0: "critical",
  1: "major",
  2: "minor",
  3: "info",
};

export function formatCodeQualityReport(findings: ReviewStateFinding[]): string {
  const issues = findings.map((finding) => {
    if (!finding.filePath || !finding.lineRange) {
      throw new Error(`Cannot report finding without a GitLab location: ${finding.title}`);
    }

    return {
      type: "issue",
      check_name: `hodor/P${finding.priority}`,
      description: finding.title,
      content: { body: finding.body },
      categories: ["Bug Risk"],
      severity: PRIORITY_TO_SEVERITY[finding.priority] ?? "info",
      location: {
        path: finding.filePath,
        lines: {
          begin: finding.lineRange.start,
          end: finding.lineRange.end,
        },
      },
      fingerprint: finding.fingerprint,
    };
  });
  return JSON.stringify(issues, null, 2);
}
