import { createHash } from "node:crypto";
import { reviewCheckName, type ReviewKey } from "./review-key.js";
import type { ReviewStateFinding, ReviewPriority } from "./types.js";

const PRIORITY_TO_SEVERITY: Record<ReviewPriority, string> = {
  0: "critical",
  1: "major",
  2: "minor",
  3: "info",
};

/**
 * GitLab merges the Code Quality reports of a pipeline's jobs by fingerprint,
 * so a keyed job scopes its fingerprints to keep both jobs' issues.
 */
function reportFingerprint(fingerprint: string, reviewKey: ReviewKey | null): string {
  return reviewKey === null
    ? fingerprint
    : createHash("sha256").update(`${reviewKey}\n${fingerprint}`).digest("hex");
}

export function formatCodeQualityReport(
  findings: ReviewStateFinding[],
  reviewKey: ReviewKey | null = null,
): string {
  const issues = findings.map((finding) => {
    if (!finding.filePath || !finding.lineRange) {
      throw new Error(`Cannot report finding without a GitLab location: ${finding.title}`);
    }

    return {
      type: "issue",
      check_name: `${reviewCheckName(reviewKey)}/P${finding.priority}`,
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
      fingerprint: reportFingerprint(finding.fingerprint, reviewKey),
    };
  });
  return JSON.stringify(issues, null, 2);
}
