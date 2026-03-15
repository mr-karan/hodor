import { createHash } from "node:crypto";
import type { ReviewOutput, ReviewFinding, ReviewPriority } from "./types.js";

const PRIORITY_TO_SEVERITY: Record<ReviewPriority, string> = {
  0: "critical",
  1: "major",
  2: "minor",
  3: "info",
};

function relativizePath(absolutePath: string, workspacePrefix?: string): string {
  if (workspacePrefix) {
    const prefix = workspacePrefix.endsWith("/") ? workspacePrefix : workspacePrefix + "/";
    if (absolutePath.startsWith(prefix)) {
      return absolutePath.slice(prefix.length);
    }
  }

  // Fallback: try common workspace patterns
  const buildsMatch = absolutePath.match(/\/builds\/[^/]+\/[^/]+\/(.+)/);
  if (buildsMatch) return buildsMatch[1];
  const workspaceMatch = absolutePath.match(/\/workspace\/(.+)/);
  if (workspaceMatch) return workspaceMatch[1];
  const hodorMatch = absolutePath.replace(/^.*\/hodor-review-[^/]+\//, "");
  if (hodorMatch !== absolutePath) return hodorMatch;
  return absolutePath;
}

function fingerprint(finding: ReviewFinding, relativePath: string): string {
  const input = `${finding.title}:${relativePath}:${finding.code_location.line_range.start}`;
  return createHash("md5").update(input).digest("hex");
}

export function formatCodeQualityReport(
  review: ReviewOutput,
  workspacePrefix?: string,
): string {
  const issues = review.findings.map((finding) => {
    const relPath = relativizePath(finding.code_location.absolute_file_path, workspacePrefix);
    return {
      type: "issue",
      check_name: `hodor/P${finding.priority}`,
      description: finding.title,
      content: { body: finding.body },
      categories: ["Bug Risk"],
      severity: PRIORITY_TO_SEVERITY[finding.priority] ?? "info",
      location: {
        path: relPath,
        lines: {
          begin: finding.code_location.line_range.start,
          end: finding.code_location.line_range.end,
        },
      },
      fingerprint: fingerprint(finding, relPath),
    };
  });

  return JSON.stringify(issues, null, 2);
}
