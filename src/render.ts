/**
 * Render structured review output into clean markdown for PR/MR comments.
 */

import type { ReviewFinding, ReviewOutput, ReviewStateFinding } from "./types.js";

export const HODOR_REVIEW_MARKER = "<!-- hodor-review -->";
export const HODOR_SUMMARY_MARKER = "<!-- hodor:summary:v1 -->";

/**
 * Render a ReviewOutput into clean markdown for posting as a PR/MR comment.
 */
export function renderMarkdown(review: ReviewOutput): string {
  const lines: string[] = [HODOR_REVIEW_MARKER];

  // Group findings by priority
  const critical: ReviewFinding[] = []; // P0, P1
  const important: ReviewFinding[] = []; // P2
  const minor: ReviewFinding[] = []; // P3

  for (const f of review.findings) {
    const p = f.priority;
    if (p <= 1) critical.push(f);
    else if (p === 2) important.push(f);
    else minor.push(f);
  }

  lines.push("### Issues Found");
  lines.push("");

  if (review.findings.length === 0) {
    lines.push("No issues found.");
    lines.push("");
  }

  if (critical.length > 0) {
    lines.push("**Critical (P0/P1)**");
    for (const f of critical) {
      lines.push(formatFinding(f));
    }
    lines.push("");
  }

  if (important.length > 0) {
    lines.push("**Important (P2)**");
    for (const f of important) {
      lines.push(formatFinding(f));
    }
    lines.push("");
  }

  if (minor.length > 0) {
    lines.push("**Minor (P3)**");
    for (const f of minor) {
      lines.push(formatFinding(f));
    }
    lines.push("");
  }

  // Summary
  lines.push("### Summary");
  lines.push(
    `Total issues: ${critical.length} critical, ${important.length} important, ${minor.length} minor.`,
  );
  lines.push("");

  // Overall verdict
  lines.push("### Overall Verdict");
  const isCorrect = review.overall_correctness === "patch is correct";
  lines.push(
    `**Status**: ${isCorrect ? "Patch is correct" : "Patch has blocking issues"}`,
  );
  lines.push("");
  if (review.overall_explanation) {
    lines.push(`**Explanation**: ${review.overall_explanation}`);
  }

  return lines.join("\n").trimEnd() + "\n";
}

export function renderSummaryMarkdown(
  review: ReviewOutput,
  options: {
    openFindings?: ReviewStateFinding[];
    fallbackFindings?: ReviewFinding[];
    fallbackHeading?: string;
    inlineCreated?: number;
    inlineDeduplicated?: number;
    reviewMode?: string;
  } = {},
): string {
  const lines: string[] = [HODOR_REVIEW_MARKER, HODOR_SUMMARY_MARKER];
  lines.push("", "### Hodor review");
  const openFindings = options.openFindings ?? review.findings;
  const fallbackFindings = options.fallbackFindings ?? review.findings;
  const counts = { blocking: 0, important: 0, minor: 0 };
  for (const finding of openFindings) {
    if (finding.priority <= 1) counts.blocking++;
    else if (finding.priority === 2) counts.important++;
    else counts.minor++;
  }

  const totalOpen = counts.blocking + counts.important + counts.minor;
  lines.push(
    "",
    "| Open findings | Count |",
    "| --- | ---: |",
    `| Critical (P0/P1) | ${counts.blocking} |`,
    `| Important (P2) | ${counts.important} |`,
    `| Minor (P3) | ${counts.minor} |`,
    "",
  );

  const verdict =
    totalOpen === 0
      ? "No open findings"
      : counts.blocking > 0
        ? "Blocking findings remain"
        : "Non-blocking findings remain";
  lines.push(`**Overall verdict:** ${verdict}`);

  const scope = options.reviewMode
    ? `${options.reviewMode[0].toUpperCase()}${options.reviewMode.slice(1)} review`
    : "Latest review";
  if (review.overall_explanation) {
    lines.push("");
    lines.push(`**${scope}:** ${review.overall_explanation}`);
  }

  const inlineCreated = options.inlineCreated ?? 0;
  const inlineDeduplicated = options.inlineDeduplicated ?? 0;
  if (inlineCreated > 0 || inlineDeduplicated > 0) {
    const delivery = [
      inlineCreated > 0 ? `${inlineCreated} new` : null,
      inlineDeduplicated > 0 ? `${inlineDeduplicated} already open` : null,
    ].filter((item): item is string => item !== null);
    lines.push("");
    lines.push(`**Inline comments:** ${delivery.join(" · ")}`);
  }

  if (fallbackFindings.length > 0) {
    lines.push(
      "",
      `### ${options.fallbackHeading ?? "Findings"}`,
      "",
      "| Finding | Location | Priority |",
      "| --- | --- | ---: |",
    );
    for (const finding of fallbackFindings) {
      const title = escapeTableCell(finding.title);
      const body = escapeTableCell(finding.body);
      const location = escapeTableCell(formatLocation(finding.code_location));
      lines.push(
        `| **${title}**<br>${body} | \`${location}\` | P${finding.priority} |`,
      );
    }
  }

  return lines.join("\n").trimEnd() + "\n";
}

function formatFinding(f: ReviewFinding): string {
  const loc = ` (\`${formatLocation(f.code_location)}\`)`;
  const title = `- **${f.title}**${loc}`;
  const body = `  - ${f.body}`;
  return `${title}\n${body}`;
}

function escapeTableCell(value: string): string {
  return value.replace(/\r?\n/g, " ").replace(/\|/g, "\\|");
}

function formatLocation(loc: {
  absolute_file_path: string;
  line_range: { start: number; end: number };
}): string {
  // Strip common workspace prefixes to get a clean relative path
  let filePath = loc.absolute_file_path;

  // GitLab CI: /builds/owner/repo/src/file.ts → src/file.ts
  const buildsMatch = filePath.match(/\/builds\/[^/]+\/[^/]+\/(.+)/);
  if (buildsMatch) {
    filePath = buildsMatch[1];
  }
  // GitHub Actions / generic workspace
  else if (filePath.includes("/workspace/")) {
    filePath = filePath.slice(filePath.indexOf("/workspace/") + "/workspace/".length);
  }
  // Temp review dirs: /tmp/hodor-review-<id>/src/file.ts → src/file.ts
  else {
    filePath = filePath.replace(/^.*\/hodor-review-[^/]+\//, "");
  }

  const { start, end } = loc.line_range;
  return start === end ? `${filePath}:${start}` : `${filePath}:${start}-${end}`;
}
