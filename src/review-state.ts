import { createHash } from "node:crypto";
import type { HodorDiscussion } from "./gitlab.js";
import type { ReviewFinding, ReviewPriority, ReviewStateFinding } from "./types.js";
import { relativizeWorkspacePath } from "./utils/path.js";

const FINDING_MARKER_RE = /<!--\s*hodor:finding:([a-f0-9]{64})\s*-->/i;
const FINDING_TITLE_RE = /^\*\*(\[P([0-3])\]\s+.+)\*\*\s*$/m;

export function getFindingFingerprint(
  finding: ReviewFinding,
  workspacePath?: string | null,
): string {
  const path = relativizeWorkspacePath(
    finding.code_location.absolute_file_path,
    workspacePath ?? undefined,
  );
  const title = finding.title.replace(/^\[P[0-3]\]\s*/, "").trim().toLowerCase();
  return createHash("sha256").update(`${path}\n${title}`).digest("hex");
}

export function getDiscussionFingerprint(body: string): string | null {
  return body.match(FINDING_MARKER_RE)?.[1]?.toLowerCase() ?? null;
}

function parseDiscussionFinding(discussion: HodorDiscussion): ReviewStateFinding | null {
  const fingerprint = getDiscussionFingerprint(discussion.body);
  const titleMatch = discussion.body.match(FINDING_TITLE_RE);
  if (!fingerprint || !titleMatch) return null;

  const titleLineEnd = discussion.body.indexOf("\n", titleMatch.index ?? 0);
  const remainder = titleLineEnd >= 0 ? discussion.body.slice(titleLineEnd + 1).trim() : "";
  const suggestionStart = remainder.indexOf("\n\n```suggestion");
  const body = (suggestionStart >= 0 ? remainder.slice(0, suggestionStart) : remainder).trim();
  const priority = Number(titleMatch[2]) as ReviewPriority;

  return {
    fingerprint,
    title: titleMatch[1],
    body,
    priority,
    filePath: discussion.filePath,
    lineRange:
      discussion.line == null
        ? undefined
        : { start: discussion.line, end: discussion.line },
  };
}

export function mergeReviewStateFindings(
  currentFindings: ReviewFinding[],
  discussions: HodorDiscussion[],
  workspacePath?: string | null,
  options: {
    includeExisting?: boolean;
    suppressResolvedCurrent?: boolean;
  } = {},
): ReviewStateFinding[] {
  const { includeExisting = true, suppressResolvedCurrent = false } = options;
  const merged = new Map<string, ReviewStateFinding>();
  const openFingerprints = new Set<string>();
  const resolvedFingerprints = new Set<string>();
  for (const discussion of discussions) {
    const fingerprint = getDiscussionFingerprint(discussion.body);
    if (!fingerprint) continue;
    if (discussion.resolved) resolvedFingerprints.add(fingerprint);
    else openFingerprints.add(fingerprint);
  }

  for (const finding of currentFindings) {
    const fingerprint = getFindingFingerprint(finding, workspacePath);
    if (
      suppressResolvedCurrent &&
      resolvedFingerprints.has(fingerprint) &&
      !openFingerprints.has(fingerprint)
    ) {
      continue;
    }
    merged.set(fingerprint, {
      fingerprint,
      title: finding.title,
      body: finding.body,
      priority: finding.priority,
      filePath: relativizeWorkspacePath(
        finding.code_location.absolute_file_path,
        workspacePath ?? undefined,
      ),
      lineRange: finding.code_location.line_range,
    });
  }

  if (includeExisting) {
    for (const discussion of discussions) {
      if (discussion.resolved) continue;
      const finding = parseDiscussionFinding(discussion);
      if (finding && !merged.has(finding.fingerprint)) {
        merged.set(finding.fingerprint, finding);
      }
    }
  }

  return [...merged.values()];
}
