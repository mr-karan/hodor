import { createHash } from "node:crypto";
import { gzipSync, gunzipSync } from "node:zlib";
import { validateReviewOutput } from "./review.js";
import { relativizeWorkspacePath } from "./utils/path.js";
import type { Platform, ReviewOutput, TrustedHodorNote } from "./types.js";

// Bumped when the review prompt or cache key changes, so older markers never match.
export const REVIEW_PROMPT_VERSION = "2026-10-07.1";

const CACHE_MARKER_RE = /<!--\s*hodor:cache:v1:([A-Za-z0-9_-]+)\s*-->/;

interface ReviewCachePayload {
  key: string;
  review: ReviewOutput;
}

/** The MR/PR a cached review belongs to and the diff range it covered. */
export interface ReviewCacheScope {
  platform: Platform;
  host: string;
  projectPath: string;
  reviewNumber: number;
  targetBranch: string;
  /** Commit the target side of the review diff is computed against. */
  baseSha: string;
}

export function getReviewCacheKey(opts: {
  scope: ReviewCacheScope;
  headSha: string;
  model: string;
  requestedReasoningEffort?: string;
  instructions?: readonly string[];
  focus?: string | null;
  guidanceSnapshotSha: string;
}): string {
  const { scope } = opts;
  return createHash("sha256")
    .update(JSON.stringify({
      version: REVIEW_PROMPT_VERSION,
      platform: scope.platform,
      host: scope.host.toLowerCase(),
      projectPath: scope.projectPath,
      reviewNumber: scope.reviewNumber,
      targetBranch: scope.targetBranch,
      baseSha: scope.baseSha,
      headSha: opts.headSha,
      model: opts.model,
      // "auto" deliberately stays stable when an identical HEAD changes from
      // a full review to an empty incremental diff on a pipeline retry.
      reasoning: opts.requestedReasoningEffort?.toLowerCase() ?? "auto",
      instructions: opts.instructions ?? [],
      focus: opts.focus ?? "",
      guidanceSnapshotSha: opts.guidanceSnapshotSha,
    }))
    .digest("hex");
}

export function buildReviewCacheMarker(
  key: string,
  review: ReviewOutput,
  workspacePath?: string | null,
): string {
  const portableReview: ReviewOutput = {
    ...review,
    findings: review.findings.map((finding) => ({
      ...finding,
      code_location: {
        ...finding.code_location,
        absolute_file_path: `/workspace/${relativizeWorkspacePath(
          finding.code_location.absolute_file_path,
          workspacePath ?? undefined,
        )}`,
      },
    })),
  };
  const payload: ReviewCachePayload = { key, review: portableReview };
  const encoded = gzipSync(JSON.stringify(payload)).toString("base64url");
  return `<!-- hodor:cache:v1:${encoded} -->`;
}

/** Only authenticated notes are decoded. Untrusted markers never reach gunzip. */
export function findCachedReview(
  notes: readonly TrustedHodorNote[],
  key: string,
): ReviewOutput | null {
  const newestFirst = [...notes].sort((a, b) =>
    Date.parse(b.updated_at ?? b.created_at ?? "") -
    Date.parse(a.updated_at ?? a.created_at ?? ""),
  );

  for (const note of newestFirst) {
    const encoded = note.body?.match(CACHE_MARKER_RE)?.[1];
    if (!encoded || encoded.length > 500_000) continue;

    try {
      const payload = JSON.parse(
        gunzipSync(Buffer.from(encoded, "base64url"), { maxOutputLength: 1_000_000 }).toString("utf-8"),
      ) as Partial<ReviewCachePayload>;
      if (payload.key !== key || !payload.review) continue;
      return validateReviewOutput(payload.review);
    } catch {
      // Ignore malformed or obsolete markers and perform a fresh review.
    }
  }

  return null;
}
