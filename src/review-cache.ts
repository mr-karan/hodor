import { createHash } from "node:crypto";
import { gzipSync, gunzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
import { validateReviewOutput } from "./review.js";
import { relativizeWorkspacePath } from "./utils/path.js";
import type { NoteEntry, ReviewOutput } from "./types.js";

export const REVIEW_PROMPT_VERSION = "2026-07-16.1";

const CACHE_MARKER_RE = /<!--\s*hodor:cache:v1:([A-Za-z0-9_-]+)\s*-->/;

interface ReviewCachePayload {
  key: string;
  review: ReviewOutput;
}

export function getReviewCacheKey(opts: {
  headSha: string;
  model: string;
  requestedReasoningEffort?: string;
  customPrompt?: string | null;
  promptFile?: string | null;
}): string {
  let promptFileContents = "";
  if (opts.promptFile) {
    promptFileContents = readFileSync(opts.promptFile, "utf-8");
  }

  return createHash("sha256")
    .update(JSON.stringify({
      version: REVIEW_PROMPT_VERSION,
      headSha: opts.headSha,
      model: opts.model,
      // "auto" deliberately stays stable when an identical HEAD changes from
      // a full review to an empty incremental diff on a pipeline retry.
      reasoning: opts.requestedReasoningEffort?.toLowerCase() ?? "auto",
      customPrompt: opts.customPrompt ?? "",
      promptFileContents,
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

export function findCachedReview(
  notes: NoteEntry[] | undefined | null,
  key: string,
): ReviewOutput | null {
  if (!notes) return null;

  const newestFirst = [...notes].sort((a, b) =>
    Date.parse(b.created_at ?? "") - Date.parse(a.created_at ?? ""),
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
