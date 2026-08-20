import type { MrMetadata, Platform } from "./types.js";
import { exec } from "./utils/exec.js";
import { logger } from "./utils/logger.js";

const HODOR_REVIEW_SHA_RE = /^\s*<!--\s*hodor:sha:([a-f0-9]{40})\s*-->/i;

export type ReviewDiffMode = "full" | "incremental" | "snapshot" | "local" | "reused";

export interface PreviousReviewBase {
  sha: string;
  mode: "incremental" | "snapshot";
}

export interface ReviewDiffArgsOptions {
  platform: Platform;
  targetBranch: string;
  diffBaseSha?: string | null;
  previousReviewSha?: string | null;
  reviewDiffMode?: ReviewDiffMode;
  localMode?: boolean;
}

export interface DiffStats {
  files: number;
  additions: number;
  deletions: number;
  bytes: number;
}

/**
 * Build the git diff arguments used by both the embedded diff and the prompt.
 *
 * A rewritten GitLab MR branch must be compared with the current MR base. The
 * previously reviewed SHA is from the old branch history; diffing from it
 * directly also includes target-branch commits brought in by a rebase.
 */
export function getReviewDiffArgs(options: ReviewDiffArgsOptions): string[] {
  const {
    platform,
    targetBranch,
    diffBaseSha,
    previousReviewSha,
    reviewDiffMode,
    localMode = false,
  } = options;
  const rebasedGitlabReview = platform === "gitlab" && reviewDiffMode === "snapshot";

  if (previousReviewSha && !rebasedGitlabReview) {
    return reviewDiffMode === "snapshot"
      ? ["--no-pager", "diff", previousReviewSha, "HEAD"]
      : ["--no-pager", "diff", `${previousReviewSha}...HEAD`];
  }
  if (localMode) return ["--no-pager", "diff", targetBranch];
  if (diffBaseSha) return ["--no-pager", "diff", diffBaseSha, "HEAD"];
  return ["--no-pager", "diff", `origin/${targetBranch}...HEAD`];
}

export function getHodorReviewShaCandidates(
  notes: MrMetadata["Notes"] | undefined | null,
): string[] {
  if (!notes || notes.length === 0) return [];

  const candidates: Array<{ sha: string; createdAtMs: number | null; index: number }> = [];
  for (const [index, note] of notes.entries()) {
    const match = note.body?.match(HODOR_REVIEW_SHA_RE);
    if (!match) continue;
    const createdAtMs = Date.parse(note.created_at ?? "");
    candidates.push({
      sha: match[1],
      createdAtMs: Number.isFinite(createdAtMs) ? createdAtMs : null,
      index,
    });
  }

  candidates.sort((a, b) => {
    if (a.createdAtMs != null && b.createdAtMs != null && a.createdAtMs !== b.createdAtMs) {
      return b.createdAtMs - a.createdAtMs;
    }
    if (a.createdAtMs != null && b.createdAtMs == null) return -1;
    if (a.createdAtMs == null && b.createdAtMs != null) return 1;
    return a.index - b.index;
  });

  return [...new Set(candidates.map(({ sha }) => sha))];
}

export async function findLatestReviewBase(
  notes: MrMetadata["Notes"] | undefined | null,
  workspacePath: string,
): Promise<PreviousReviewBase | null> {
  const candidates = getHodorReviewShaCandidates(notes);
  if (candidates.length === 0) return null;

  logger.info(`Found ${candidates.length} previous Hodor review marker(s)`);
  for (const sha of candidates) {
    try {
      let objectType: string;
      try {
        ({ stdout: objectType } = await exec("git", ["cat-file", "-t", sha], {
          cwd: workspacePath,
        }));
      } catch {
        // Rewritten commits may no longer be reachable from the checked-out
        // branch, especially in shallow CI clones. Most Git servers still let
        // an authenticated client fetch the exact object by SHA.
        await exec("git", ["fetch", "--quiet", "origin", sha], {
          cwd: workspacePath,
        });
        ({ stdout: objectType } = await exec("git", ["cat-file", "-t", sha], {
          cwd: workspacePath,
        }));
      }
      if (objectType.trim() !== "commit") throw new Error("not a commit");

      try {
        await exec("git", ["merge-base", "--is-ancestor", sha, "HEAD"], {
          cwd: workspacePath,
        });
        return { sha, mode: "incremental" };
      } catch {
        logger.info(
          `Previous review SHA ${sha.slice(0, 8)} is not an ancestor; using snapshot delta`,
        );
        return { sha, mode: "snapshot" };
      }
    } catch {
      logger.info(
        `Skipping previous review SHA ${sha.slice(0, 8)}; commit is unavailable`,
      );
    }
  }
  return null;
}

/** Backwards-compatible helper for callers that only accept ancestor diffs. */
export async function findLatestValidReviewSha(
  notes: MrMetadata["Notes"] | undefined | null,
  workspacePath: string,
): Promise<string | null> {
  const base = await findLatestReviewBase(notes, workspacePath);
  return base?.mode === "incremental" ? base.sha : null;
}

export function getDiffStats(diff: string): DiffStats {
  let files = 0;
  let additions = 0;
  let deletions = 0;

  for (const line of diff.split("\n")) {
    if (line.startsWith("diff --git ")) files++;
    else if (line.startsWith("+") && !line.startsWith("+++")) additions++;
    else if (line.startsWith("-") && !line.startsWith("---")) deletions++;
  }

  return {
    files,
    additions,
    deletions,
    bytes: Buffer.byteLength(diff, "utf-8"),
  };
}

export function getChangedFiles(diff: string): string[] {
  const files: string[] = [];
  for (const match of diff.matchAll(/^diff --git a\/(.*?) b\/(.*?)$/gm)) {
    files.push(match[2]);
  }
  return [...new Set(files)];
}

const DIFF_SKIP_PATTERNS: RegExp[] = [
  /(?:^|\/)testdata\//,
  /(?:^|\/)(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|go\.sum|Cargo\.lock|poetry\.lock|Gemfile\.lock|composer\.lock)$/,
  /\.mdx?$/,
];

export function filterEmbeddedDiff(
  rawDiff: string,
): { filtered: string; skippedFiles: string[] } {
  const skippedFiles: string[] = [];
  const sections = rawDiff.split(/(?=^diff --git )/m);
  const kept: string[] = [];
  for (const section of sections) {
    const match = section.match(/^diff --git a\/(.*?) b\//);
    if (!match) {
      kept.push(section);
      continue;
    }
    const filePath = match[1];
    if (DIFF_SKIP_PATTERNS.some((pattern) => pattern.test(filePath))) {
      skippedFiles.push(filePath);
    } else {
      kept.push(section);
    }
  }
  return { filtered: kept.join(""), skippedFiles };
}
