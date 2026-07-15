import type { MrMetadata } from "./types.js";
import { exec } from "./utils/exec.js";
import { logger } from "./utils/logger.js";

const HODOR_REVIEW_SHA_RE = /^\s*<!--\s*hodor:sha:([a-f0-9]{40})\s*-->/i;

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

export async function findLatestValidReviewSha(
  notes: MrMetadata["Notes"] | undefined | null,
  workspacePath: string,
): Promise<string | null> {
  const candidates = getHodorReviewShaCandidates(notes);
  if (candidates.length === 0) return null;

  logger.info(`Found ${candidates.length} previous Hodor review marker(s)`);
  for (const sha of candidates) {
    try {
      const { stdout: objectType } = await exec("git", ["cat-file", "-t", sha], {
        cwd: workspacePath,
      });
      if (objectType.trim() !== "commit") throw new Error("not a commit");
      await exec("git", ["merge-base", "--is-ancestor", sha, "HEAD"], {
        cwd: workspacePath,
      });
      return sha;
    } catch {
      logger.info(
        `Skipping previous review SHA ${sha.slice(0, 8)}; not a valid ancestor of HEAD`,
      );
    }
  }
  return null;
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
