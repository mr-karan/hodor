import { exec, execJson } from "./utils/exec.js";
import { isRecord, parsePaginatedJsonArrays } from "./utils/json.js";
import { logger } from "./utils/logger.js";
import type { MrMetadata, NoteAuthor, UntrustedNote } from "./types.js";

export class GitHubAPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GitHubAPIError";
  }
}

function ghApiArgs(host: string, path: string, ...extra: string[]): string[] {
  return ["api", "--hostname", host, path, ...extra];
}

/**
 * Resolve a GitHub account: the authenticated user when `login` is omitted,
 * else the named user or app bot (e.g. `name[bot]`).
 */
export async function fetchGithubUser(
  host: string,
  login?: string,
): Promise<{ userId: number; login: string }> {
  const label = login ? `GitHub user ${login}` : "the authenticated GitHub user";
  let user: unknown;
  try {
    user = await execJson<unknown>("gh", ghApiArgs(host, login ? `users/${encodeURIComponent(login)}` : "user"));
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitHubAPIError(`Failed to resolve ${label}: ${msg}`);
  }
  const userId = isRecord(user) ? user.id : undefined;
  const userLogin = isRecord(user) ? user.login : undefined;
  if (typeof userId !== "number" || !Number.isSafeInteger(userId) || typeof userLogin !== "string") {
    throw new GitHubAPIError(`${label} has no numeric id`);
  }
  return { userId, login: userLogin };
}

/**
 * Fetch PR metadata. Notes come from REST, not `gh pr view`: GraphQL author
 * data has only the login, and a bot login matches a same-named human
 * account. REST includes the numeric `user.id` that provenance compares.
 */
export async function fetchGithubPrMetadata(
  owner: string,
  repo: string,
  prNumber: number,
  host: string,
): Promise<MrMetadata> {
  const [raw, notes] = await Promise.all([
    fetchGithubPrInfo(owner, repo, prNumber),
    fetchGithubPrNotes(owner, repo, prNumber, host).catch((err: unknown) => {
      logger.warn(`Failed to fetch PR notes: ${err instanceof Error ? err.message : err}`);
      return undefined;
    }),
  ]);
  return { ...normalizeGithubMetadata(raw), Notes: notes };
}

/** Issue comments and review bodies on a PR, with numeric author ids. */
export async function fetchGithubPrNotes(
  owner: string,
  repo: string,
  prNumber: number,
  host: string,
): Promise<UntrustedNote[]> {
  const repoPath = `repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`;
  const fetchPages = async (path: string): Promise<Array<Record<string, unknown>>> => {
    try {
      const { stdout } = await exec("gh", ghApiArgs(host, path, "--paginate"));
      return parsePaginatedJsonArrays(stdout);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new GitHubAPIError(`Failed to fetch ${path}: ${msg}`);
    }
  };
  const [comments, reviews] = await Promise.all([
    fetchPages(`${repoPath}/issues/${prNumber}/comments?per_page=100`),
    fetchPages(`${repoPath}/pulls/${prNumber}/reviews?per_page=100`),
  ]);

  return [
    ...comments.map((comment) => ({
      body: typeof comment.body === "string" ? comment.body : "",
      author: parseGithubRestUser(comment.user),
      created_at: typeof comment.created_at === "string" ? comment.created_at : undefined,
      updated_at: typeof comment.updated_at === "string" ? comment.updated_at : undefined,
      system: false,
    })),
    ...reviews.map((review) => ({
      body: typeof review.body === "string" ? review.body : "",
      author: parseGithubRestUser(review.user),
      created_at: typeof review.submitted_at === "string" ? review.submitted_at : undefined,
      system: false,
    })),
  ];
}

function parseGithubRestUser(value: unknown): NoteAuthor | undefined {
  if (!isRecord(value)) return undefined;
  return {
    id: typeof value.id === "number" && Number.isSafeInteger(value.id) ? value.id : undefined,
    username: typeof value.login === "string" ? value.login : undefined,
  };
}

export async function fetchGithubPrInfo(
  owner: string,
  repo: string,
  prNumber: number | string,
): Promise<Record<string, unknown>> {
  const fields = [
    "number",
    "title",
    "body",
    "author",
    "baseRefName",
    "headRefName",
    "baseRefOid",
    "headRefOid",
    "changedFiles",
    "labels",
    "state",
    "isDraft",
    "createdAt",
    "updatedAt",
    "mergeable",
    "url",
  ];

  const repoFullPath = `${owner}/${repo}`;
  try {
    return await execJson<Record<string, unknown>>("gh", [
      "pr",
      "view",
      String(prNumber),
      "-R",
      repoFullPath,
      "--json",
      fields.join(","),
    ]);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitHubAPIError(msg);
  }
}

export function normalizeGithubMetadata(
  raw: Record<string, unknown>,
): MrMetadata {
  const author = (raw.author as Record<string, string>) ?? {};
  const labels = (raw.labels as Array<Record<string, string>>) ?? [];

  return {
    title: raw.title as string | undefined,
    description: (raw.body as string) ?? "",
    source_branch: raw.headRefName as string | undefined,
    target_branch: raw.baseRefName as string | undefined,
    changes_count: raw.changedFiles as number | undefined,
    labels: labels.map((lbl) => ({ name: lbl.name ?? lbl.id })),
    author: {
      username: author.login ?? author.name,
      name: author.name,
    },
  };
}
