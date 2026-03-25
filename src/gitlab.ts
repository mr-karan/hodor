import { exec, execJson } from "./utils/exec.js";
import { logger } from "./utils/logger.js";
import type { MrMetadata, NoteEntry } from "./types.js";

const DEFAULT_GITLAB_HOST = "gitlab.com";

/**
 * Parse concatenated JSON arrays from `glab api --paginate`.
 * glab outputs `[...][...][...]` — one array per page, no delimiter.
 * We track bracket depth (respecting strings/escapes) to find each
 * top-level array, parse them individually, and merge with flat().
 */
export function parseGlabPaginatedJson(raw: string): Array<Record<string, unknown>> {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const chunks: string[] = [];
  let depth = 0;
  let inString = false;
  let escaped = false;
  let start = -1;

  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\" && inString) {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;

    if (ch === "[") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "]") {
      depth--;
      if (depth === 0 && start >= 0) {
        chunks.push(trimmed.slice(start, i + 1));
        start = -1;
      }
    }
  }

  const results: Array<Record<string, unknown>> = [];
  for (const chunk of chunks) {
    const parsed = JSON.parse(chunk) as Array<Record<string, unknown>>;
    if (Array.isArray(parsed)) {
      results.push(...parsed);
    }
  }
  return results;
}

export class GitLabAPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GitLabAPIError";
  }
}

function normalizeBaseUrl(host?: string | null): string {
  const candidate =
    host ||
    process.env.GITLAB_HOST ||
    process.env.CI_SERVER_URL ||
    DEFAULT_GITLAB_HOST;
  const trimmed = candidate.trim() || DEFAULT_GITLAB_HOST;
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed.replace(/\/+$/, "");
  }
  return `https://${trimmed}`.replace(/\/+$/, "");
}

function encodedProjectPath(owner: string, repo: string): string {
  const projectPath = [owner.replace(/^\/+|\/+$/g, ""), repo.replace(/^\/+|\/+$/g, "")]
    .filter(Boolean)
    .join("/");
  return encodeURIComponent(projectPath);
}

function glabEnv(host?: string | null): NodeJS.ProcessEnv {
  const env = { ...process.env };
  // Ensure glab knows which host to talk to
  const baseUrl = normalizeBaseUrl(host);
  const hostname = baseUrl.replace(/^https?:\/\//, "");
  env.GITLAB_HOST = hostname;
  return env;
}

/**
 * Fetch merge request metadata using glab api.
 */
export async function fetchGitlabMrInfo(
  owner: string,
  repo: string,
  mrNumber: number | string,
  host?: string | null,
  options?: { includeComments?: boolean },
): Promise<MrMetadata> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  let mrData: Record<string, unknown>;
  try {
    mrData = await execJson<Record<string, unknown>>(
      "glab",
      ["api", `projects/${encoded}/merge_requests/${mrNumber}`],
      { env },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to fetch MR !${mrNumber}: ${msg}`);
  }

  const metadata: MrMetadata = {
    title: mrData.title as string | undefined,
    description: (mrData.description as string) ?? "",
    source_branch: mrData.source_branch as string | undefined,
    target_branch: mrData.target_branch as string | undefined,
    changes_count: mrData.changes_count as number | undefined,
    labels: mrData.labels as string[] | undefined,
    author: mrData.author as { username?: string; name?: string } | undefined,
    pipeline: mrData.pipeline as { status?: string; web_url?: string } | undefined,
    state: mrData.state as string | undefined,
  };

  if (options?.includeComments) {
    try {
      // glab --paginate concatenates JSON arrays across pages (e.g., `[...][...]`).
      // Parse each top-level array separately and merge, avoiding regex on raw JSON
      // which could corrupt string values containing `][`.
      const { stdout: rawNotes } = await exec(
        "glab",
        ["api", `projects/${encoded}/merge_requests/${mrNumber}/notes`, "--paginate"],
        { env },
      );
      const notes = parseGlabPaginatedJson(rawNotes);
      metadata.Notes = notes.map((n) => ({
        body: (n.body as string) ?? "",
        author: n.author as { username?: string; name?: string } | undefined,
        created_at: n.created_at as string | undefined,
        system: n.system as boolean | undefined,
      }));
    } catch (err) {
      logger.warn(`Failed to fetch MR notes: ${err instanceof Error ? err.message : err}`);
    }
  }

  return metadata;
}

/**
 * Post a comment on a GitLab merge request using glab api.
 */
export async function postGitlabMrComment(
  owner: string,
  repo: string,
  mrNumber: number | string,
  body: string,
  host?: string | null,
): Promise<void> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  try {
    await exec(
      "glab",
      [
        "api",
        `projects/${encoded}/merge_requests/${mrNumber}/notes`,
        "--method",
        "POST",
        "--field",
        `body=${body}`,
      ],
      { env },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to post comment to MR !${mrNumber}: ${msg}`);
  }
}

/**
 * Summarize GitLab notes into a human-readable bullet list.
 */
export function summarizeGitlabNotes(
  notes: NoteEntry[] | undefined | null,
  maxEntries = 5,
): string {
  if (!notes || notes.length === 0) return "";

  const trivialPatterns = new Set([
    "lgtm",
    "+1",
    "-1",
    "👍",
    "👎",
    "thanks",
    "thank you",
    "looks good",
    "approved",
    "🚀",
    "✅",
    "❌",
  ]);

  const filtered: Array<{ username: string; body: string; createdAt: string }> = [];
  for (const note of notes) {
    const body = (note.body ?? "").trim();
    if (!body) continue;
    if (note.system) continue;
    if (body.length < 20) continue;

    const bodyLower = body.toLowerCase();
    let isTrivial = false;
    for (const pattern of trivialPatterns) {
      if (bodyLower.includes(pattern) && body.length < 50) {
        isTrivial = true;
        break;
      }
    }
    if (isTrivial) continue;

    const username =
      note.author?.username ?? note.author?.name ?? "unknown";
    filtered.push({ username, body, createdAt: note.created_at ?? "" });
  }

  // Sort oldest first
  filtered.sort((a, b) => a.createdAt.localeCompare(b.createdAt));

  // Take most recent
  const recent = filtered.slice(-maxEntries);

  const lines: string[] = [];
  for (const { username, body, createdAt } of recent) {
    let timestampStr = "";
    if (createdAt) {
      try {
        const dt = new Date(createdAt);
        timestampStr = dt.toISOString().replace("T", " ").slice(0, 16);
      } catch {
        timestampStr = createdAt.slice(0, 10);
      }
    }

    const header = timestampStr
      ? `- ${timestampStr} @${username}:`
      : `- @${username}:`;
    const indentedBody = body.split("\n").join("\n  ");
    lines.push(`${header}\n  ${indentedBody}`);
  }

  return lines.join("\n");
}
