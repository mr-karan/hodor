import { exec, execJson } from "./utils/exec.js";
import { logger } from "./utils/logger.js";
import type { MrMetadata, NoteEntry } from "./types.js";
import { HODOR_REVIEW_MARKER } from "./render.js";

export { HODOR_REVIEW_MARKER };

export interface DiffRefs {
  base_sha: string;
  head_sha: string;
  start_sha: string;
}

const DEFAULT_GITLAB_HOST = "gitlab.com";

/**
 * Match notes Hodor itself created. We require the marker to appear at the very start
 * of the note (after optional whitespace). This avoids deleting/resolving human notes
 * that quote the marker incidentally (e.g., in a code block discussing Hodor itself).
 */
function isHodorNote(body: unknown, marker = HODOR_REVIEW_MARKER): boolean {
  if (typeof body !== "string") return false;
  return body.trimStart().startsWith(marker);
}

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
    try {
      const parsed = JSON.parse(chunk) as Array<Record<string, unknown>>;
      if (Array.isArray(parsed)) results.push(...parsed);
    } catch (err) {
      logger.warn(
        `Skipping malformed glab pagination chunk: ${err instanceof Error ? err.message : err}`,
      );
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
        "--input",
        "-",
      ],
      { env, input: JSON.stringify({ body }) },
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

function isPositionErrorMessage(message: string): boolean {
  const lower = message.toLowerCase();
  const hasPositionHint =
    lower.includes("line_code") ||
    lower.includes("position") ||
    lower.includes("must be part of the diff") ||
    lower.includes("part of the diff");
  const hasStatusHint = lower.includes("400") || lower.includes("422");
  return hasPositionHint && hasStatusHint;
}

export async function getGitlabMrDiffRefs(
  owner: string,
  repo: string,
  mrNumber: number | string,
  host?: string | null,
): Promise<DiffRefs> {
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
    throw new GitLabAPIError(`Failed to fetch diff refs for MR !${mrNumber}: ${msg}`);
  }

  const diffRefs = mrData.diff_refs as Record<string, unknown> | undefined;
  const base_sha = diffRefs?.base_sha;
  const head_sha = diffRefs?.head_sha;
  const start_sha = diffRefs?.start_sha;

  if (
    typeof base_sha !== "string" ||
    typeof head_sha !== "string" ||
    typeof start_sha !== "string" ||
    !base_sha ||
    !head_sha ||
    !start_sha
  ) {
    throw new GitLabAPIError(`MR !${mrNumber} has missing or incomplete diff_refs`);
  }

  return { base_sha, head_sha, start_sha };
}

export async function postGitlabInlineComment(
  owner: string,
  repo: string,
  mrNumber: number | string,
  body: string,
  filePath: string,
  line: number,
  diffRefs: DiffRefs,
  host?: string | null,
): Promise<Record<string, unknown> | null> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);
  const endpoint = `projects/${encoded}/merge_requests/${mrNumber}/discussions`;

  const payload: Record<string, unknown> = {
    body,
    position: {
      base_sha: diffRefs.base_sha,
      head_sha: diffRefs.head_sha,
      start_sha: diffRefs.start_sha,
      position_type: "text",
      old_path: filePath,
      new_path: filePath,
      new_line: line,
    },
  };

  try {
    return await execJson<Record<string, unknown>>(
      "glab",
      ["api", endpoint, "--method", "POST", "--input", "-"],
      {
        env,
        input: JSON.stringify(payload),
      },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (isPositionErrorMessage(msg)) {
      logger.warn(`Skipping inline comment for ${filePath}:${line}: ${msg}`);
      return null;
    }
    throw new GitLabAPIError(`Failed to post inline comment to MR !${mrNumber}: ${msg}`);
  }
}

export async function createGitlabDraftNote(
  owner: string,
  repo: string,
  mrNumber: number | string,
  body: string,
  host?: string | null,
  opts?: { filePath?: string; line?: number; diffRefs?: DiffRefs },
): Promise<Record<string, unknown>> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);
  const endpoint = `projects/${encoded}/merge_requests/${mrNumber}/draft_notes`;

  const payload: Record<string, unknown> = {
    note: body,
  };

  if (opts?.filePath && typeof opts.line === "number" && opts.diffRefs) {
    payload.position = {
      base_sha: opts.diffRefs.base_sha,
      head_sha: opts.diffRefs.head_sha,
      start_sha: opts.diffRefs.start_sha,
      position_type: "text",
      old_path: opts.filePath,
      new_path: opts.filePath,
      new_line: opts.line,
    };
  }

  try {
    return await execJson<Record<string, unknown>>(
      "glab",
      ["api", endpoint, "--method", "POST", "--input", "-"],
      {
        env,
        input: JSON.stringify(payload),
      },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to create draft note for MR !${mrNumber}: ${msg}`);
  }
}

export async function bulkPublishGitlabDraftNotes(
  owner: string,
  repo: string,
  mrNumber: number | string,
  host?: string | null,
): Promise<void> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  try {
    await exec(
      "glab",
      [
        "api",
        `projects/${encoded}/merge_requests/${mrNumber}/draft_notes/bulk_publish`,
        "--method",
        "POST",
      ],
      { env },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to bulk publish draft notes for MR !${mrNumber}: ${msg}`);
  }
}

type GitlabCommitStatusState = "pending" | "running" | "success" | "failed" | "canceled";

export async function postGitlabCommitStatus(
  owner: string,
  repo: string,
  sha: string,
  state: GitlabCommitStatusState,
  host?: string | null,
  opts?: { name?: string; description?: string; targetUrl?: string },
): Promise<void> {
  const allowedStates = new Set<GitlabCommitStatusState>([
    "pending",
    "running",
    "success",
    "failed",
    "canceled",
  ]);
  if (!allowedStates.has(state)) {
    throw new GitLabAPIError(`Invalid GitLab commit status state: ${state}`);
  }

  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);
  const endpoint = `projects/${encoded}/statuses/${sha}`;
  const payload: Record<string, unknown> = {
    state,
    name: opts?.name ?? "hodor",
  };

  if (opts?.description) {
    payload.description = opts.description;
  }
  if (opts?.targetUrl) {
    payload.target_url = opts.targetUrl;
  }

  try {
    await exec("glab", ["api", endpoint, "--method", "POST", "--input", "-"], {
      env,
      input: JSON.stringify(payload),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to post commit status for ${sha}: ${msg}`);
  }
}

export async function cleanupHodorComments(
  owner: string,
  repo: string,
  mrNumber: number | string,
  host?: string | null,
  marker = HODOR_REVIEW_MARKER,
): Promise<number> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  let notes: Array<Record<string, unknown>>;
  try {
    const { stdout: rawNotes } = await exec(
      "glab",
      [
        "api",
        `projects/${encoded}/merge_requests/${mrNumber}/notes?per_page=100`,
        "--paginate",
      ],
      { env },
    );
    notes = parseGlabPaginatedJson(rawNotes);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to list notes for MR !${mrNumber}: ${msg}`);
  }

  const matchedNotes = notes.filter((note) => isHodorNote(note.body, marker));

  let deletedCount = 0;
  let failedCount = 0;
  for (const note of matchedNotes) {
    const noteId = note.id;
    if (typeof noteId !== "number") continue;

    try {
      await exec(
        "glab",
        [
          "api",
          `projects/${encoded}/merge_requests/${mrNumber}/notes/${noteId}`,
          "--method",
          "DELETE",
        ],
        { env },
      );
      deletedCount += 1;
      logger.debug(`Deleted GitLab MR note ${noteId}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // Inline (discussion) notes can't be deleted via /notes/{id}; log and continue
      // so one stale note doesn't abort the whole cleanup.
      logger.warn(`Skipping note ${noteId} on MR !${mrNumber}: ${msg}`);
      failedCount += 1;
    }
  }

  if (failedCount > 0) {
    logger.warn(`Cleanup left ${failedCount} note(s) undeleted on MR !${mrNumber}`);
  }

  return deletedCount;
}

export async function listHodorDiscussions(
  owner: string,
  repo: string,
  mrNumber: number | string,
  host?: string | null,
  marker = HODOR_REVIEW_MARKER,
): Promise<
  Array<{
    discussionId: string;
    noteId: number;
    body: string;
    resolved: boolean;
    filePath?: string;
    line?: number;
  }>
> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  let discussions: Array<Record<string, unknown>>;
  try {
    const { stdout: rawDiscussions } = await exec(
      "glab",
      [
        "api",
        `projects/${encoded}/merge_requests/${mrNumber}/discussions?per_page=100`,
        "--paginate",
      ],
      { env },
    );
    discussions = parseGlabPaginatedJson(rawDiscussions);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to list discussions for MR !${mrNumber}: ${msg}`);
  }

  const results: Array<{
    discussionId: string;
    noteId: number;
    body: string;
    resolved: boolean;
    filePath?: string;
    line?: number;
  }> = [];

  for (const discussion of discussions) {
    const discussionId = discussion.id;
    if (typeof discussionId !== "string") {
      continue;
    }

    const notes = discussion.notes;
    if (!Array.isArray(notes)) {
      continue;
    }

    for (const note of notes) {
      if (!note || typeof note !== "object") {
        continue;
      }
      const noteObj = note as Record<string, unknown>;
      const noteId = noteObj.id;
      const body = noteObj.body;
      if (typeof noteId !== "number" || typeof body !== "string" || !isHodorNote(body, marker)) {
        continue;
      }

      const position =
        noteObj.position && typeof noteObj.position === "object"
          ? (noteObj.position as Record<string, unknown>)
          : undefined;

      const filePath = typeof position?.new_path === "string" ? position.new_path : undefined;
      const line = typeof position?.new_line === "number" ? position.new_line : undefined;

      results.push({
        discussionId,
        noteId,
        body,
        resolved: Boolean(noteObj.resolved),
        filePath,
        line,
      });
    }
  }

  return results;
}

export async function resolveGitlabDiscussions(
  owner: string,
  repo: string,
  mrNumber: number | string,
  discussionIds: string[],
  host?: string | null,
): Promise<number> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  let resolvedCount = 0;

  for (const discussionId of discussionIds) {
    try {
      await exec(
        "glab",
        [
          "api",
          `projects/${encoded}/merge_requests/${mrNumber}/discussions/${discussionId}`,
          "--method",
          "PUT",
          "--input",
          "-",
        ],
        {
          env,
          input: JSON.stringify({ resolved: true }),
        },
      );
      resolvedCount += 1;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn(`Failed to resolve discussion ${discussionId} on MR !${mrNumber}: ${msg}`);
    }
  }

  return resolvedCount;
}
