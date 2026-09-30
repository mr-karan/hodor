import { exec, execJson } from "./utils/exec.js";
import { logger } from "./utils/logger.js";
import { isRecord, parsePaginatedJsonArrays } from "./utils/json.js";
import type { GitlabPublisherIdentity, MrMetadata, NoteAuthor, NoteEntry } from "./types.js";
import {
  HODOR_REVIEW_MARKER,
  HODOR_SUMMARY_MARKER,
  renderSupersededSummary,
} from "./render.js";
import { getDiscussionFingerprint, getFixedMarker } from "./review-state.js";

export { HODOR_REVIEW_MARKER, HODOR_SUMMARY_MARKER };

export interface DiffRefs {
  base_sha: string;
  head_sha: string;
  start_sha: string;
}

export interface HodorDiscussion {
  discussionId: string;
  noteId: number;
  body: string;
  resolved: boolean;
  filePath?: string;
  line?: number;
  /** Head SHA from the publisher's latest fixed-reply for this finding, if any. */
  fixedAtSha?: string;
  /** Username GitLab reports as having resolved the thread. */
  resolvedBy?: string;
  /** Latest created, updated, or resolved time across the thread's notes. */
  updatedAt?: string;
  /**
   * Replies in the thread from other accounts, oldest first. Untrusted text:
   * never read as Hodor state, even when it copies a Hodor marker.
   */
  humanReplies: ThreadReply[];
}

export interface ThreadReply {
  author: string;
  body: string;
}

const DEFAULT_GITLAB_HOST = "gitlab.com";

/**
 * Match notes Hodor itself created. The body must begin with a hodor-owned HTML
 * comment (either `<!-- hodor-review -->` or a sibling like `<!-- hodor:sha:... -->`
 * that hodor prepends to summary comments). Anchoring at the start avoids deleting
 * human notes that quote the marker incidentally (e.g., a code block discussing hodor).
 */
const HODOR_NOTE_PREFIX_RE = /^\s*<!--\s*hodor[-:]/;
const HODOR_CACHE_MARKER_RE = /<!--\s*hodor:cache:v1:[A-Za-z0-9_-]+\s*-->\s*/g;
const HODOR_SHA_PREFIX_RE = /^\s*<!--\s*hodor:sha:[a-f0-9]{40}\s*-->/i;
const HODOR_SUPERSEDED_PREFIX_RE = /^\s*<!--\s*hodor-review\s*-->\s*<!--\s*hodor:superseded\s*-->/;

export function isHodorGeneratedNote(body: unknown): boolean {
  if (typeof body !== "string") return false;
  // Fast path: body starts with the canonical marker (allowing leading whitespace).
  if (body.trimStart().startsWith(HODOR_REVIEW_MARKER)) return true;
  // Accept hodor's own SHA prefix, e.g. `<!-- hodor:sha:abc -->\n<!-- hodor-review -->\n...`
  if (HODOR_NOTE_PREFIX_RE.test(body)) {
    // Require the canonical marker to appear somewhere in the body so we don't
    // resolve unrelated `<!-- hodor:foo -->` notes that aren't review summaries.
    return body.includes(HODOR_REVIEW_MARKER);
  }
  return false;
}

function parseGitlabAuthor(value: unknown): NoteAuthor | undefined {
  if (!isRecord(value)) return undefined;
  return {
    id: typeof value.id === "number" && Number.isSafeInteger(value.id) ? value.id : undefined,
    username: typeof value.username === "string" ? value.username : undefined,
    name: typeof value.name === "string" ? value.name : undefined,
  };
}

/** True when a raw GitLab note is a user note written by the publishing identity. */
function isPublisherNote(note: Record<string, unknown>, identity: GitlabPublisherIdentity): boolean {
  return note.system !== true && parseGitlabAuthor(note.author)?.id === identity.userId;
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

function projectPath(owner: string, repo: string): string {
  return [owner.replace(/^\/+|\/+$/g, ""), repo.replace(/^\/+|\/+$/g, "")]
    .filter(Boolean)
    .join("/");
}

function encodedProjectPath(owner: string, repo: string): string {
  return encodeURIComponent(projectPath(owner, repo));
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
      const notes = parsePaginatedJsonArrays(rawNotes);
      metadata.Notes = notes.map((n) => ({
        body: typeof n.body === "string" ? n.body : "",
        author: parseGitlabAuthor(n.author),
        created_at: typeof n.created_at === "string" ? n.created_at : undefined,
        updated_at: typeof n.updated_at === "string" ? n.updated_at : undefined,
        system: n.system === true,
      }));
    } catch (err) {
      logger.warn(`Failed to fetch MR notes: ${err instanceof Error ? err.message : err}`);
    }
  }

  return metadata;
}

/**
 * Resolve the numeric id of the account glab authenticates as on this host.
 * Hodor trusts only notes written by this id.
 */
export async function fetchGitlabPublisherIdentity(
  host?: string | null,
): Promise<GitlabPublisherIdentity> {
  let user: unknown;
  try {
    user = await execJson<unknown>("glab", ["api", "user"], { env: glabEnv(host) });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to resolve the authenticated GitLab user: ${msg}`);
  }
  const userId = isRecord(user) ? user.id : undefined;
  if (typeof userId !== "number" || !Number.isSafeInteger(userId)) {
    throw new GitLabAPIError("Authenticated GitLab user has no numeric id");
  }
  return { platform: "gitlab", userId };
}

export interface PublishedSummary {
  /** Id of the new summary note, or null when GitLab's response had none. */
  noteId: number | null;
  /** Older Hodor summaries collapsed to a pointer at the new one. */
  collapsed: number;
}

/**
 * Post a new summary note, so it lands at the bottom of the MR and notifies
 * participants, then collapse Hodor's older summaries into a link to it.
 * Only a failed POST throws. Collapse failures are logged as warnings.
 */
export async function publishGitlabMrSummary(
  owner: string,
  repo: string,
  mrNumber: number | string,
  body: string,
  host: string | null | undefined,
  identity: GitlabPublisherIdentity,
): Promise<PublishedSummary> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);
  const notesEndpoint = `projects/${encoded}/merge_requests/${mrNumber}/notes`;

  let postStdout: string;
  try {
    ({ stdout: postStdout } = await exec(
      "glab",
      ["api", notesEndpoint, "--method", "POST", "-H", "Content-Type: application/json", "--input", "-"],
      { env, input: JSON.stringify({ body }) },
    ));
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to post summary to MR !${mrNumber}: ${msg}`);
  }

  const noteId = parseCreatedNoteId(postStdout);
  if (noteId === null) {
    logger.warn(`GitLab returned no note id for the new summary on MR !${mrNumber}; older summaries stay as they are`);
    return { noteId, collapsed: 0 };
  }

  let priorSummaryIds: number[];
  try {
    const { stdout } = await exec("glab", ["api", `${notesEndpoint}?per_page=100`, "--paginate"], { env });
    priorSummaryIds = parsePaginatedJsonArrays(stdout)
      .filter((note) => isActiveSummaryNote(note, identity))
      .map((note) => note.id)
      // Note ids increase monotonically. Collapse only older summaries, so two
      // overlapping runs cannot collapse each other's newer note.
      .filter((id): id is number => typeof id === "number" && id < noteId);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.warn(`Failed to list notes to collapse older summaries on MR !${mrNumber}: ${msg}`);
    return { noteId, collapsed: 0 };
  }

  const collapsedBody = renderSupersededSummary(
    `${normalizeBaseUrl(host)}/${projectPath(owner, repo)}/-/merge_requests/${mrNumber}#note_${noteId}`,
  );
  let collapsed = 0;
  for (const priorId of priorSummaryIds) {
    try {
      await exec(
        "glab",
        ["api", `${notesEndpoint}/${priorId}`, "--method", "PUT", "-H", "Content-Type: application/json", "--input", "-"],
        { env, input: JSON.stringify({ body: collapsedBody }) },
      );
      collapsed += 1;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn(`Failed to collapse older summary note ${priorId} on MR !${mrNumber}: ${msg}`);
    }
  }
  return { noteId, collapsed };
}

function parseCreatedNoteId(stdout: string): number | null {
  let note: unknown;
  try {
    note = JSON.parse(stdout.trim());
  } catch {
    return null;
  }
  const id = isRecord(note) ? note.id : undefined;
  return typeof id === "number" && Number.isSafeInteger(id) ? id : null;
}

/** A summary note by the publisher that still carries machine state. */
function isActiveSummaryNote(note: Record<string, unknown>, identity: GitlabPublisherIdentity): boolean {
  const body = note.body;
  if (
    typeof body !== "string" ||
    !isPublisherNote(note, identity) ||
    note.type != null ||
    note.position != null ||
    isSupersededSummary(body)
  ) {
    return false;
  }
  return (
    body.includes(HODOR_SUMMARY_MARKER) ||
    (HODOR_SHA_PREFIX_RE.test(body) && body.includes(HODOR_REVIEW_MARKER))
  );
}

/** True for an older summary that was collapsed to a link to a newer one. */
function isSupersededSummary(body: string): boolean {
  return HODOR_SUPERSEDED_PREFIX_RE.test(body);
}

/**
 * Summarize notes that are not authenticated Hodor state into a bullet list.
 * A participant note that copies a Hodor marker stays here, as human context.
 */
export function summarizeGitlabNotes(
  notes: readonly NoteEntry[] | undefined | null,
  maxEntries = 5,
): string {
  return summarizeNotes(notes, maxEntries, (note) => note.provenance !== "hodor");
}

/** Summarize only notes that partitionNotesByProvenance authenticated. */
export function summarizeHodorNotes(
  notes: readonly NoteEntry[] | undefined | null,
  maxEntries = 5,
): string {
  return summarizeNotes(
    notes,
    maxEntries,
    // Fixed-replies are thread state, shown with their finding in the prompt.
    (note) =>
      note.provenance === "hodor" &&
      !isSupersededSummary(note.body ?? "") &&
      getFixedMarker(note.body ?? "") === null,
  );
}

function summarizeNotes(
  notes: readonly NoteEntry[] | undefined | null,
  maxEntries: number,
  include: (note: NoteEntry) => boolean,
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
    if (!include(note)) continue;
    // Cache payloads are machine-only and can be large. Never feed their
    // compressed representation back into reviewer context.
    const body = (note.body ?? "").replace(HODOR_CACHE_MARKER_RE, "").trim();
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
    const boundedBody = body.length > 2_000 ? `${body.slice(0, 1_999).trimEnd()}…` : body;
    const indentedBody = boundedBody.split("\n").join("\n  ");
    lines.push(`${header}\n  ${indentedBody}`);
  }

  return lines.join("\n");
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
      ["api", endpoint, "--method", "POST", "-H", "Content-Type: application/json", "--input", "-"],
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

export async function publishGitlabDraftNote(
  owner: string,
  repo: string,
  mrNumber: number | string,
  draftNoteId: number | string,
  host?: string | null,
): Promise<void> {
  const encoded = encodedProjectPath(owner, repo);
  const env = glabEnv(host);

  try {
    await exec(
      "glab",
      [
        "api",
        `projects/${encoded}/merge_requests/${mrNumber}/draft_notes/${draftNoteId}/publish`,
        "--method",
        "PUT",
      ],
      { env },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to publish draft note ${draftNoteId} for MR !${mrNumber}: ${msg}`);
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
    await exec(
      "glab",
      ["api", endpoint, "--method", "POST", "-H", "Content-Type: application/json", "--input", "-"],
      {
        env,
        input: JSON.stringify(payload),
      },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to post commit status for ${sha}: ${msg}`);
  }
}

/**
 * List resolvable Hodor discussions written by the publishing identity. Notes
 * from other authors are dropped before fingerprinting, deduplication, merge,
 * status, and code quality, even if they copy a Hodor marker. A fixed-reply
 * counts only when the publishing identity wrote it in the same discussion.
 */
export async function listHodorDiscussions(
  owner: string,
  repo: string,
  mrNumber: number | string,
  host: string | null | undefined,
  identity: GitlabPublisherIdentity,
): Promise<HodorDiscussion[]> {
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
    discussions = parsePaginatedJsonArrays(rawDiscussions);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to list discussions for MR !${mrNumber}: ${msg}`);
  }

  const results: HodorDiscussion[] = [];

  for (const discussion of discussions) {
    const discussionId = discussion.id;
    if (typeof discussionId !== "string") {
      continue;
    }

    const notes = discussion.notes;
    if (!Array.isArray(notes)) {
      continue;
    }

    const entries: Array<Omit<HodorDiscussion, "humanReplies" | "updatedAt">> = [];
    const humanReplies: ThreadReply[] = [];
    let updatedAt: string | undefined;
    // Notes arrive oldest first, so the latest fixed-reply wins.
    const fixedAtShaByFingerprint = new Map<string, string>();
    for (const noteObj of notes) {
      if (!isRecord(noteObj)) {
        continue;
      }
      updatedAt = latestTimestamp(updatedAt, noteObj.created_at, noteObj.updated_at, noteObj.resolved_at);
      const noteId = noteObj.id;
      const body = noteObj.body;
      if (typeof noteId !== "number" || typeof body !== "string" || noteObj.system === true) {
        continue;
      }
      if (!isPublisherNote(noteObj, identity)) {
        const author = parseGitlabAuthor(noteObj.author);
        humanReplies.push({ author: author?.username ?? author?.name ?? "unknown", body });
        continue;
      }
      if (!isHodorGeneratedNote(body)) {
        continue;
      }

      const fixed = getFixedMarker(body);
      if (fixed) {
        fixedAtShaByFingerprint.set(fixed.fingerprint, fixed.sha);
        continue;
      }

      const position = isRecord(noteObj.position) ? noteObj.position : undefined;

      const filePath =
        typeof position?.new_path === "string"
          ? position.new_path
          : typeof position?.old_path === "string"
            ? position.old_path
            : undefined;
      const line =
        typeof position?.new_line === "number"
          ? position.new_line
          : typeof position?.old_line === "number"
            ? position.old_line
            : undefined;

      // Skip non-resolvable threads. GitLab wraps the summary-comment note in a
      // discussion envelope with `resolvable: false`. Only diff/review threads
      // (resolvable: true) hold findings.
      if (noteObj.resolvable !== true) {
        continue;
      }

      const resolvedBy = parseGitlabAuthor(noteObj.resolved_by)?.username;
      entries.push({
        discussionId,
        noteId,
        body,
        resolved: Boolean(noteObj.resolved),
        filePath,
        line,
        ...(resolvedBy ? { resolvedBy } : {}),
      });
    }

    for (const entry of entries) {
      const fingerprint = getDiscussionFingerprint(entry.body);
      const fixedAtSha = fingerprint ? fixedAtShaByFingerprint.get(fingerprint) : undefined;
      results.push({
        ...entry,
        ...(fixedAtSha ? { fixedAtSha } : {}),
        ...(updatedAt ? { updatedAt } : {}),
        humanReplies,
      });
    }
  }

  return results;
}

function latestTimestamp(current: string | undefined, ...values: unknown[]): string | undefined {
  let latest = current;
  for (const value of values) {
    if (typeof value !== "string" || Number.isNaN(Date.parse(value))) continue;
    if (latest === undefined || Date.parse(value) > Date.parse(latest)) latest = value;
  }
  return latest;
}

/** Add a reply note to an existing MR discussion. Reporter access is enough. */
export async function replyToGitlabDiscussion(
  owner: string,
  repo: string,
  mrNumber: number | string,
  discussionId: string,
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
        `projects/${encoded}/merge_requests/${mrNumber}/discussions/${encodeURIComponent(discussionId)}/notes`,
        "--method",
        "POST",
        "-H",
        "Content-Type: application/json",
        "--input",
        "-",
      ],
      { env, input: JSON.stringify({ body }) },
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitLabAPIError(`Failed to reply to discussion ${discussionId} on MR !${mrNumber}: ${msg}`);
  }
}
