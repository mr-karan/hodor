import { createHash } from "node:crypto";
import type { HodorDiscussion, ThreadReply } from "./gitlab.js";
import { HODOR_REVIEW_MARKER } from "./render.js";
import type { ReviewFinding, ReviewPriority, ReviewStateFinding } from "./types.js";
import { relativizeWorkspacePath } from "./utils/path.js";

const FINDING_MARKER_RE = /<!--\s*hodor:finding:([a-f0-9]{64})\s*-->/i;
const FIXED_MARKER_RE = /<!--\s*hodor:fixed:([a-f0-9]{64}):([a-f0-9]{40})\s*-->/i;
const COMMIT_SHA_RE = /^[a-f0-9]{40}$/;
const SHORT_ID_MIN_LENGTH = 8;
const SHORT_ID_STEP = 4;
export const MAX_PROMPT_FINDING_THREADS = 15;
const MAX_THREAD_REPLIES = 2;
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

export interface FixedMarker {
  fingerprint: string;
  sha: string;
}

export function getFixedMarker(body: string): FixedMarker | null {
  const match = body.match(FIXED_MARKER_RE);
  if (!match) return null;
  return { fingerprint: match[1].toLowerCase(), sha: match[2].toLowerCase() };
}

export function isCommitSha(value: string): boolean {
  return COMMIT_SHA_RE.test(value);
}

/** The reply Hodor posts on a finding thread that a review confirmed fixed. */
export function buildFixedReplyBody(fingerprint: string, headSha: string): string {
  return (
    `${HODOR_REVIEW_MARKER}\n<!-- hodor:fixed:${fingerprint}:${headSha} -->\n` +
    `Fixed in \`${headSha.slice(0, 8)}\`. Resolve this thread if you agree.`
  );
}

/** An open Hodor finding thread the model may confirm fixed, by `id`. */
export interface FixCandidate {
  id: string;
  fingerprint: string;
  discussionId: string;
  /** Title with its priority tag, for example "[P2] Keep the schema in sync". */
  title: string;
  filePath?: string;
}

/**
 * Shortest fingerprint prefix (8, 12, 16, ... hex) that no other open thread
 * fingerprint shares. Deterministic for a given set of open threads.
 */
function getShortId(fingerprint: string, openFingerprints: ReadonlySet<string>): string {
  for (let length = SHORT_ID_MIN_LENGTH; length < fingerprint.length; length += SHORT_ID_STEP) {
    const prefix = fingerprint.slice(0, length);
    const shared = [...openFingerprints].some(
      (other) => other !== fingerprint && other.startsWith(prefix),
    );
    if (!shared) return prefix;
  }
  return fingerprint;
}

/**
 * Open Hodor threads without a fixed-reply, each with a short id. Threads
 * that share a full fingerprint with another open thread get no id: an id
 * must name exactly one thread. Ids are unique prefixes across all open
 * threads, so resolveFixedThreads maps them back without ambiguity.
 */
export function buildFixCandidates(discussions: readonly HodorDiscussion[]): FixCandidate[] {
  const threadsByFingerprint = new Map<string, HodorDiscussion[]>();
  for (const discussion of discussions) {
    if (discussion.resolved) continue;
    const fingerprint = getDiscussionFingerprint(discussion.body);
    if (!fingerprint) continue;
    const threads = threadsByFingerprint.get(fingerprint) ?? [];
    threads.push(discussion);
    threadsByFingerprint.set(fingerprint, threads);
  }

  const openFingerprints = new Set(threadsByFingerprint.keys());
  const candidates: FixCandidate[] = [];
  for (const [fingerprint, threads] of threadsByFingerprint) {
    if (threads.length !== 1) continue;
    const [thread] = threads;
    if (thread.fixedAtSha) continue;
    const finding = parseDiscussionFinding(thread);
    if (!finding) continue;
    candidates.push({
      id: getShortId(fingerprint, openFingerprints),
      fingerprint,
      discussionId: thread.discussionId,
      title: finding.title,
      filePath: thread.filePath,
    });
  }
  return candidates;
}

export type FindingThreadStatus = "open" | "fixed" | "resolved";

/** One Hodor finding thread as the review prompt shows it. */
export interface FindingThread {
  /** Set only when the model may list this thread in resolved_findings. */
  fixId?: string;
  title: string;
  filePath?: string;
  status: FindingThreadStatus;
  resolvedBy?: string;
  /** Up to the two most recent non-trivial human replies, oldest first. */
  replies: ThreadReply[];
}

/** A reply with no letters or digits (emoji, punctuation), or a bare +1/-1. */
function isTrivialReply(body: string): boolean {
  const text = body.trim();
  return /^[+-]1$/.test(text) || !/[\p{L}\p{N}]/u.test(text);
}

function threadRank(thread: FindingThread): number {
  if (thread.status === "open") return thread.fixId ? 0 : 1;
  return thread.status === "fixed" ? 2 : 3;
}

/**
 * Hodor finding threads for the review prompt, open and resolved, with their
 * latest human replies. Open threads come first; resolved threads follow,
 * newest first. A thread gets a fix id only when it is a candidate and its
 * file is in the reviewed diff.
 */
export function selectFindingThreads(
  discussions: readonly HodorDiscussion[],
  candidates: readonly FixCandidate[],
  changedFiles: readonly string[],
  limit = MAX_PROMPT_FINDING_THREADS,
): FindingThread[] {
  const candidatesByDiscussion = new Map(candidates.map((candidate) => [candidate.discussionId, candidate]));
  const changed = new Set(changedFiles);
  const threads: Array<{ thread: FindingThread; updatedAt: number }> = [];
  for (const discussion of discussions) {
    const finding = parseDiscussionFinding(discussion);
    if (!finding) continue;
    const candidate = candidatesByDiscussion.get(discussion.discussionId);
    const status: FindingThreadStatus = discussion.resolved
      ? "resolved"
      : discussion.fixedAtSha ? "fixed" : "open";
    const fixId =
      candidate && candidate.filePath && changed.has(candidate.filePath) ? candidate.id : undefined;
    threads.push({
      thread: {
        ...(fixId ? { fixId } : {}),
        title: finding.title,
        filePath: discussion.filePath,
        status,
        ...(status === "resolved" && discussion.resolvedBy ? { resolvedBy: discussion.resolvedBy } : {}),
        replies: discussion.humanReplies
          .filter((reply) => !isTrivialReply(reply.body))
          .slice(-MAX_THREAD_REPLIES),
      },
      updatedAt: Date.parse(discussion.updatedAt ?? "") || 0,
    });
  }
  return threads
    .sort((a, b) => threadRank(a.thread) - threadRank(b.thread) || b.updatedAt - a.updatedAt)
    .slice(0, limit)
    .map(({ thread }) => thread);
}

export type FixRejectionReason =
  | "unknown id"
  | "duplicate id"
  | "file not in the reviewed diff"
  | "re-reported in this review";

/**
 * Keep only ids that name a candidate shown in this run, whose thread file is
 * in the reviewed diff, and whose finding this review did not report again.
 */
export function selectVerifiedFixes(
  ids: readonly string[],
  candidates: readonly FixCandidate[],
  context: {
    changedFiles: readonly string[];
    currentFingerprints: ReadonlySet<string>;
  },
): { accepted: string[]; rejected: Array<{ id: string; reason: FixRejectionReason }> } {
  const candidatesById = new Map(candidates.map((candidate) => [candidate.id, candidate]));
  const changedFiles = new Set(context.changedFiles);
  const accepted: string[] = [];
  const rejected: Array<{ id: string; reason: FixRejectionReason }> = [];
  for (const id of ids) {
    const candidate = candidatesById.get(id);
    if (!candidate) {
      rejected.push({ id, reason: "unknown id" });
    } else if (accepted.includes(id)) {
      rejected.push({ id, reason: "duplicate id" });
    } else if (!candidate.filePath || !changedFiles.has(candidate.filePath)) {
      rejected.push({ id, reason: "file not in the reviewed diff" });
    } else if (context.currentFingerprints.has(candidate.fingerprint)) {
      rejected.push({ id, reason: "re-reported in this review" });
    } else {
      accepted.push(id);
    }
  }
  return { accepted, rejected };
}

/**
 * Map verified short ids to the open thread each one names. An id that
 * matches no open thread, or more than one, is skipped.
 */
export function resolveFixedThreads(
  ids: readonly string[],
  discussions: readonly HodorDiscussion[],
): Map<string, HodorDiscussion> {
  const resolved = new Map<string, HodorDiscussion>();
  for (const id of ids) {
    const matches = discussions.filter((discussion) =>
      !discussion.resolved &&
      getDiscussionFingerprint(discussion.body)?.startsWith(id.toLowerCase()) === true,
    );
    if (matches.length !== 1) continue;
    const [thread] = matches;
    const fingerprint = getDiscussionFingerprint(thread.body);
    if (fingerprint) resolved.set(fingerprint, thread);
  }
  return resolved;
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

export interface ReviewState {
  /** Findings that still count as open: current ones plus open threads not confirmed fixed. */
  open: ReviewStateFinding[];
  /** Open threads confirmed fixed and waiting for a human to resolve them. */
  fixedAwaiting: number;
}

/**
 * An open thread is fixed and waiting to be resolved when it has a trusted
 * fixed-reply or this review verified it fixed, unless this review reported
 * the same finding again. A re-reported finding is open, even after an older
 * fixed-reply.
 */
export function mergeReviewStateFindings(
  currentFindings: ReviewFinding[],
  discussions: HodorDiscussion[],
  workspacePath?: string | null,
  options: {
    suppressResolvedCurrent?: boolean;
    /** Verified ids from this review's resolved_findings. */
    resolvedFindingIds?: readonly string[];
  } = {},
): ReviewState {
  const { suppressResolvedCurrent = false, resolvedFindingIds = [] } = options;
  const merged = new Map<string, ReviewStateFinding>();
  const openFingerprints = new Set<string>();
  const resolvedFingerprints = new Set<string>();
  const repliedFixedFingerprints = new Set<string>();
  for (const discussion of discussions) {
    const fingerprint = getDiscussionFingerprint(discussion.body);
    if (!fingerprint) continue;
    if (discussion.resolved) {
      resolvedFingerprints.add(fingerprint);
    } else {
      openFingerprints.add(fingerprint);
      if (discussion.fixedAtSha) repliedFixedFingerprints.add(fingerprint);
    }
  }

  const currentFingerprints = new Set<string>();
  for (const finding of currentFindings) {
    const fingerprint = getFindingFingerprint(finding, workspacePath);
    currentFingerprints.add(fingerprint);
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

  const fixedFingerprints = new Set(
    [...repliedFixedFingerprints, ...resolveFixedThreads(resolvedFindingIds, discussions).keys()]
      .filter((fingerprint) => !currentFingerprints.has(fingerprint)),
  );

  for (const discussion of discussions) {
    if (discussion.resolved) continue;
    const finding = parseDiscussionFinding(discussion);
    if (finding && !merged.has(finding.fingerprint) && !fixedFingerprints.has(finding.fingerprint)) {
      merged.set(finding.fingerprint, finding);
    }
  }

  return { open: [...merged.values()], fixedAwaiting: fixedFingerprints.size };
}
