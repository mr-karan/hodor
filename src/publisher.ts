import { randomUUID } from "node:crypto";
import { exec } from "./utils/exec.js";
import { logger } from "./utils/logger.js";
import { relativizeWorkspacePath } from "./utils/path.js";
import { postGiteaPrComment } from "./gitea.js";
import {
  createGitlabDraftNote,
  fetchGitlabDiffFiles,
  fetchGitlabPublisherIdentity,
  getGitlabMrDiffRefs,
  gitlabNoteUrl,
  HODOR_REVIEW_MARKER,
  listHodorDiscussions,
  listRecentGitlabDiffNotes,
  postGitlabCommitStatus,
  publishGitlabMrSummary,
  publishGitlabDraftNote,
  replyToGitlabDiscussion,
  type DiffRefs,
  type HodorDiscussion,
  type PersistedDiffNote,
  type PersistedDiffPosition,
} from "./gitlab.js";
import { findLinePosition, type GitlabFileDiff, type GitlabLinePosition } from "./gitlab-diff-position.js";
import { detectPlatform, parsePrUrl } from "./platform.js";
import { HODOR_SUMMARY_MARKER, renderMarkdown, renderSummaryMarkdown } from "./render.js";
import { belongsToReviewKey, leadingMarkerLines, reviewCheckName, withReviewKey, type ReviewKey } from "./review-key.js";
import {
  buildFixedReplyBody,
  getDiscussionFingerprint,
  getFindingFingerprint,
  isCommitSha,
  mergeReviewStateFindings,
  resolveFixedThreads,
} from "./review-state.js";
import type {
  GitlabPublisherIdentity,
  ParsedPrUrl,
  PostCommentResult,
  ReviewFinding,
  ReviewMetrics,
  ReviewOutput,
  ReviewStateFinding,
} from "./types.js";


export async function postGitlabReviewCommitStatus(
  parsed: ParsedPrUrl,
  findings: ReviewStateFinding[],
  diffRefs: DiffRefs,
  reviewKey: ReviewKey | null = null,
): Promise<void> {
  const blocking = findings.filter((finding) => finding.priority <= 1).length;
  const state = blocking > 0 ? "failed" : "success";
  const description =
    blocking > 0
      ? `${blocking} blocking Hodor finding(s) unresolved`
      : findings.length > 0
        ? `${findings.length} non-blocking Hodor finding(s) unresolved`
        : "No issues found";

  await postGitlabCommitStatus(
    parsed.owner,
    parsed.repo,
    diffRefs.head_sha,
    state,
    parsed.host,
    { name: reviewCheckName(reviewKey), description },
  );
}

/**
 * GitLab publication edits Hodor's own notes and reads Hodor's own
 * discussions, so it needs the publishing identity. Without one, Hodor posts
 * nothing: it cannot tell its notes from forged ones.
 */
async function resolveGitlabIdentityForPosting(
  parsed: ParsedPrUrl,
): Promise<{ identity: GitlabPublisherIdentity } | { error: string }> {
  try {
    return { identity: await fetchGitlabPublisherIdentity(parsed.host) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const text = `Refusing to post to MR !${parsed.prNumber}: cannot resolve the GitLab publishing identity (${message})`;
    logger.error(text);
    return { error: text };
  }
}

/** Post the summary note and return its web URL, or null when GitLab gave no id. */
async function publishSummary(
  parsed: ParsedPrUrl,
  body: string,
  identity: GitlabPublisherIdentity,
  reviewKey: ReviewKey | null,
): Promise<string | null> {
  const { noteId, collapsed } = await publishGitlabMrSummary(
    parsed.owner,
    parsed.repo,
    parsed.prNumber,
    body,
    parsed.host,
    identity,
    reviewKey,
  );
  logger.info(
    `Posted summary note${noteId === null ? "" : ` ${noteId}`}; collapsed ${collapsed} older summary note(s)`,
  );
  return noteId === null
    ? null
    : gitlabNoteUrl(parsed.owner, parsed.repo, parsed.prNumber, noteId, parsed.host);
}

function appendReviewDetails(
  body: string,
  model?: string | null,
  metricsFooter?: string | null,
): string {
  if (!model && !metricsFooter) return body;

  const details = ["<details>", "<summary>Review details</summary>", ""];
  if (model) details.push(`- Model: \`${displayModel(model)}\``);
  if (metricsFooter) {
    if (model) details.push("");
    details.push(metricsFooter);
  }
  details.push("", "</details>");
  return `${body.trimEnd()}\n\n${details.join("\n")}\n`;
}

/** The model id without provider routing, or the profile name of a Bedrock ARN. */
export function displayModel(model: string): string {
  const baseModel = model.slice(model.lastIndexOf("@") + 1);
  return baseModel.startsWith("arn:")
    ? baseModel.slice(baseModel.lastIndexOf("/") + 1)
    : baseModel;
}

export async function postReviewComment(opts: {
  prUrl: string;
  reviewText: string;
  model?: string | null;
  metricsFooter?: string | null;
  headSha?: string | null;
  cacheMarker?: string | null;
  reviewKey?: ReviewKey | null;
}): Promise<PostCommentResult> {
  const { prUrl, reviewText, model, metricsFooter, headSha, cacheMarker, reviewKey = null } = opts;
  const platform = detectPlatform(prUrl);
  const parsed = parsePrUrl(prUrl);
  let body = reviewText;
  if (platform === "gitlab" && !body.includes(HODOR_SUMMARY_MARKER)) {
    body = body.replace(
      HODOR_REVIEW_MARKER,
      `${HODOR_REVIEW_MARKER}\n${HODOR_SUMMARY_MARKER}`,
    );
  }
  if (headSha) body = `<!-- hodor:sha:${headSha} -->\n${body}`;
  if (cacheMarker) body = body.replace("\n", `\n${cacheMarker}\n`);
  body = withReviewKey(body, reviewKey);
  body = appendReviewDetails(body, model, metricsFooter);

  try {
    if (platform === "github") {
      await exec("gh", [
        "pr",
        "review",
        String(parsed.prNumber),
        "--repo",
        `${parsed.owner}/${parsed.repo}`,
        "--comment",
        "--body",
        body,
      ]);
      return { success: true, platform, prNumber: parsed.prNumber };
    }
    if (platform === "gitea") {
      await postGiteaPrComment(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        body,
        parsed.host,
      );
      return { success: true, platform, prNumber: parsed.prNumber };
    }

    const resolved = await resolveGitlabIdentityForPosting(parsed);
    if ("error" in resolved) {
      return { success: false, platform, error: resolved.error };
    }
    const summaryUrl = await publishSummary(parsed, body, resolved.identity, reviewKey);
    return {
      success: true,
      platform,
      mrNumber: parsed.prNumber,
      summaryPosted: true,
      ...(summaryUrl ? { summaryUrl } : {}),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    logger.error(`Failed to post comment: ${message}`);
    return { success: false, platform, error: message };
  }
}

export async function postReviewStructured(opts: {
  prUrl: string;
  review: ReviewOutput;
  model?: string | null;
  metricsFooter?: string | null;
  reviewStyle?: "summary" | "inline" | "hybrid";
  commitStatus?: boolean;
  headSha?: string | null;
  workspacePath?: string | null;
  cacheMarker?: string | null;
  skipSummary?: boolean;
  skipInline?: boolean;
  reviewMode?: ReviewMetrics["reviewMode"];
  reviewKey?: ReviewKey | null;
}): Promise<PostCommentResult> {
  const {
    prUrl,
    review,
    model,
    metricsFooter,
    reviewStyle = "hybrid",
    commitStatus = false,
    headSha,
    workspacePath,
    cacheMarker,
    skipSummary = false,
    skipInline = false,
    reviewMode,
    reviewKey = null,
  } = opts;

  const platform = detectPlatform(prUrl);
  if (platform !== "gitlab") {
    return postReviewComment({
      prUrl,
      reviewText: renderMarkdown(review),
      model,
      metricsFooter,
      headSha,
      cacheMarker,
      reviewKey,
    });
  }

  const parsed = parsePrUrl(prUrl);
  const resolved = await resolveGitlabIdentityForPosting(parsed);
  if ("error" in resolved) {
    return {
      success: false,
      platform: "gitlab",
      mrNumber: parsed.prNumber,
      error: resolved.error,
      errors: [resolved.error],
    };
  }
  const { identity } = resolved;
  const errors: string[] = [];
  let diffRefs: DiffRefs;
  try {
    diffRefs = await getGitlabMrDiffRefs(
      parsed.owner,
      parsed.repo,
      parsed.prNumber,
      parsed.host,
    );
  } catch (error) {
    const message = `diff refs: ${error instanceof Error ? error.message : String(error)}`;
    logger.warn(`${message}; posting the summary without completion markers`);
    const fallback = await postReviewComment({
      prUrl,
      reviewText: renderMarkdown(review),
      model,
      metricsFooter,
      reviewKey,
    });
    return {
      ...fallback,
      success: false,
      error: message,
      errors: [message, ...(fallback.error ? [fallback.error] : [])],
    };
  }

  const existingByFingerprint = new Map<string, Set<string>>();
  let discussions: HodorDiscussion[] = [];
  let discussionListingFailed = false;
  try {
    discussions = await listHodorDiscussions(
      parsed.owner,
      parsed.repo,
      parsed.prNumber,
      parsed.host,
      identity,
      reviewKey,
    );
    for (const discussion of discussions) {
      if (discussion.resolved) continue;
      const fingerprint = getDiscussionFingerprint(discussion.body);
      if (!fingerprint) continue;
      const ids = existingByFingerprint.get(fingerprint) ?? new Set<string>();
      ids.add(discussion.discussionId);
      existingByFingerprint.set(fingerprint, ids);
    }
  } catch (error) {
    discussionListingFailed = true;
    const message = error instanceof Error ? error.message : String(error);
    if (commitStatus) {
      errors.push(`discussion listing: ${message}`);
    }
    logger.warn(`Failed to list open Hodor discussions for review state: ${message}`);
  }

  const currentFingerprints = new Set(
    review.findings.map((finding) => getFindingFingerprint(finding, workspacePath)),
  );
  const fixedReplyResult = await replyToFixedThreads({
    parsed,
    fixedThreads: resolveFixedThreads(review.resolved_findings ?? [], discussions),
    currentFingerprints,
    headSha,
    reviewKey,
  });
  errors.push(...fixedReplyResult.errors);
  const { open: reviewFindings, fixedAwaiting } = mergeReviewStateFindings(
    review.findings,
    discussions,
    workspacePath,
    {
      suppressResolvedCurrent: skipInline,
      resolvedFindingIds: [...fixedReplyResult.persisted],
    },
  );
  // Cached reviews still refresh summaries when mutable thread state exists.
  const shouldSkipSummary = skipSummary && discussions.length === 0 && !discussionListingFailed;
  const carriedFindings = reviewFindings.filter((finding) => !currentFingerprints.has(finding.fingerprint));
  const earlierThreads = carriedFindings.flatMap((finding) => {
    const thread = discussions.find((discussion) =>
      !discussion.resolved && getDiscussionFingerprint(discussion.body) === finding.fingerprint,
    );
    return thread ? [{
      title: finding.title,
      url: gitlabNoteUrl(parsed.owner, parsed.repo, parsed.prNumber, thread.noteId, parsed.host),
    }] : [];
  });

  let inlineCreated = 0;
  let inlineFailed = 0;
  let inlineDeduplicated = 0;
  let diffError: string | null = null;
  const ownedDrafts: OwnedDraft[] = [];
  const failedFindings: ReviewFinding[] = [];
  if (reviewStyle !== "summary" && !skipInline) {
    const newFindings = review.findings.filter(
      (finding) => !existingByFingerprint.has(getFindingFingerprint(finding, workspacePath)),
    );
    inlineDeduplicated = review.findings.length - newFindings.length;
    let diffFiles: GitlabFileDiff[] = [];
    if (newFindings.length > 0) {
      try {
        diffFiles = await fetchGitlabDiffFiles(parsed.owner, parsed.repo, parsed.prNumber, parsed.host, diffRefs);
      } catch (error) {
        // Unlike an unanchorable line, a failed diff read says nothing about the findings,
        // so inline delivery stays incomplete and a rerun at this HEAD tries again.
        diffError = error instanceof Error ? error.message : String(error);
        const message = `inline notes: cannot load the MR diff (${diffError})`;
        errors.push(message);
        logger.warn(`${message}; delivering ${newFindings.length} finding(s) in the summary only`);
        inlineFailed += newFindings.length;
        failedFindings.push(...newFindings);
      }
    }
    for (const finding of diffError === null ? newFindings : []) {
      const fingerprint = getFindingFingerprint(finding, workspacePath);
      const relPath = relativizeWorkspacePath(
        finding.code_location.absolute_file_path,
        workspacePath ?? undefined,
      );
      const located = findLinePosition(diffFiles, relPath, finding.code_location.line_range);
      if (!located.ok) {
        // The summary carries findings GitLab cannot anchor inline (docs/AUTOMATED_REVIEWS.md).
        const message = `inline note for ${finding.title}: no GitLab diff position (${located.reason})`;
        errors.push(message);
        logger.warn(`${message}; delivering it in the summary only`);
        inlineFailed++;
        failedFindings.push(finding);
        continue;
      }

      const title = /^\[P[0-3]\]/.test(finding.title)
        ? finding.title
        : `[P${finding.priority}] ${finding.title}`;
      // Publication returns no note, so this marker is how the readback finds this draft's note.
      const deliveryMarker = `<!-- hodor:delivery:${randomUUID()} -->`;
      let body = withReviewKey(
        `${HODOR_REVIEW_MARKER}\n<!-- hodor:finding:${fingerprint} -->\n${deliveryMarker}\n**${title}**\n\n${finding.body}`,
        reviewKey,
      );

      if (finding.suggestion) {
        const { start, end } = finding.code_location.line_range;
        const span = Math.max(0, end - start);
        body += `\n\n\`\`\`suggestion:-0+${span}\n${finding.suggestion}\n\`\`\``;
      }

      try {
        const draftNote = await createGitlabDraftNote(
          parsed.owner,
          parsed.repo,
          parsed.prNumber,
          body,
          parsed.host,
          { position: located.position, diffRefs },
        );
        inlineCreated++;
        if (draftNote.id !== null) {
          ownedDrafts.push({ id: draftNote.id, finding, fingerprint, deliveryMarker, position: located.position });
        } else {
          const message = `draft publish: created draft note for ${finding.title} has no valid id`;
          errors.push(message);
          logger.warn(message);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        errors.push(`inline note for ${finding.title}: ${message}`);
        logger.warn(`Failed to create inline note for "${finding.title}": ${message}`);
        inlineFailed++;
        failedFindings.push(finding);
      }
    }
  }

  logger.info(
    `Created ${inlineCreated} inline draft note(s)` +
      `${inlineDeduplicated > 0 ? ` (${inlineDeduplicated} already open)` : ""}` +
      `${inlineFailed > 0 ? ` (${inlineFailed} failed)` : ""}`,
  );

  // GitLab bulk_publish publishes all pending drafts of this user, including other runs.
  const publishedDrafts: OwnedDraft[] = [];
  for (const draft of ownedDrafts) {
    try {
      await publishGitlabDraftNote(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        draft.id,
        parsed.host,
      );
      publishedDrafts.push(draft);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`draft publish: ${message}`);
      logger.warn(`Failed to publish draft note ${draft.id}: ${message}`);
    }
  }
  if (publishedDrafts.length > 0) {
    logger.info(`Requested publication of ${publishedDrafts.length} draft note(s) individually; verifying the persisted notes`);
  }
  const unverified = await verifyPersistedDrafts(parsed, identity, publishedDrafts, { diffRefs, reviewKey }, errors);
  failedFindings.push(...unverified.map((draft) => draft.finding));
  const inlinePublished = publishedDrafts.length - unverified.length;
  const draftsPublished = inlineCreated > 0 && inlinePublished === inlineCreated;
  const inlineDeliveryComplete = diffError === null && (inlineCreated === 0 || draftsPublished);

  let summaryPosted = false;
  let summaryUrl: string | null = null;
  if (
    !shouldSkipSummary &&
    (
      reviewStyle === "summary" ||
      reviewStyle === "hybrid" ||
      review.findings.length === 0 ||
      failedFindings.length > 0
    )
  ) {
    const summaryFindings = review.findings;
    let summaryBody = renderSummaryMarkdown(review, {
      openFindings: reviewFindings,
      fallbackFindings: summaryFindings,
      fallbackHeading:
        reviewMode === "incremental" ? "Findings from this review" : "Findings",
      inlineCreated: reviewStyle === "summary" ? undefined : inlinePublished,
      inlineDeduplicated:
        reviewStyle === "summary" ? undefined : inlineDeduplicated,
      reviewMode,
      reviewedSha: headSha,
      reviewKey,
      carriedOver: carriedFindings.length,
      earlierThreads,
      fixedReplyFailures: fixedReplyResult.errors.length,
      fixedAwaiting,
      asOf: new Date(),
    });
    // Neither cached reuse nor an empty incremental diff may skip undelivered drafts.
    if (inlineDeliveryComplete) {
      if (headSha) summaryBody = `<!-- hodor:sha:${headSha} -->\n${summaryBody}`;
      if (cacheMarker) summaryBody = summaryBody.replace("\n", `\n${cacheMarker}\n`);
    }
    summaryBody = withReviewKey(summaryBody, reviewKey);
    summaryBody = appendReviewDetails(summaryBody, model, metricsFooter);
    try {
      summaryUrl = await publishSummary(parsed, summaryBody, identity, reviewKey);
      summaryPosted = true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`summary comment: ${message}`);
      logger.warn(`Failed to post summary comment: ${message}`);
    }
  }

  let commitStatusPosted = false;
  if (commitStatus && !discussionListingFailed) {
    try {
      await postGitlabReviewCommitStatus(parsed, reviewFindings, diffRefs, reviewKey);
      commitStatusPosted = true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`commit status: ${message}`);
      logger.warn(`Failed to post commit status: ${message}`);
    }
  }

  const baseDeliveryComplete =
    reviewStyle === "summary"
      ? summaryPosted || shouldSkipSummary
      : reviewStyle === "hybrid"
        ? (summaryPosted || shouldSkipSummary) && inlineDeliveryComplete
        : (inlineFailed === 0 || summaryPosted) &&
          inlineDeliveryComplete &&
          (review.findings.length > 0 || summaryPosted);
  const success = baseDeliveryComplete && (!commitStatus || commitStatusPosted) && fixedReplyResult.errors.length === 0;

  return {
    success,
    platform: "gitlab",
    mrNumber: parsed.prNumber,
    error: success ? undefined : errors[0] ?? "Review delivery was incomplete",
    errors,
    summaryPosted,
    ...(summaryUrl ? { summaryUrl } : {}),
    inlineCreated,
    inlineFailed,
    draftsPublished,
    commitStatusPosted,
    fixedReplies: fixedReplyResult.posted,
    fixedAwaiting,
    reviewStateComplete: !discussionListingFailed,
    reviewFindings,
  };
}

/** A draft this invocation created, with what its published note must match. */
interface OwnedDraft {
  id: number;
  finding: ReviewFinding;
  fingerprint: string;
  deliveryMarker: string;
  position: GitlabLinePosition;
}

/**
 * GitLab can answer a draft publish with success yet persist no note, so a
 * draft counts as delivered only when exactly one diff note by the publishing
 * identity carries its delivery marker, and that note's leading markers name
 * this review key and finding at the requested position of the same diff
 * version. Returns the drafts that could not be verified and records why in
 * `errors`. Unverified means not confirmed, not proven absent.
 */
async function verifyPersistedDrafts(
  parsed: ParsedPrUrl,
  identity: GitlabPublisherIdentity,
  drafts: readonly OwnedDraft[],
  expected: { diffRefs: DiffRefs; reviewKey: ReviewKey | null },
  errors: string[],
): Promise<OwnedDraft[]> {
  if (drafts.length === 0) return [];
  let notes: PersistedDiffNote[];
  try {
    notes = await listRecentGitlabDiffNotes(parsed.owner, parsed.repo, parsed.prNumber, parsed.host, identity);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    errors.push(`inline note readback: ${message}`);
    logger.warn(`Cannot verify ${drafts.length} published inline note(s): ${message}`);
    return [...drafts];
  }
  const unverified = drafts.filter((draft) => {
    const problem = deliveryProblem(draft, notes, expected);
    if (problem === null) return false;
    const message = `inline note for ${draft.finding.title}: unverified after publishing draft ${draft.id} (${problem})`;
    errors.push(message);
    logger.warn(message);
    return true;
  });
  logger.info(`Verified ${drafts.length - unverified.length} of ${drafts.length} published inline note(s)`);
  return unverified;
}

function deliveryProblem(
  draft: OwnedDraft,
  notes: readonly PersistedDiffNote[],
  expected: { diffRefs: DiffRefs; reviewKey: ReviewKey | null },
): string | null {
  // Any note quoting the marker counts, so a copy elsewhere fails verification instead of hiding.
  const matches = notes.filter((note) => note.body.includes(draft.deliveryMarker));
  if (matches.length === 0) return "no note with its delivery marker among the newest notes read back";
  if (matches.length > 1) return `${matches.length} notes carry its delivery marker`;
  const [note] = matches;
  if (note.id === null || note.position === null) return "the note has a malformed id or position";
  const envelope = leadingMarkerLines(note.body).map((line) => line.trim());
  if (
    !envelope.includes(HODOR_REVIEW_MARKER) ||
    !envelope.includes(draft.deliveryMarker) ||
    getDiscussionFingerprint(envelope.join("\n")) !== draft.fingerprint ||
    !belongsToReviewKey(note.body, expected.reviewKey)
  ) {
    return "the note's leading Hodor markers do not name this review key and finding";
  }
  if (!atPosition(note.position, draft.position, expected.diffRefs)) {
    return "the note is not at the requested position of this diff version";
  }
  return null;
}

/** Exact refs: a note GitLab moved to another diff version does not verify. */
function atPosition(actual: PersistedDiffPosition, position: GitlabLinePosition, diffRefs: DiffRefs): boolean {
  const { anchor } = position;
  return (
    actual.positionType === "text" &&
    actual.diffRefs.base_sha === diffRefs.base_sha &&
    actual.diffRefs.start_sha === diffRefs.start_sha &&
    actual.diffRefs.head_sha === diffRefs.head_sha &&
    actual.oldPath === position.oldPath &&
    actual.newPath === position.newPath &&
    actual.newLine === anchor.newLine &&
    actual.oldLine === (anchor.kind === "context" ? anchor.oldLine : null)
  );
}

/**
 * Reply on each thread this review verified fixed. Hodor cannot resolve
 * threads at Reporter access, so a human resolves them. Return the fingerprints
 * persisted in trusted replies and report incomplete delivery explicitly.
 */
async function replyToFixedThreads(opts: {
  parsed: ParsedPrUrl;
  fixedThreads: ReadonlyMap<string, HodorDiscussion>;
  currentFingerprints: ReadonlySet<string>;
  headSha?: string | null;
  reviewKey: ReviewKey | null;
}): Promise<{ posted: number; persisted: Set<string>; errors: string[] }> {
  const { parsed, fixedThreads, currentFingerprints, headSha, reviewKey } = opts;
  const persisted = new Set<string>();
  const errors: string[] = [];
  let posted = 0;
  for (const [fingerprint, thread] of fixedThreads) {
    if (currentFingerprints.has(fingerprint)) continue;
    if (thread.fixedAtSha) {
      persisted.add(fingerprint);
      continue;
    }
    if (!headSha || !isCommitSha(headSha)) {
      errors.push(`fixed reply for ${thread.discussionId}: no 40-character head SHA`);
      continue;
    }
    try {
      await replyToGitlabDiscussion(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        thread.discussionId,
        withReviewKey(buildFixedReplyBody(fingerprint, headSha), reviewKey),
        parsed.host,
      );
      posted++;
      persisted.add(fingerprint);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`fixed reply for ${thread.discussionId}: ${message}`);
      logger.warn(`Failed to reply on fixed thread ${thread.discussionId}: ${message}`);
    }
  }
  if (posted > 0) logger.info(`Replied on ${posted} thread(s) verified fixed`);
  return { posted, persisted, errors };
}
