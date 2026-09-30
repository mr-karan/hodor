import { exec } from "./utils/exec.js";
import { logger } from "./utils/logger.js";
import { relativizeWorkspacePath } from "./utils/path.js";
import { postGiteaPrComment } from "./gitea.js";
import {
  bulkPublishGitlabDraftNotes,
  createGitlabDraftNote,
  fetchGitlabPublisherIdentity,
  getGitlabMrDiffRefs,
  gitlabNoteUrl,
  HODOR_REVIEW_MARKER,
  listHodorDiscussions,
  postGitlabCommitStatus,
  publishGitlabMrSummary,
  publishGitlabDraftNote,
  replyToGitlabDiscussion,
  type DiffRefs,
  type HodorDiscussion,
} from "./gitlab.js";
import { detectPlatform, parsePrUrl } from "./platform.js";
import { HODOR_SUMMARY_MARKER, renderMarkdown, renderSummaryMarkdown } from "./render.js";
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
): Promise<void> {
  const blocking = findings.filter((finding) => finding.priority <= 1).length;
  const state = blocking > 0 ? "failed" : "success";
  const description =
    blocking > 0
      ? `${blocking} blocking issue(s) found`
      : findings.length > 0
        ? `${findings.length} non-blocking issue(s)`
        : "No issues found";

  await postGitlabCommitStatus(
    parsed.owner,
    parsed.repo,
    diffRefs.head_sha,
    state,
    parsed.host,
    { description },
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
): Promise<string | null> {
  const { noteId, collapsed } = await publishGitlabMrSummary(
    parsed.owner,
    parsed.repo,
    parsed.prNumber,
    body,
    parsed.host,
    identity,
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
}): Promise<PostCommentResult> {
  const { prUrl, reviewText, model, metricsFooter, headSha, cacheMarker } = opts;
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
    const summaryUrl = await publishSummary(parsed, body, resolved.identity);
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
    const message = error instanceof Error ? error.message : String(error);
    logger.warn(`Failed to get diff_refs, falling back to summary mode: ${message}`);
    return postReviewComment({
      prUrl,
      reviewText: renderMarkdown(review),
      model,
      metricsFooter,
      headSha,
      cacheMarker,
    });
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

  const { open: reviewFindings, fixedAwaiting } = mergeReviewStateFindings(
    review.findings,
    discussions,
    workspacePath,
    {
      suppressResolvedCurrent: skipInline,
      resolvedFindingIds: review.resolved_findings,
    },
  );
  const currentFingerprints = new Set(
    review.findings.map((finding) => getFindingFingerprint(finding, workspacePath)),
  );

  let inlineCreated = 0;
  let inlineFailed = 0;
  let inlineDeduplicated = 0;
  const draftNoteIds: Array<number | string> = [];
  const failedFindings: ReviewFinding[] = [];
  if (reviewStyle !== "summary" && !skipInline) {
    for (const finding of review.findings) {
      const fingerprint = getFindingFingerprint(finding, workspacePath);
      if (existingByFingerprint.has(fingerprint)) {
        inlineDeduplicated++;
        continue;
      }

      const relPath = relativizeWorkspacePath(
        finding.code_location.absolute_file_path,
        workspacePath ?? undefined,
      );
      const title = /^\[P[0-3]\]/.test(finding.title)
        ? finding.title
        : `[P${finding.priority}] ${finding.title}`;
      let body = `${HODOR_REVIEW_MARKER}\n<!-- hodor:finding:${fingerprint} -->\n**${title}**\n\n${finding.body}`;

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
          {
            filePath: relPath,
            line: finding.code_location.line_range.start,
            diffRefs,
          },
        );
        if (typeof draftNote.id === "number" || typeof draftNote.id === "string") {
          draftNoteIds.push(draftNote.id);
        }
        inlineCreated++;
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

  let draftsPublished = false;
  if (inlineCreated > 0) {
    try {
      await bulkPublishGitlabDraftNotes(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        parsed.host,
      );
      draftsPublished = true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      logger.warn(`Failed to bulk publish draft notes: ${message}`);
      if (draftNoteIds.length === inlineCreated) {
        let individuallyPublished = 0;
        for (const draftNoteId of draftNoteIds) {
          try {
            await publishGitlabDraftNote(
              parsed.owner,
              parsed.repo,
              parsed.prNumber,
              draftNoteId,
              parsed.host,
            );
            individuallyPublished++;
          } catch (publishError) {
            const publishMessage =
              publishError instanceof Error ? publishError.message : String(publishError);
            errors.push(`draft publish: ${publishMessage}`);
            logger.warn(`Failed to publish draft note ${draftNoteId}: ${publishMessage}`);
          }
        }
        draftsPublished = individuallyPublished === inlineCreated;
        if (draftsPublished) {
          logger.info(`Published ${individuallyPublished} draft note(s) individually`);
        }
      } else {
        errors.push(`draft publish: ${message}`);
      }
    }
  }

  let summaryPosted = false;
  let summaryUrl: string | null = null;
  if (
    !skipSummary &&
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
      inlineCreated: reviewStyle === "summary" ? undefined : inlineCreated,
      inlineDeduplicated:
        reviewStyle === "summary" ? undefined : inlineDeduplicated,
      reviewMode,
      reviewedSha: headSha,
      carriedOver: reviewFindings.filter((finding) => !currentFingerprints.has(finding.fingerprint)).length,
      fixedAwaiting,
      asOf: new Date(),
    });
    if (headSha) summaryBody = `<!-- hodor:sha:${headSha} -->\n${summaryBody}`;
    if (cacheMarker) summaryBody = summaryBody.replace("\n", `\n${cacheMarker}\n`);
    summaryBody = appendReviewDetails(summaryBody, model, metricsFooter);
    try {
      summaryUrl = await publishSummary(parsed, summaryBody, identity);
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
      await postGitlabReviewCommitStatus(parsed, reviewFindings, diffRefs);
      commitStatusPosted = true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errors.push(`commit status: ${message}`);
      logger.warn(`Failed to post commit status: ${message}`);
    }
  }

  const fixedReplies = await replyToFixedThreads({
    parsed,
    fixedThreads: resolveFixedThreads(review.resolved_findings ?? [], discussions),
    currentFingerprints,
    headSha,
  });

  const baseDeliveryComplete =
    reviewStyle === "summary"
      ? summaryPosted || skipSummary
      : reviewStyle === "hybrid"
        ? (summaryPosted || skipSummary) &&
          (inlineCreated === 0 || draftsPublished)
        : (inlineFailed === 0 || summaryPosted) &&
          (inlineCreated === 0 || draftsPublished) &&
          (review.findings.length > 0 || summaryPosted);
  const success = baseDeliveryComplete && (!commitStatus || commitStatusPosted);

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
    fixedReplies,
    fixedAwaiting,
    reviewStateComplete: !discussionListingFailed,
    reviewFindings,
  };
}

/**
 * Reply on each thread this review verified fixed. Hodor cannot resolve
 * threads at Reporter access, so a human resolves them. A thread that already
 * has a fixed-reply gets no second one. Failures are warnings only.
 * Returns the number of replies posted.
 */
async function replyToFixedThreads(opts: {
  parsed: ParsedPrUrl;
  fixedThreads: ReadonlyMap<string, HodorDiscussion>;
  currentFingerprints: ReadonlySet<string>;
  headSha?: string | null;
}): Promise<number> {
  const { parsed, fixedThreads, currentFingerprints, headSha } = opts;
  if (fixedThreads.size === 0) return 0;
  if (!headSha || !isCommitSha(headSha)) {
    logger.warn("Skipping fixed-thread replies: no 40-character head SHA for the fixed marker");
    return 0;
  }
  let posted = 0;
  for (const [fingerprint, thread] of fixedThreads) {
    if (thread.fixedAtSha || currentFingerprints.has(fingerprint)) continue;
    try {
      await replyToGitlabDiscussion(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        thread.discussionId,
        buildFixedReplyBody(fingerprint, headSha),
        parsed.host,
      );
      posted++;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      logger.warn(`Failed to reply on fixed thread ${thread.discussionId}: ${message}`);
    }
  }
  if (posted > 0) logger.info(`Replied on ${posted} thread(s) verified fixed`);
  return posted;
}
