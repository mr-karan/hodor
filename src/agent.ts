import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import type { Api, Model } from "@earendil-works/pi-ai";
import { logger } from "./utils/logger.js";
import { exec } from "./utils/exec.js";
import {
  fetchGithubPrInfo,
  normalizeGithubMetadata,
} from "./github.js";
import {
  fetchGitlabMrInfo,
  postGitlabMrComment,
  getGitlabMrDiffRefs,
  cleanupHodorComments,
  createGitlabDraftNote,
  bulkPublishGitlabDraftNotes,
  postGitlabCommitStatus,
  listHodorDiscussions,
  resolveGitlabDiscussions,
  HODOR_REVIEW_MARKER,
} from "./gitlab.js";
import {
  fetchGiteaPrInfo,
  postGiteaPrComment,
} from "./gitea.js";
import type { DiffRefs } from "./gitlab.js";
import { setupWorkspace, cleanupWorkspace } from "./workspace.js";
import { buildPrReviewPrompt } from "./prompt.js";
import { parseModelString, mapReasoningEffort } from "./model.js";
import { formatMetricsMarkdown, printMetrics } from "./metrics.js";
import { SUBMIT_REVIEW_SCHEMA, validateReviewOutput } from "./review.js";
import { REVIEW_SYSTEM_PROMPT } from "./system-prompt.js";
import { renderMarkdown, renderSummaryMarkdown } from "./render.js";
import { relativizeWorkspacePath } from "./utils/path.js";
import type {
  Platform,
  ParsedPrUrl,
  ReviewMetrics,
  PostCommentResult,
  MrMetadata,
  ReviewOutput,
} from "./types.js";

export interface AgentProgressEvent {
  type: "tool_start" | "tool_end" | "thinking" | "turn_start" | "turn_end" | "agent_start" | "agent_end" | "text_delta" | "thinking_delta" | "tool_result";
  toolName?: string;
  toolArgs?: string;
  isError?: boolean;
  turnIndex?: number;
  delta?: string;
  result?: string;
}

export function detectPlatform(prUrl: string): Platform {
  const url = new URL(prUrl);
  const hostname = url.hostname;
  if (prUrl.includes("/-/merge_requests/") || hostname.includes("gitlab")) {
    return "gitlab";
  }
  // Gitea/Forgejo: /pulls/ (plural) — must check before GitHub since /pulls/ contains /pull/
  if (prUrl.includes("/pulls/") || hostname.includes("gitea") || hostname.includes("forgejo") || hostname.includes("codeberg")) {
    return "gitea";
  }
  if (prUrl.includes("/pull/") || hostname.includes("github")) {
    return "github";
  }
  throw new Error(
    `Cannot detect platform for URL: ${prUrl}. Expected a GitHub (/pull/), GitLab (/-/merge_requests/), or Gitea/Forgejo (/pulls/) URL.`,
  );
}

export function parsePrUrl(prUrl: string): ParsedPrUrl {
  const url = new URL(prUrl);
  const pathParts = url.pathname.split("/").filter(Boolean);
  const host = url.host;

  // GitHub format: /owner/repo/pull/123
  if (pathParts.length >= 4 && pathParts[2] === "pull") {
    const prNumber = parseInt(pathParts[3], 10);
    if (!Number.isSafeInteger(prNumber) || prNumber <= 0) {
      throw new Error(`Invalid PR number in URL: ${prUrl}. Expected a positive integer after /pull/.`);
    }
    return {
      owner: pathParts[0],
      repo: pathParts[1],
      prNumber,
      host,
    };
  }

  // Gitea/Forgejo format: /owner/repo/pulls/123
  if (pathParts.length >= 4 && pathParts[2] === "pulls") {
    const prNumber = parseInt(pathParts[3], 10);
    if (!Number.isSafeInteger(prNumber) || prNumber <= 0) {
      throw new Error(`Invalid PR number in URL: ${prUrl}. Expected a positive integer after /pulls/.`);
    }
    return {
      owner: pathParts[0],
      repo: pathParts[1],
      prNumber,
      host,
    };
  }

  // GitLab format: /group/subgroup/repo/-/merge_requests/123
  const mrIndex = pathParts.indexOf("merge_requests");
  if (mrIndex >= 0) {
    if (mrIndex < 2 || mrIndex + 1 >= pathParts.length) {
      throw new Error(
        `Invalid GitLab MR URL format: ${prUrl}. Expected .../-/merge_requests/<number>`,
      );
    }
    if (pathParts[mrIndex - 1] !== "-") {
      throw new Error(
        `Invalid GitLab MR URL format: ${prUrl}. Missing '/-/' segment before merge_requests.`,
      );
    }

    const repo = pathParts[mrIndex - 2];
    const ownerParts = pathParts.slice(0, mrIndex - 2);
    const owner =
      ownerParts.length > 0 ? ownerParts.join("/") : pathParts[0];
    const prNumber = parseInt(pathParts[mrIndex + 1], 10);
    if (!Number.isSafeInteger(prNumber) || prNumber <= 0) {
      throw new Error(`Invalid MR number in URL: ${prUrl}. Expected a positive integer after /merge_requests/.`);
    }
    return { owner, repo, prNumber, host };
  }

  throw new Error(
    `Invalid PR/MR URL format: ${prUrl}. Expected GitHub (/pull/), GitLab (/-/merge_requests/), or Gitea/Forgejo (/pulls/) URL.`,
  );
}

function formatLocationRelative(
  loc: { absolute_file_path: string },
  workspacePath?: string | null,
): string {
  return relativizeWorkspacePath(loc.absolute_file_path, workspacePath ?? undefined);
}

/**
 * Post a pass/fail commit status to a GitLab MR head SHA based on review priorities.
 * Findings with priority <= 1 (P0/P1) are treated as blocking.
 */
export async function postGitlabReviewCommitStatus(
  parsed: ParsedPrUrl,
  review: ReviewOutput,
  diffRefs: DiffRefs,
): Promise<void> {
  const blocking = review.findings.filter((f) => f.priority <= 1).length;
  const state = blocking > 0 ? "failed" : "success";
  const description =
    blocking > 0
      ? `${blocking} blocking issue(s) found`
      : review.findings.length > 0
        ? `${review.findings.length} non-blocking issue(s)`
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

export async function postReviewComment(opts: {
  prUrl: string;
  reviewText: string;
  model?: string | null;
  metricsFooter?: string | null;
  headSha?: string | null;
}): Promise<PostCommentResult> {
  const { prUrl, reviewText, model, metricsFooter, headSha } = opts;
  const platform = detectPlatform(prUrl);
  logger.info(`Posting comment to ${platform} PR/MR: ${prUrl}`);

  let parsed: ParsedPrUrl;
  try {
    parsed = parsePrUrl(prUrl);
  } catch (err) {
    return { success: false, error: String(err) };
  }

  let body = reviewText;
  if (headSha) {
    body = `<!-- hodor:sha:${headSha} -->\n${body}`;
  }
  if (model) {
    body = `${body}\n\n---\n\nReview generated by Hodor (model: \`${model}\`)`;
  }
  if (metricsFooter) {
    body = `${body}\n\n${metricsFooter}`;
  }

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
      logger.info(`Successfully posted review to GitHub PR #${parsed.prNumber}`);
      return { success: true, platform: "github", prNumber: parsed.prNumber };
    } else if (platform === "gitea") {
      await postGiteaPrComment(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        body,
        parsed.host,
      );
      logger.info(`Successfully posted review to Gitea PR #${parsed.prNumber}`);
      return { success: true, platform: "gitea", prNumber: parsed.prNumber };
    } else {
      await postGitlabMrComment(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        body,
        parsed.host,
      );
      logger.info(
        `Successfully posted review to GitLab MR !${parsed.prNumber}`,
      );
      return {
        success: true,
        platform: "gitlab",
        mrNumber: parsed.prNumber,
      };
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error(`Failed to post comment: ${msg}`);
    return { success: false, error: msg };
  }
}

export async function postReviewStructured(opts: {
  prUrl: string;
  review: ReviewOutput;
  model?: string | null;
  metricsFooter?: string | null;
  reviewStyle?: "summary" | "inline" | "hybrid";
  commitStatus?: boolean;
  codeQualityPath?: string | null;
  headSha?: string | null;
  workspacePath?: string | null;
}): Promise<PostCommentResult> {
  const {
    prUrl,
    review,
    model,
    metricsFooter,
    reviewStyle,
    commitStatus,
    codeQualityPath,
    headSha,
    workspacePath,
  } = opts;

  const platform = detectPlatform(prUrl);
  if (platform === "github") {
    return postReviewComment({
      prUrl,
      reviewText: renderMarkdown(review),
      model,
      metricsFooter,
      headSha,
    });
  }

  if (reviewStyle === "summary") {
    return postReviewComment({
      prUrl,
      reviewText: renderMarkdown(review),
      model,
      metricsFooter,
      headSha,
    });
  }

  const parsed = parsePrUrl(prUrl);

  try {
    const discussions = await listHodorDiscussions(
      parsed.owner,
      parsed.repo,
      parsed.prNumber,
      parsed.host,
    );
    const unresolvedIds = [...new Set(
      discussions.filter((d) => !d.resolved).map((d) => d.discussionId),
    )];
    if (unresolvedIds.length > 0) {
      const resolved = await resolveGitlabDiscussions(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        unresolvedIds,
        parsed.host,
      );
      if (resolved > 0) logger.info(`Resolved ${resolved} old Hodor discussion(s)`);
    }
  } catch (err) {
    logger.warn(`Failed to resolve old discussions: ${err instanceof Error ? err.message : err}`);
  }

  try {
    const deleted = await cleanupHodorComments(
      parsed.owner,
      parsed.repo,
      parsed.prNumber,
      parsed.host,
    );
    if (deleted > 0) logger.info(`Cleaned up ${deleted} old Hodor comment(s)`);
  } catch (err) {
    logger.warn(`Failed to cleanup old comments: ${err instanceof Error ? err.message : err}`);
  }

  let diffRefs: DiffRefs | null = null;
  try {
    diffRefs = await getGitlabMrDiffRefs(
      parsed.owner,
      parsed.repo,
      parsed.prNumber,
      parsed.host,
    );
  } catch (err) {
    logger.warn(`Failed to get diff_refs, falling back to summary mode: ${err instanceof Error ? err.message : err}`);
  }

  if (!diffRefs) {
    return postReviewComment({
      prUrl,
      reviewText: renderMarkdown(review),
      model,
      metricsFooter,
      headSha,
    });
  }

  let inlineCount = 0;
  let failedCount = 0;
  let summaryPosted = false;
  let draftsPublished = false;
  let statusPosted = false;
  const postingErrors: string[] = [];

  for (const finding of review.findings) {
    const relPath = formatLocationRelative(finding.code_location, workspacePath);
    const priorityTag = `[P${finding.priority}]`;
    const title = /^\[P[0-3]\]/.test(finding.title)
      ? finding.title
      : `${priorityTag} ${finding.title}`;
    let body = `${HODOR_REVIEW_MARKER}\n**${title}**\n\n${finding.body}`;

    if (finding.suggestion) {
      // GitLab suggestion blocks anchor to the comment's line and extend `+N` lines below.
      // Single-line: `suggestion:-0+0` (replace one line). Range: `suggestion:-0+N`
      // where N is the number of additional lines beyond the anchor.
      const { start, end } = finding.code_location.line_range;
      const span = Math.max(0, end - start);
      body += `\n\n\`\`\`suggestion:-0+${span}\n${finding.suggestion}\n\`\`\``;
    }

    try {
      await createGitlabDraftNote(
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
      inlineCount++;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn(`Failed to create inline note for "${finding.title}": ${msg}`);
      postingErrors.push(`inline note: ${msg}`);
      failedCount++;
    }
  }

  logger.info(`Created ${inlineCount} inline draft note(s)${failedCount > 0 ? ` (${failedCount} failed)` : ""}`);

  if (reviewStyle === "hybrid" || reviewStyle === undefined) {
    let summaryBody = renderSummaryMarkdown(review);
    if (headSha) summaryBody = `<!-- hodor:sha:${headSha} -->\n${summaryBody}`;
    if (model) summaryBody += `\n---\n\nReview generated by Hodor (model: \`${model}\`)`;
    if (metricsFooter) summaryBody += `\n\n${metricsFooter}`;
    try {
      await postGitlabMrComment(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        summaryBody,
        parsed.host,
      );
      summaryPosted = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn(`Failed to post summary comment: ${msg}`);
      postingErrors.push(`summary comment: ${msg}`);
    }
  }

  if (inlineCount > 0) {
    try {
      await bulkPublishGitlabDraftNotes(
        parsed.owner,
        parsed.repo,
        parsed.prNumber,
        parsed.host,
      );
      logger.info("Published all draft notes");
      draftsPublished = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn(`Failed to bulk publish draft notes: ${msg}`);
      postingErrors.push(`draft publish: ${msg}`);
    }
  }

  if (commitStatus && diffRefs) {
    try {
      await postGitlabReviewCommitStatus(parsed, review, diffRefs);
      logger.info("Posted commit status");
      statusPosted = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      logger.warn(`Failed to post commit status: ${msg}`);
      postingErrors.push(`commit status: ${msg}`);
    }
  }

  if (codeQualityPath) {
    try {
      const { formatCodeQualityReport } = await import("./codequality.js");
      const report = formatCodeQualityReport(review, workspacePath ?? undefined);
      const { writeFileSync } = await import("node:fs");
      writeFileSync(codeQualityPath, report, "utf-8");
      logger.info(`Wrote code quality report to ${codeQualityPath}`);
    } catch (err) {
      logger.warn(`Failed to write code quality report: ${err instanceof Error ? err.message : err}`);
    }
  }

  const visibleResult = summaryPosted || (inlineCount > 0 && draftsPublished) || statusPosted;
  const expectedInlineComments = reviewStyle === "inline" && review.findings.length > 0;
  if ((postingErrors.length > 0 && !visibleResult) || (expectedInlineComments && inlineCount === 0)) {
    return {
      success: false,
      platform: "gitlab",
      mrNumber: parsed.prNumber,
      error: postingErrors[0] ?? "No GitLab inline comments were created",
    };
  }

  return {
    success: true,
    platform: "gitlab",
    mrNumber: parsed.prNumber,
  };
}

export async function reviewPr(opts: {
  prUrl?: string;
  model?: string;
  reasoningEffort?: string;
  customPrompt?: string | null;
  promptFile?: string | null;
  cleanup?: boolean;
  workspaceDir?: string | null;
  includeMetricsFooter?: boolean;
  onEvent?: (event: AgentProgressEvent) => void;
  bedrockTags?: Record<string, string> | null;
  localMode?: boolean;
  diffAgainst?: string;
}): Promise<{ review: ReviewOutput; metricsFooter: string | null; headSha: string | null; metrics: ReviewMetrics; workspacePath: string }> {
  const {
    prUrl,
    model = "anthropic/claude-sonnet-4-5-20250929",
    reasoningEffort,
    customPrompt,
    promptFile,
    cleanup = true,
    workspaceDir,
    includeMetricsFooter = false,
    onEvent,
    bedrockTags,
    localMode = false,
    diffAgainst,
  } = opts;

  logger.info(`Starting PR review for: ${localMode ? "local diff" : prUrl}`);

  let owner = "", repo = "", host = "";
  let prNumber = 0;
  let platform: Platform = "github";

  if (!localMode && prUrl) {
    const urlParsed = parsePrUrl(prUrl);
    owner = urlParsed.owner;
    repo = urlParsed.repo;
    prNumber = urlParsed.prNumber;
    host = urlParsed.host;
    platform = detectPlatform(prUrl);
    logger.info(`Platform: ${platform}, Repo: ${owner}/${repo}, PR: ${prNumber}, Host: ${host}`);
  }

  // --- Preflight: validate model + credentials before any expensive I/O ---
  const parsed = parseModelString(model);
  const thinkingLevel = mapReasoningEffort(reasoningEffort);

  // Snapshot env vars we may mutate, restore in finally block.
  const envSnapshot: Record<string, string | undefined> = {
    AWS_REGION: process.env.AWS_REGION,
  };

  // Import pi SDK
  const {
    AuthStorage,
    createAgentSession,
    DefaultResourceLoader,
    ModelRegistry,
    SessionManager,
    SettingsManager,
    getAgentDir,
  } = await import("@earendil-works/pi-coding-agent");

  // In-memory auth storage avoids loading ~/.pi/auth.json — env vars only.
  const authStorage = AuthStorage.inMemory();
  if (process.env.LLM_API_KEY) {
    authStorage.setRuntimeApiKey(parsed.provider, process.env.LLM_API_KEY);
  }
  const modelRegistry = ModelRegistry.inMemory(authStorage);

  // Resolve model — use registry for known models, construct manually for custom ARNs
  let piModel: Model<Api>;
  if (parsed.modelId.startsWith("arn:")) {
    // Custom bedrock ARN (inference profile, cross-region, etc.)
    // Extract region from ARN: arn:aws:bedrock:<region>:<account>:...
    const arnParts = parsed.modelId.split(":");
    const region = arnParts.length >= 4 ? arnParts[3] : "us-east-1";
    // Set AWS_REGION so the BedrockRuntimeClient uses the correct endpoint
    if (!process.env.AWS_REGION && !process.env.AWS_DEFAULT_REGION) {
      process.env.AWS_REGION = region;
    }
    piModel = {
      id: parsed.modelId,
      name: parsed.modelId,
      api: "bedrock-converse-stream",
      provider: "amazon-bedrock",
      baseUrl: `https://bedrock-runtime.${region}.amazonaws.com`,
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 200000,
      maxTokens: 16384,
    } as Model<Api>;
    logger.info(`Custom bedrock ARN model — region: ${region}`);
  } else {
    const registryModel = modelRegistry.find(parsed.provider, parsed.modelId);
    if (registryModel) {
      piModel = registryModel;
    } else if (parsed.provider === "openrouter") {
      piModel = {
        id: parsed.modelId,
        name: parsed.modelId,
        api: "openai-completions",
        provider: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
        reasoning: true,
        input: ["text", "image"] as ("text" | "image")[],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 256000,
        maxTokens: 65536,
      } as Model<Api>;
      logger.warn(`Using best-effort unregistered OpenRouter model — ${parsed.modelId}`);
    } else {
      throw new Error(
        `Unsupported model "${model}". Provider "${parsed.provider}" is recognized by pi-ai, but model "${parsed.modelId}" was not found in the installed registry.`,
      );
    }
  }

  // Note: For bedrock, don't preflight-check AWS credentials because the SDK
  // resolves them from many sources (env vars, IMDS, ECS task role, IRSA,
  // ~/.aws/credentials, etc.) and we can't reliably detect all of them.
  if (parsed.provider !== "amazon-bedrock") {
    const resolvedKey = await modelRegistry.getApiKeyForProvider(parsed.provider);
    if (!resolvedKey) {
      throw new Error(
        `No API key found for provider "${parsed.provider}". Set the provider-specific environment variable, configure pi auth, or set LLM_API_KEY.`,
      );
    }
  }
  logger.info("Preflight OK — model and credentials validated");

  // --- End preflight ---

  // Setup workspace
  let workspacePath: string;
  let targetBranch: string;
  let diffBaseSha: string | null = null;
  let isTemporary = false;

  if (localMode) {
    // Resolve to git repo root so paths from git diff match tool expectations
    const cwd = workspaceDir ?? process.cwd();
    try {
      const { stdout: toplevel } = await exec("git", ["rev-parse", "--show-toplevel"], { cwd });
      workspacePath = toplevel.trim();
    } catch {
      workspacePath = cwd; // fallback if not in a git repo
    }
    targetBranch = diffAgainst ?? "origin/main";
    logger.info(`Local mode: workspace=${workspacePath}, diffAgainst=${targetBranch}`);
  } else {
    const wsResult = await setupWorkspace({
      platform,
      owner,
      repo,
      prNumber: String(prNumber),
      host,
      workingDir: workspaceDir ?? undefined,
      reuse: workspaceDir != null,
    });
    workspacePath = wsResult.workspace;
    targetBranch = wsResult.targetBranch;
    diffBaseSha = wsResult.diffBaseSha;
    isTemporary = wsResult.isTemporary;
  }

  try {
    let mrMetadata: MrMetadata | null = null;
    if (!localMode && platform === "gitlab") {
      try {
        mrMetadata = await fetchGitlabMrInfo(owner, repo, prNumber, host, {
          includeComments: true,
        });
      } catch (err) {
        logger.warn(`Failed to fetch GitLab metadata: ${err}`);
      }
    } else if (!localMode && platform === "github") {
      try {
        const githubRaw = await fetchGithubPrInfo(owner, repo, prNumber);
        mrMetadata = normalizeGithubMetadata(githubRaw);
      } catch (err) {
        logger.warn(`Failed to fetch GitHub metadata: ${err}`);
      }
    } else if (!localMode && platform === "gitea") {
      try {
        mrMetadata = await fetchGiteaPrInfo(owner, repo, prNumber, host, {
          includeComments: true,
        });
      } catch (err) {
        logger.warn(`Failed to fetch Gitea metadata: ${err}`);
      }
    }

    // Detect previous hodor review SHA for incremental mode
    let previousReviewSha: string | null = null;
    if (mrMetadata?.Notes) {
      for (const note of mrMetadata.Notes) {
        const match = note.body?.match(/<!-- hodor:sha:([a-f0-9]{40}) -->/);
        if (match) {
          previousReviewSha = match[1]; // take the last (most recent) match
        }
      }
    }
    if (previousReviewSha) {
      try {
        // Verify it's a commit (not a blob/tree) and is an ancestor of HEAD
        const { stdout: objType } = await exec("git", ["cat-file", "-t", previousReviewSha], { cwd: workspacePath });
        if (objType.trim() !== "commit") throw new Error("not a commit");
        await exec("git", ["merge-base", "--is-ancestor", previousReviewSha, "HEAD"], { cwd: workspacePath });
        logger.info(`Incremental mode: previous review at ${previousReviewSha.slice(0, 8)}`);
      } catch {
        logger.info(`Previous review SHA ${previousReviewSha.slice(0, 8)} not valid ancestor of HEAD, doing full review`);
        previousReviewSha = null;
      }
    }

    // Get HEAD SHA for embedding in posted comments (skip in local mode — no posting)
    let headSha: string | null = null;
    if (!localMode) {
      const { stdout: headShaRaw } = await exec("git", ["rev-parse", "HEAD"], { cwd: workspacePath });
      headSha = headShaRaw.trim();
    }

    // Pre-fetch diff for embedding in prompt (avoids per-file tool calls)
    const MAX_EMBED_BYTES = 200 * 1024; // 200KB
    let embeddedDiff: string | null = null;
    try {
      const diffArgs = previousReviewSha
        ? ["--no-pager", "diff", `${previousReviewSha}...HEAD`]
        : diffBaseSha
          ? ["--no-pager", "diff", diffBaseSha, "HEAD"]
          : localMode
            ? ["--no-pager", "diff", targetBranch]  // includes uncommitted changes
            : ["--no-pager", "diff", `origin/${targetBranch}...HEAD`];
      const { stdout: rawDiff } = await exec("git", diffArgs, { cwd: workspacePath });
      if (Buffer.byteLength(rawDiff, "utf-8") <= MAX_EMBED_BYTES) {
        embeddedDiff = rawDiff;
        logger.info(`Embedding diff in prompt (${Buffer.byteLength(rawDiff, "utf-8")} bytes)`);
      } else {
        logger.info(`Diff too large to embed (${Buffer.byteLength(rawDiff, "utf-8")} bytes), using command mode`);
      }
    } catch (err) {
      logger.warn(`Failed to pre-fetch diff, falling back to command mode: ${err}`);
    }

    // Build prompt (always uses JSON template; rendered to markdown post-hoc)
    const prompt = buildPrReviewPrompt({
      prUrl: prUrl ?? `local diff (against ${targetBranch})`,
      platform,
      targetBranch,
      diffBaseSha,
      mrMetadata,
      customInstructions: customPrompt,
      customPromptFile: promptFile,
      embeddedDiff,
      previousReviewSha,
      localMode,
    });

    const startTime = Date.now();
    const settingsManager = SettingsManager.inMemory({
      compaction: { enabled: true },
    });
    const skillPaths = [
      join(workspacePath, ".pi", "skills"),
      join(workspacePath, ".hodor", "skills"),
    ].filter((p) => existsSync(p));
    const resourceLoader = new DefaultResourceLoader({
      cwd: workspacePath,
      agentDir: getAgentDir(),
      settingsManager,
      systemPrompt: REVIEW_SYSTEM_PROMPT,
      appendSystemPrompt: [],
      noExtensions: true,
      noSkills: true,
      noPromptTemplates: true,
      noThemes: true,
      additionalSkillPaths: skillPaths,
      agentsFilesOverride: () => ({ agentsFiles: [] }),
    });
    await resourceLoader.reload();
    const { skills, diagnostics: skillDiagnostics } = resourceLoader.getSkills();
    if (skills.length > 0) {
      logger.info(`Discovered ${skills.length} repository skill(s)`);
      for (const skill of skills) {
        logger.info(`Found skill: ${skill.name} (${skill.filePath})`);
      }
    }
    for (const diagnostic of skillDiagnostics) {
      const path = diagnostic.path ? ` (${diagnostic.path})` : "";
      logger.warn(`Skill diagnostic: ${diagnostic.message}${path}`);
    }

    let submittedReview: ReviewOutput | null = null;
    let submitReviewCalls = 0;
    const submitReviewTool: ToolDefinition = {
      name: "submit_review",
      label: "Submit Review",
      description: "Submit the final structured review after the analysis is complete.",
      promptSnippet: "Submit the final structured review (call exactly once when done)",
      parameters: SUBMIT_REVIEW_SCHEMA,
      execute: async (_toolCallId, params, _signal, _onUpdate, _ctx) => {
        submitReviewCalls++;
        if (submittedReview) {
          logger.warn("Agent called submit_review more than once; ignoring duplicate submission");
          return {
            content: [{
              type: "text",
              text: "Review already submitted. Do not call submit_review again.",
            }],
            details: { ignoredDuplicate: true },
          };
        }

        submittedReview = validateReviewOutput(params as ReviewOutput);
        logger.info(
          `Received structured review via submit_review (${submittedReview.findings.length} finding(s))`,
        );
        return {
          content: [{
            type: "text",
            text: "Review received. Do not output the review as normal text.",
          }],
          details: {},
        };
      },
    };

    const { session } = await createAgentSession({
      cwd: workspacePath,
      model: piModel,
      thinkingLevel,
      tools: ["read", "bash", "grep", "find", "ls"],
      customTools: [submitReviewTool],
      authStorage,
      modelRegistry,
      sessionManager: SessionManager.inMemory(),
      settingsManager,
      resourceLoader,
    });

    // Inject Bedrock cost allocation tags into stream requests
    if (bedrockTags && parsed.provider === "bedrock") {
      type AgentWithStream = { agent: { streamFn: (...args: unknown[]) => unknown } };
      const agent = (session as unknown as AgentWithStream).agent;
      const originalStreamFn = agent.streamFn;
      agent.streamFn = (...args: unknown[]) => {
        const options = (args[2] ?? {}) as Record<string, unknown>;
        return originalStreamFn(args[0], args[1], { ...options, requestMetadata: bedrockTags });
      };
      logger.info(`Bedrock cost allocation tags: ${JSON.stringify(bedrockTags)}`);
    }

    // Subscribe to agent events for progress + metrics tracking
    let turnCount = 0;
    let toolCallCount = 0;

    /** Extract human-readable summary from tool args */
    function formatToolArgs(_toolName: string, args: unknown): string {
      if (typeof args === "string") return args.slice(0, 200);
      const obj = args as Record<string, unknown> | undefined;
      if (!obj) return "";
      // bash tool: show the command, strip workspace prefix
      if (obj.command) {
        return String(obj.command)
          .replace(new RegExp(`cd ${workspacePath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")} && `), "")
          .slice(0, 200);
      }
      // grep/find: show pattern + path
      if (obj.pattern) {
        const path = obj.path ? ` in ${obj.path}` : "";
        return `${obj.pattern}${path}`;
      }
      // read/ls: show the path
      if (obj.path || obj.file_path) return String(obj.path ?? obj.file_path);
      return JSON.stringify(obj).slice(0, 200);
    }

    /** Extract text content from tool result */
    function formatToolResult(result: unknown): string {
      if (typeof result === "string") return result;
      const obj = result as Record<string, unknown> | undefined;
      if (!obj) return "";
      // pi-sdk wraps results as {content: [{type: "text", text: "..."}]}
      const content = obj.content as Array<{ type?: string; text?: string }> | undefined;
      if (Array.isArray(content)) {
        return content
          .filter((c) => c.type === "text" && c.text)
          .map((c) => c.text)
          .join("\n");
      }
      return JSON.stringify(result)?.slice(0, 500) ?? "";
    }

    session.subscribe((event) => {
      switch (event.type) {
        case "agent_start":
          onEvent?.({ type: "agent_start" });
          break;
        case "agent_end":
          onEvent?.({ type: "agent_end" });
          break;
        case "turn_start":
          turnCount++;
          onEvent?.({ type: "turn_start", turnIndex: turnCount });
          break;
        case "turn_end":
          onEvent?.({ type: "turn_end", turnIndex: turnCount });
          break;
        case "tool_execution_start":
          toolCallCount++;
          onEvent?.({
            type: "tool_start",
            toolName: event.toolName,
            toolArgs: formatToolArgs(event.toolName, event.args),
          });
          break;
        case "tool_execution_end":
          onEvent?.({
            type: "tool_end",
            toolName: event.toolName,
            isError: event.isError,
            result: formatToolResult(event.result),
          });
          break;
        case "message_start":
          onEvent?.({ type: "thinking" });
          break;
        case "message_update": {
          const msgEvent = (event as Record<string, unknown>).assistantMessageEvent as
            { type: string; delta?: string } | undefined;
          if (!msgEvent?.delta) break;
          if (msgEvent.type === "text_delta") {
            onEvent?.({ type: "text_delta", delta: msgEvent.delta });
          } else if (msgEvent.type === "thinking_delta") {
            onEvent?.({ type: "thinking_delta", delta: msgEvent.delta });
          }
          break;
        }
      }
    });

    logger.info("Sending prompt to agent...");
    await session.prompt(prompt);

    // Check for agent errors (pi-ai swallows LLM errors into state.error)
    const agentError = (session as unknown as { state: { error?: string } }).state?.error;
    if (agentError) {
      throw new Error(`LLM request failed: ${agentError}`);
    }

    if (!submittedReview) {
      const rawText = session.getLastAssistantText() ?? "";
      if (rawText) {
        logger.debug(`Last assistant text without submit_review (first 500 chars): ${rawText.slice(0, 500)}`);
      } else {
        const messages = (session as unknown as { state: { messages: unknown[] } }).state?.messages;
        const lastMsg = messages?.[messages.length - 1];
        logger.debug(`Last message: ${JSON.stringify(lastMsg)?.slice(0, 500)}`);
      }
      if (submitReviewCalls > 0) {
        throw new Error("Agent called submit_review but did not provide a valid review payload");
      }
      throw new Error("Agent did not call submit_review");
    }

    const review = submittedReview as ReviewOutput;
    if (submitReviewCalls > 1) {
      logger.warn(`Agent called submit_review ${submitReviewCalls} times; using the first valid submission`);
    }
    logger.info(
      `Captured ${review.findings.length} finding(s), verdict: ${review.overall_correctness}`,
    );

    const durationSeconds = (Date.now() - startTime) / 1000;
    logger.info(`Review complete (${review.findings.length} finding(s))`);

    // Aggregate usage from all assistant messages
    interface MsgUsage {
      input: number;
      output: number;
      cacheRead: number;
      cacheWrite: number;
      totalTokens: number;
      cost: { total: number };
    }
    interface AssistantMsg {
      role: string;
      usage?: MsgUsage;
    }

    const allMessages = (
      session as unknown as { state: { messages: AssistantMsg[] } }
    ).state?.messages ?? [];

    let inputTokens = 0;
    let outputTokens = 0;
    let cacheReadTokens = 0;
    let cacheWriteTokens = 0;
    let totalTokens = 0;
    let cost = 0;

    for (const msg of allMessages) {
      if (msg.role === "assistant" && msg.usage) {
        inputTokens += msg.usage.input ?? 0;
        outputTokens += msg.usage.output ?? 0;
        cacheReadTokens += msg.usage.cacheRead ?? 0;
        cacheWriteTokens += msg.usage.cacheWrite ?? 0;
        totalTokens += msg.usage.totalTokens ?? 0;
        cost += msg.usage.cost?.total ?? 0;
      }
    }

    const metrics: ReviewMetrics = {
      inputTokens,
      outputTokens,
      cacheReadTokens,
      cacheWriteTokens,
      totalTokens,
      cost,
      turns: turnCount,
      toolCalls: toolCallCount,
      durationSeconds: Math.round(durationSeconds),
    };
    printMetrics(metrics);

    let metricsFooter: string | null = null;
    if (includeMetricsFooter) {
      metricsFooter = formatMetricsMarkdown(metrics);
    }

    return { review, metricsFooter, headSha, metrics, workspacePath };
  } finally {
    // Restore mutated env vars
    for (const [key, val] of Object.entries(envSnapshot)) {
      if (val === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = val;
      }
    }

    if (cleanup && isTemporary) {
      logger.info("Cleaning up workspace...");
      await cleanupWorkspace(workspacePath);
    }
  }
}
