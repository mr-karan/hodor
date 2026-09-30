import { existsSync } from "node:fs";
import { join } from "node:path";
import {
  createAgentSession,
  createCodemodeExtension,
  DefaultResourceLoader,
  getAgentDir,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import type { AgentSession, ToolDefinition } from "@earendil-works/pi-coding-agent";
import type { Api, Model, ThinkingLevel } from "@earendil-works/pi-ai";
import { logger } from "./utils/logger.js";
import { exec } from "./utils/exec.js";
import { fetchGithubPrMetadata } from "./github.js";
import {
  fetchGitlabMrInfo,
  listHodorDiscussions,
} from "./gitlab.js";
import {
  fetchGiteaPrInfo,
} from "./gitea.js";
import { setupWorkspace, cleanupWorkspace } from "./workspace.js";
import { buildMrSections, buildPrReviewPrompt } from "./prompt.js";
import {
  addOpenAiBedrockReasoning,
  buildBedrockArnModel,
  extractBedrockArnRegion,
  getDefaultReasoningEffortForModel,
  isOpenAiBedrockModel,
  parseModelString,
  qualifiesForSingleTurnReview,
  selectReasoningEffort,
  stripBedrockRegionalPrefix,
} from "./model.js";
import {
  assertPublicOpenRouterFallbackAllowed,
  createModelRuntime,
  loadModelsJsonConfig,
} from "./models-json.js";
import { createCodemodeLimitsExtension, type CodemodeLimits } from "./codemode-limits.js";
import { formatMetricsMarkdown } from "./metrics.js";
import { SUBMIT_REVIEW_SCHEMA, validateReviewOutput } from "./review.js";
import { resolveReviewLocations } from "./resolve-location.js";
import { createReviewToolset, REVIEW_TOOL_NAMES, type ReviewToolset } from "./review-tools.js";
import { buildReviewSystemPrompt } from "./system-prompt.js";
import {
  loadDefaultReviewInstructions,
  validateReviewInstructions,
} from "./review-instructions.js";
import { detectPlatform, parsePrUrl } from "./platform.js";
import {
  filterEmbeddedDiff,
  findLatestReviewBase,
  getReviewDiffArgs,
  getChangedFiles,
  getDiffStats,
  resolveReviewBaseSha,
  type DiffStats,
  type ReviewDiffMode,
} from "./review-diff.js";
import { partitionNotesByProvenance, resolvePublisherIdentity } from "./provenance.js";
import {
  buildFixCandidates,
  getFindingFingerprint,
  MAX_PROMPT_FINDING_THREADS,
  selectFindingThreads,
  selectVerifiedFixes,
  type FindingThread,
  type FixCandidate,
} from "./review-state.js";
import {
  buildReviewCacheMarker,
  findCachedReview,
  getReviewCacheKey,
} from "./review-cache.js";
import {
  buildSubmitReviewRecoveryPrompt,
  parseReviewFromAssistantText,
  SUBMIT_REVIEW_RECOVERY_ATTEMPTS,
  summarizeLastAssistantMessage,
} from "./review-recovery.js";
export { detectPlatform, parsePrUrl } from "./platform.js";
export { filterEmbeddedDiff, getHodorReviewShaCandidates } from "./review-diff.js";
export { buildSubmitReviewRecoveryPrompt, parseReviewFromAssistantText } from "./review-recovery.js";
export {
  postGitlabReviewCommitStatus,
  postReviewComment,
  postReviewStructured,
} from "./publisher.js";
import type {
  Platform,
  ReviewContextManifest,
  ReviewMetrics,
  ReviewRange,
  MrMetadata,
  ReviewOutput,
} from "./types.js";

export interface AgentProgressEvent {
  type: "tool_start" | "tool_end" | "thinking" | "turn_start" | "turn_end" | "agent_start" | "agent_end" | "text_delta" | "thinking_delta" | "tool_result" | "retry" | "compaction";
  toolName?: string;
  toolArgs?: string;
  /** Set on tool_start and tool_end. */
  toolCallId?: string;
  /** Set on tool_start and tool_end of calls a codemode script made. */
  parentToolCallId?: string;
  isError?: boolean;
  turnIndex?: number;
  delta?: string;
  result?: string;
  phase?: "start" | "end";
  attempt?: number;
  maxAttempts?: number;
  delayMs?: number;
  reason?: string;
  success?: boolean;
}


type StreamFunction = AgentSession["agent"]["streamFunction"];

/**
 * Wrap a Pi stream function to add Hodor-specific Bedrock request fields:
 * cost allocation tags and OpenAI-on-Bedrock reasoning effort.
 */
export function wrapBedrockStream(
  streamFunction: StreamFunction,
  opts: {
    bedrockTags?: Record<string, string> | null;
    openAiReasoning?: ThinkingLevel;
  },
): StreamFunction {
  const { bedrockTags, openAiReasoning } = opts;
  return (model, context, options) => {
    const originalOnPayload = options?.onPayload;
    const onPayload = openAiReasoning
      ? async (payload: unknown, payloadModel: Model<Api>) => {
          const transformed = originalOnPayload
            ? await originalOnPayload(payload, payloadModel)
            : undefined;
          return addOpenAiBedrockReasoning(
            transformed === undefined ? payload : transformed,
            openAiReasoning,
          );
        }
      : originalOnPayload;
    return streamFunction(model, context, {
      ...options,
      ...(bedrockTags ? { requestMetadata: bedrockTags } : {}),
      ...(onPayload ? { onPayload } : {}),
    });
  };
}

/**
 * Tool names and definitions for the review session.
 *
 * Pi filters customTools through the same `tools` allowlist as its built-ins
 * (_refreshToolRegistry in agent-session.js), and a custom definition replaces
 * a built-in of the same name. So every tool, including submit_review, is
 * named here, and read/grep/find/ls resolve to Hodor's confined versions.
 * The model gets no shell.
 */
export function getReviewSessionTools(opts: {
  singleTurn: boolean;
  reviewTools: ToolDefinition[];
  submitReviewTool: ToolDefinition;
  codemode?: boolean;
}): { tools: string[]; customTools: ToolDefinition[] } {
  if (opts.singleTurn) {
    return { tools: ["submit_review"], customTools: [opts.submitReviewTool] };
  }
  return {
    tools: [...REVIEW_TOOL_NAMES, "submit_review", ...(opts.codemode ? ["codemode"] : [])],
    customTools: [...opts.reviewTools, opts.submitReviewTool],
  };
}

/**
 * Build the session's resource loader. Disk and project extensions, prompt
 * templates, themes, and context files stay off. With `codemode`, only Pi's
 * codemode built-in loads, with its `models` API disabled so scripts cannot
 * call classifiers with the session's credentials.
 */
export async function createReviewResourceLoader(opts: {
  cwd: string;
  agentDir: string;
  settingsManager: SettingsManager;
  systemPrompt?: string;
  skillPaths?: string[];
  codemode?: boolean;
  /** Overrides the enforced codemode script limits (tests use a short timeout). */
  codemodeLimits?: CodemodeLimits;
}): Promise<DefaultResourceLoader> {
  const resourceLoader = new DefaultResourceLoader({
    cwd: opts.cwd,
    agentDir: opts.agentDir,
    settingsManager: opts.settingsManager,
    ...(opts.systemPrompt !== undefined
      ? { systemPromptOverride: () => opts.systemPrompt, appendSystemPromptOverride: () => [] }
      : {}),
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    additionalSkillPaths: opts.skillPaths ?? [],
    agentsFilesOverride: () => ({ agentsFiles: [] }),
    ...(opts.codemode
      ? {
        extensionFactories: [
          {
            name: "codemode",
            builtin: true,
            factory: createCodemodeExtension({ models: false, mode: "on" }),
          },
          // Codemode sets no script timeout by default; enforce one.
          { name: "hodor-codemode-limits", factory: createCodemodeLimitsExtension(opts.codemodeLimits) },
        ],
        additionalExtensionPaths: ["builtin:codemode"],
      }
      : {}),
  });
  await resourceLoader.reload();
  const extensionErrors = resourceLoader.getExtensions().errors;
  if (extensionErrors.length > 0) {
    throw new Error(`Failed to load review extensions: ${extensionErrors.map((e) => e.error).join("; ")}`);
  }
  return resourceLoader;
}

export async function reviewPr(opts: {
  prUrl?: string;
  model?: string;
  reasoningEffort?: string;
  reviewInstructions?: string | null;
  additionalInstructions?: string | null;
  cleanup?: boolean;
  workspaceDir?: string | null;
  includeMetricsFooter?: boolean;
  onEvent?: (event: AgentProgressEvent) => void;
  bedrockTags?: Record<string, string> | null;
  localMode?: boolean;
  diffAgainst?: string;
  full?: boolean;
  targetBranchOverride?: string;
  tinyDiffFastPath?: boolean;
  /** Let the model batch tool calls through Pi's codemode sandbox. */
  codemode?: boolean;
}): Promise<{
  review: ReviewOutput;
  metricsFooter: string | null;
  headSha: string | null;
  metrics: ReviewMetrics;
  workspacePath: string;
  cacheMarker: string | null;
  reusedReview: boolean;
  range: ReviewRange;
  /** Null for local and reused reviews, which build no MR prompt context. */
  context: ReviewContextManifest | null;
}> {
  const {
    prUrl,
    model = "anthropic/claude-opus-5-5",
    reasoningEffort,
    reviewInstructions,
    additionalInstructions,
    cleanup = true,
    workspaceDir,
    includeMetricsFooter = false,
    onEvent,
    bedrockTags,
    localMode = false,
    diffAgainst,
    full = false,
    targetBranchOverride,
    tinyDiffFastPath = false,
    codemode = false,
  } = opts;

  const effectiveReviewInstructions = reviewInstructions == null
    ? loadDefaultReviewInstructions()
    : validateReviewInstructions(reviewInstructions, "review instructions");
  const effectiveAdditionalInstructions = additionalInstructions == null
    ? null
    : validateReviewInstructions(additionalInstructions, "additional instructions");
  const composedSystemPrompt = buildReviewSystemPrompt({
    reviewInstructions: effectiveReviewInstructions,
    additionalInstructions: effectiveAdditionalInstructions,
  });

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
  const modelsJson = loadModelsJsonConfig();
  const parsed = parseModelString(model, modelsJson?.providers);

  // Snapshot env vars we may mutate, restore in finally block.
  const envSnapshot: Record<string, string | undefined> = {
    AWS_REGION: process.env.AWS_REGION,
    AWS_DEFAULT_REGION: process.env.AWS_DEFAULT_REGION,
  };

  // HODOR_MODELS_JSON optionally adds or overrides providers (for example a
  // self-hosted OpenAI-compatible gateway). Unset reads no user state.
  const modelRuntime = await createModelRuntime(modelsJson);
  if (process.env.LLM_API_KEY) {
    await modelRuntime.setRuntimeApiKey(parsed.provider, process.env.LLM_API_KEY);
  }

  // Resolve model — use registry for known models, construct manually for custom ARNs
  let piModel = modelRuntime.getModel(parsed.provider, parsed.modelId) as Model<Api> | undefined;
  if (parsed.modelId.startsWith("arn:")) {
    // Custom bedrock ARN (application/system inference profile, provisioned
    // throughput, etc.).
    const region = extractBedrockArnRegion(parsed.modelId);
    // Set AWS_REGION so the BedrockRuntimeClient uses the correct endpoint
    if (!process.env.AWS_REGION && !process.env.AWS_DEFAULT_REGION) {
      process.env.AWS_REGION = region;
    }

    const baseModel = parsed.baseModelId
      ? (modelRuntime.getModel(parsed.provider, parsed.baseModelId) as Model<Api> | undefined)
      : undefined;
    if (parsed.baseModelId && !baseModel) {
      throw new Error(
        `Base model "${parsed.baseModelId}" for Bedrock ARN "${parsed.modelId}" was not found in the installed pi-ai registry.`,
      );
    }

    piModel = buildBedrockArnModel({ arn: parsed.modelId, baseModel, region });
    if (baseModel) {
      logger.info(
        `Custom bedrock ARN model — region: ${region}, capabilities from ${baseModel.id}`,
      );
    } else {
      logger.warn(
        `Custom bedrock ARN model — region: ${region}, no base model given. ` +
          `Prompt caching, reasoning, and cost reporting are disabled. ` +
          `Append "@<base-model-id>" to the model string (e.g. "${model}@global.anthropic.claude-opus-5") to restore them.`,
      );
    }
  } else if (!piModel) {
    if (parsed.provider === "amazon-bedrock") {
      // Bedrock adds regional inference-profile prefixes before pi-ai's model
      // catalog catches up. Inherit capabilities from the unprefixed registry
      // model, while sending the original regional id to Bedrock.
      const inferredBaseModelId = stripBedrockRegionalPrefix(parsed.modelId);
      const baseModelId = parsed.baseModelId ?? inferredBaseModelId;
      const baseModel = baseModelId
        ? (modelRuntime.getModel(parsed.provider, baseModelId) as Model<Api> | undefined)
        : undefined;
      if (!baseModel) {
        const hint = parsed.baseModelId
          ? `Base model "${parsed.baseModelId}" was not found in the installed pi-ai registry.`
          : `Append "@<base-model-id>" if this is a custom inference profile.`;
        throw new Error(
          `Unsupported Bedrock model "${parsed.modelId}". ${hint}`,
        );
      }

      const region = process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? "us-east-1";
      piModel = buildBedrockArnModel({ arn: parsed.modelId, baseModel, region });
      logger.info(
        `Regional bedrock model, region: ${region}, capabilities from ${baseModel.id}`,
      );
    } else if (parsed.provider === "openrouter") {
      assertPublicOpenRouterFallbackAllowed(modelsJson, parsed.modelId);
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
  const modelDefaultThinkingLevel = getDefaultReasoningEffortForModel(piModel);

  // Note: For bedrock, don't preflight-check AWS credentials because the SDK
  // resolves them from many sources (env vars, IMDS, ECS task role, IRSA,
  // ~/.aws/credentials, etc.) and we can't reliably detect all of them.
  if (parsed.provider !== "amazon-bedrock") {
    const resolvedKey = await modelRuntime.getAuth(piModel);
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

  // --full with an explicit target overrides the detected base. Drop the CI
  // merge-base SHA so the diff uses origin/<target>...HEAD against the given ref.
  // CI clones don't fetch arbitrary branches, so fetch-and-verify the ref first
  // and fail loudly rather than silently reviewing against a missing base.
  if (!localMode && full && targetBranchOverride) {
    logger.info(`Full review: overriding target branch to '${targetBranchOverride}'`);
    try {
      await exec("git", ["fetch", "--quiet", "origin", targetBranchOverride], { cwd: workspacePath });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Failed to fetch --target-branch '${targetBranchOverride}' from origin for --full review: ${msg}`);
    }
    try {
      await exec("git", ["rev-parse", "--verify", "--quiet", `origin/${targetBranchOverride}`], { cwd: workspacePath });
    } catch {
      throw new Error(`--target-branch 'origin/${targetBranchOverride}' not found after fetch; cannot run --full review against it.`);
    }
    targetBranch = targetBranchOverride;
    diffBaseSha = null;
  }

  let activeSession: AgentSession | undefined;
  let reviewToolset: ReviewToolset | undefined;

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
        mrMetadata = await fetchGithubPrMetadata(owner, repo, prNumber, host);
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

    // Get HEAD SHA for embedding in posted comments (skip in local mode — no posting)
    let headSha: string | null = null;
    if (!localMode) {
      const { stdout: headShaRaw } = await exec("git", ["rev-parse", "HEAD"], { cwd: workspacePath });
      headSha = headShaRaw.trim();
    }

    // Authenticate note provenance once. Only Hodor-marked notes written by
    // the publishing identity carry machine state; everything else, including
    // forged markers, stays untrusted context for the prompt.
    const publisherIdentity = localMode ? null : await resolvePublisherIdentity(platform, host);
    const notes = partitionNotesByProvenance(mrMetadata?.Notes, publisherIdentity);
    if (mrMetadata) mrMetadata.Notes = [...notes.others, ...notes.hodor];

    // A successful Hodor summary contains a compressed, validated copy of the
    // structured result. Reuse it for an identical review identity so pipeline
    // retries can regenerate artifacts and retry delivery without another LLM
    // invocation. Explicit --full reviews always bypass this fast path.
    let reviewCacheKey: string | null = null;
    const reviewBaseSha = !localMode && headSha
      ? await resolveReviewBaseSha(workspacePath, targetBranch, diffBaseSha)
      : null;
    if (!full && headSha && reviewBaseSha) {
      reviewCacheKey = getReviewCacheKey({
        scope: {
          platform,
          host,
          projectPath: `${owner}/${repo}`,
          reviewNumber: prNumber,
          targetBranch,
          baseSha: reviewBaseSha,
        },
        headSha,
        model,
        requestedReasoningEffort: reasoningEffort,
        reviewInstructions: effectiveReviewInstructions,
        additionalInstructions: effectiveAdditionalInstructions,
      });
      const cachedReview = findCachedReview(notes.hodor, reviewCacheKey);
      if (cachedReview) {
        logger.info(`Reusing cached Hodor review for HEAD ${headSha.slice(0, 8)}`);
        const metrics: ReviewMetrics = {
          inputTokens: 0,
          outputTokens: 0,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          totalTokens: 0,
          cost: 0,
          turns: 0,
          toolCalls: 0,
          durationSeconds: 0,
          reviewMode: "reused",
          reasoningEffort: reasoningEffort ?? "auto",
          diffFiles: 0,
          diffAdditions: 0,
          diffDeletions: 0,
          diffBytes: 0,
          reused: true,
          fastPath: false,
        };
        logger.info(`Review telemetry: ${JSON.stringify({
          project: `${owner}/${repo}`,
          mr: prNumber,
          headSha: headSha.slice(0, 12),
          model,
          outcome: "reused",
          reviewMode: metrics.reviewMode,
          reasoningEffort: metrics.reasoningEffort,
          fastPath: false,
          reused: true,
          findings: cachedReview.findings.length,
        })}`);
        return {
          review: cachedReview,
          metricsFooter: includeMetricsFooter ? formatMetricsMarkdown(metrics) : null,
          headSha,
          metrics,
          workspacePath,
          cacheMarker: null,
          reusedReview: true,
          range: { headSha, targetBranch, baseSha: reviewBaseSha },
          context: null,
        };
      }
    }

    // Prefer the latest reviewed commit. Preserve three-dot semantics while it
    // is an ancestor; after a force-push/rebase, use a direct snapshot delta.
    const previousReviewBase = full || localMode
      ? null
      : await findLatestReviewBase(notes.hodor, workspacePath);
    const previousReviewSha = previousReviewBase?.sha ?? null;
    let reviewMode: ReviewDiffMode = localMode
      ? "local"
      : previousReviewBase?.mode ?? "full";
    if (full) {
      reviewMode = "full";
      logger.info("Full review mode: ignoring previous hodor reviews, diffing entire source-vs-target range");
    } else if (previousReviewBase) {
      logger.info(`${previousReviewBase.mode === "snapshot" ? "Snapshot delta" : "Incremental"} mode: previous review at ${previousReviewSha?.slice(0, 8)}`);
    }

    // Pre-fetch diff for embedding in prompt (avoids per-file tool calls)
    const MAX_EMBED_BYTES = 200 * 1024; // 200KB
    let embeddedDiff: string | null = null;
    let rawReviewDiff: string;
    let reviewDiff: string | null = null;
    let diffStats: DiffStats | null = null;
    let changedFiles: string[] = [];
    try {
      const diffArgs = getReviewDiffArgs({
        platform,
        targetBranch,
        diffBaseSha,
        previousReviewSha,
        reviewDiffMode: previousReviewBase?.mode,
        localMode,
      });
      const { stdout: rawDiff } = await exec("git", diffArgs, { cwd: workspacePath });
      rawReviewDiff = rawDiff;
      const { filtered: filteredDiff, skippedFiles } = filterEmbeddedDiff(rawDiff);
      if (skippedFiles.length > 0) {
        logger.info(`Filtered ${skippedFiles.length} file(s) from embedded diff: ${skippedFiles.join(", ")}`);
      }
      reviewDiff = filteredDiff;
      diffStats = getDiffStats(filteredDiff);
      changedFiles = getChangedFiles(filteredDiff);
      if (Buffer.byteLength(filteredDiff, "utf-8") <= MAX_EMBED_BYTES) {
        embeddedDiff = filteredDiff;
        logger.info(`Embedding diff in prompt (${Buffer.byteLength(filteredDiff, "utf-8")} bytes, raw: ${Buffer.byteLength(rawDiff, "utf-8")} bytes)`);
      } else {
        logger.info(`Diff too large to embed (${Buffer.byteLength(filteredDiff, "utf-8")} bytes filtered, ${Buffer.byteLength(rawDiff, "utf-8")} bytes raw), serving it through git_diff`);
      }
    } catch (err) {
      // The agent has no shell, so git_diff is its only view of the change.
      // Without this diff there is nothing to review.
      throw new Error(`Failed to compute the review diff: ${err instanceof Error ? err.message : err}`);
    }

    // Earlier Hodor finding threads give the model human replies as context
    // and name the open ones it may confirm fixed.
    let fixCandidates: FixCandidate[] = [];
    let findingThreads: FindingThread[] = [];
    let droppedFindingThreads = 0;
    // Human replies in Hodor finding threads are shown with their thread, so
    // the top-level human notes leave them out.
    const findingThreadNoteIds = new Set<number>();
    if (!localMode && platform === "gitlab" && publisherIdentity?.platform === "gitlab") {
      try {
        const discussions = await listHodorDiscussions(owner, repo, prNumber, host, publisherIdentity);
        for (const discussion of discussions) {
          for (const reply of discussion.humanReplies) findingThreadNoteIds.add(reply.noteId);
        }
        const candidates = buildFixCandidates(discussions);
        const allThreads = selectFindingThreads(discussions, candidates, changedFiles, Number.POSITIVE_INFINITY);
        findingThreads = allThreads.slice(0, MAX_PROMPT_FINDING_THREADS);
        droppedFindingThreads = allThreads.length - findingThreads.length;
        const presentedIds = new Set(findingThreads.flatMap((thread) => thread.fixId ? [thread.fixId] : []));
        fixCandidates = candidates.filter((candidate) => presentedIds.has(candidate.id));
        logger.info(
          `Showing ${findingThreads.length} Hodor finding thread(s); ${fixCandidates.length} may be confirmed fixed`,
        );
      } catch (err) {
        logger.warn(`Failed to list Hodor finding threads for the prompt: ${err instanceof Error ? err.message : err}`);
      }
    }

    const thinkingLevel = selectReasoningEffort({
      requested: reasoningEffort,
      modelDefault: modelDefaultThinkingLevel,
      mode: reviewMode,
      forcedFull: full,
      diff: reviewDiff,
      stats: diffStats,
    });
    if (thinkingLevel) {
      logger.info(`Reasoning effort for ${piModel.name}: ${thinkingLevel}${reasoningEffort ? " (explicit)" : " (adaptive)"}`);
    }

    const singleTurn = tinyDiffFastPath && qualifiesForSingleTurnReview({
      diff: reviewDiff,
      stats: diffStats,
      embedded: embeddedDiff != null,
    });
    if (singleTurn) {
      logger.info(
        `Single-turn fast path: tiny low-risk diff (${diffStats?.files} file(s), ` +
          `${(diffStats?.additions ?? 0) + (diffStats?.deletions ?? 0)} changed line(s)); exposing only submit_review`,
      );
    }

    // The review tools confine the model to the tracked tree. Location
    // resolution uses the same manifest, so build it on the fast path too.
    reviewToolset = await createReviewToolset({ workspacePath, reviewDiff: rawReviewDiff });
    logger.info(`Review tools confined to ${reviewToolset.tree.files.length} tracked file(s)`);

    const mrSections = buildMrSections(mrMetadata, { excludeNoteIds: findingThreadNoteIds });
    const context: ReviewContextManifest | null = localMode
      ? null
      : {
        hodorThreads: {
          open: findingThreads.filter((thread) => thread.status === "open").length,
          fixedWaiting: findingThreads.filter((thread) => thread.status === "fixed").length,
          resolved: findingThreads.filter((thread) => thread.status === "resolved").length,
          droppedByLimit: droppedFindingThreads,
        },
        humanComments: mrSections.humanNotes,
        priorHodorReviews: mrSections.priorHodorReviews,
      };
    if (context) logger.info(`Review context: ${JSON.stringify(context)}`);

    // Build the dynamic review task sent as the first user message.
    const prompt = buildPrReviewPrompt({
      prUrl: prUrl ?? `local diff (against ${targetBranch})`,
      platform,
      targetBranch,
      diffBaseSha,
      mrSections,
      embeddedDiff,
      previousReviewSha,
      reviewDiffMode: reviewMode,
      changedFiles,
      localMode,
      singleTurn,
      findingThreads,
    });

    const startTime = Date.now();
    // Pi defaults cacheWarming to "streaming", which can send extra paid
    // requests. A one-shot CI review has no later turn to warm for.
    const settingsManager = SettingsManager.inMemory({
      compaction: { enabled: true },
      cacheWarming: "off",
      // Bound SDK retry waits so a throttled provider cannot stall a CI job.
      retry: { maxRetries: 3, maxAgentDelayMs: 30_000 },
    });
    const skillPaths = [join(workspacePath, ".agents", "skills")]
      .filter((p) => existsSync(p));
    // Codemode scripts run in a QuickJS sandbox that can only call the
    // session's tools, which are Hodor's confined ones.
    const useCodemode = codemode && !singleTurn;
    if (useCodemode) logger.info("Codemode enabled");
    const resourceLoader = await createReviewResourceLoader({
      cwd: workspacePath,
      agentDir: getAgentDir(),
      settingsManager,
      systemPrompt: composedSystemPrompt,
      skillPaths,
      codemode: useCodemode,
    });
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
      constrainedSampling: { type: "json_schema", strict: "prefer" },
      // Codemode scripts must not submit: a nested call would lose `terminate`.
      exposure: "model-only",
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

        try {
          submittedReview = validateReviewOutput(params as ReviewOutput);
        } catch (err) {
          logger.warn(`Invalid submit_review payload: ${err instanceof Error ? err.message : err}`);
          throw err;
        }
        logger.info(
          `Received structured review via submit_review (${submittedReview.findings.length} finding(s))`,
        );
        return {
          content: [{
            type: "text",
            text: "Review received. Do not output the review as normal text.",
          }],
          details: {},
          terminate: true,
        };
      },
    };

    const { session } = await createAgentSession({
      cwd: workspacePath,
      model: piModel,
      thinkingLevel,
      ...getReviewSessionTools({
        singleTurn,
        reviewTools: reviewToolset.definitions,
        submitReviewTool,
        codemode: useCodemode,
      }),
      modelRuntime,
      sessionManager: SessionManager.inMemory(),
      settingsManager,
      resourceLoader,
    });
    activeSession = session;

    // Inject Hodor-specific Bedrock fields into stream requests.
    const openAiReasoning = thinkingLevel && isOpenAiBedrockModel(piModel)
      ? thinkingLevel
      : undefined;
    if (parsed.provider === "amazon-bedrock" && (bedrockTags || openAiReasoning)) {
      session.agent.streamFunction = wrapBedrockStream(session.agent.streamFunction, {
        bedrockTags,
        openAiReasoning,
      });
      if (bedrockTags) {
        logger.info(`Bedrock cost allocation tags: ${JSON.stringify(bedrockTags)}`);
      }
    }

    // Subscribe to agent events for progress + metrics tracking
    let turnCount = 0;
    let toolCallCount = 0;
    let nestedToolCallCount = 0;
    let codemodeCallCount = 0;

    /** Extract human-readable summary from tool args */
    function formatToolArgs(toolName: string, args: unknown): string {
      if (typeof args === "string") return args.slice(0, 200);
      const obj = args as Record<string, unknown> | undefined;
      if (!obj || Object.keys(obj).length === 0) return "";
      // submit_review: the outcome, not the payload
      if (toolName === "submit_review" && Array.isArray(obj.findings)) {
        const findings = obj.findings.length;
        const fixed = Array.isArray(obj.resolved_findings) ? obj.resolved_findings.length : 0;
        return `${findings} finding${findings === 1 ? "" : "s"}${fixed > 0 ? `, ${fixed} confirmed fixed` : ""}`;
      }
      // grep/find: show the quoted pattern + path
      if (obj.pattern) {
        const path = obj.path ? ` in ${obj.path}` : "";
        return `"${obj.pattern}"${path}`;
      }
      // codemode: the script size, not its source
      if (typeof obj.code === "string") {
        const lines = obj.code.trimEnd().split("\n").length;
        return `${lines}-line script`;
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
          if (event.parentToolCallId) nestedToolCallCount++;
          if (event.toolName === "codemode") codemodeCallCount++;
          onEvent?.({
            type: "tool_start",
            toolName: event.toolName,
            toolArgs: formatToolArgs(event.toolName, event.args),
            toolCallId: event.toolCallId,
            ...(event.parentToolCallId ? { parentToolCallId: event.parentToolCallId } : {}),
          });
          break;
        case "tool_execution_end":
          onEvent?.({
            type: "tool_end",
            toolName: event.toolName,
            isError: event.isError,
            result: formatToolResult(event.result),
            toolCallId: event.toolCallId,
            ...(event.parentToolCallId ? { parentToolCallId: event.parentToolCallId } : {}),
          });
          break;
        case "auto_retry_start":
          logger.info(
            `Retrying LLM request (attempt ${event.attempt}/${event.maxAttempts}) in ${event.delayMs}ms: ${event.errorMessage}`,
          );
          onEvent?.({
            type: "retry",
            phase: "start",
            attempt: event.attempt,
            maxAttempts: event.maxAttempts,
            delayMs: event.delayMs,
            reason: event.errorMessage,
          });
          break;
        case "auto_retry_end":
          logger.info(
            event.success
              ? `LLM retry succeeded on attempt ${event.attempt}`
              : `LLM retries exhausted after ${event.attempt} attempt(s): ${event.finalError ?? "unknown error"}`,
          );
          onEvent?.({
            type: "retry",
            phase: "end",
            attempt: event.attempt,
            success: event.success,
            reason: event.finalError,
          });
          break;
        case "compaction_start":
          logger.info(`Compacting context (reason: ${event.reason})`);
          onEvent?.({ type: "compaction", phase: "start", reason: event.reason });
          break;
        case "compaction_end":
          logger.info(
            `Compaction finished (reason: ${event.reason}, aborted: ${event.aborted}, will retry: ${event.willRetry})`,
          );
          onEvent?.({
            type: "compaction",
            phase: "end",
            reason: event.reason,
            success: !event.aborted && event.errorMessage === undefined,
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

    const throwIfAgentErrored = (): void => {
      // pi-agent-core stores failed/aborted assistant turns in state.errorMessage.
      const agentError = session.state.errorMessage;
      if (agentError) {
        throw new Error(`LLM request failed: ${agentError}`);
      }
    };

    const recoverReviewFromAssistantText = (source: string): boolean => {
      const rawText = session.getLastAssistantText() ?? "";
      if (!rawText.trim()) return false;

      const parsedReview = parseReviewFromAssistantText(rawText);
      if (!parsedReview) return false;

      submittedReview = parsedReview;
      logger.warn(
        `Recovered structured review from assistant text after ${source}; model did not call submit_review`,
      );
      return true;
    };

    logger.info("Sending prompt to agent...");
    await session.prompt(prompt);
    throwIfAgentErrored();

    if (!submittedReview) {
      recoverReviewFromAssistantText("initial agent run");
    }

    for (
      let attempt = 1;
      !submittedReview && attempt <= SUBMIT_REVIEW_RECOVERY_ATTEMPTS;
      attempt++
    ) {
      logger.warn(
        `Agent ended without a valid submit_review (${summarizeLastAssistantMessage(session)}); ` +
        `requesting recovery ${attempt}/${SUBMIT_REVIEW_RECOVERY_ATTEMPTS}`,
      );
      await session.prompt(buildSubmitReviewRecoveryPrompt(attempt, SUBMIT_REVIEW_RECOVERY_ATTEMPTS));
      throwIfAgentErrored();
      recoverReviewFromAssistantText(`recovery attempt ${attempt}`);
    }

    if (!submittedReview) {
      const diagnostic = summarizeLastAssistantMessage(session);
      if (submitReviewCalls > 0) {
        throw new Error(
          `Agent called submit_review but did not provide a valid review payload after ` +
          `${SUBMIT_REVIEW_RECOVERY_ATTEMPTS} recovery attempt(s): ${diagnostic}`,
        );
      }
      throw new Error(
        `Agent did not call submit_review after ${SUBMIT_REVIEW_RECOVERY_ATTEMPTS} recovery attempt(s): ${diagnostic}`,
      );
    }

    const rawReview = submittedReview as ReviewOutput;
    if (submitReviewCalls > 1) {
      logger.warn(`Agent called submit_review ${submitReviewCalls} times; using the first valid submission`);
    }

    // Resolve each finding's line_range from its quoted snippet against the
    // checked-out file, correcting model line-number errors before posting.
    const { review: locatedReview, stats: locationStats } = resolveReviewLocations(rawReview, {
      trackedTree: reviewToolset.tree,
      diffText: embeddedDiff,
    });
    if (locationStats.corrected > 0 || locationStats.unmatched > 0) {
      logger.info(
        `Location resolution: ${locationStats.corrected} corrected, ${locationStats.confirmed} confirmed, ` +
          `${locationStats.unmatched} unmatched, ${locationStats.noSnippet} without snippet`,
      );
    }

    // The model's resolved_findings is a claim. Keep only ids shown in this
    // prompt, on files in the reviewed diff, and not reported again now.
    const { accepted: verifiedFixes, rejected: rejectedFixes } = selectVerifiedFixes(
      locatedReview.resolved_findings ?? [],
      fixCandidates,
      {
        changedFiles,
        currentFingerprints: new Set(
          locatedReview.findings.map((finding) => getFindingFingerprint(finding, workspacePath)),
        ),
      },
    );
    for (const { id, reason } of rejectedFixes) {
      logger.info(`Ignoring resolved_findings id ${JSON.stringify(id.slice(0, 80))}: ${reason}`);
    }
    const review: ReviewOutput = {
      findings: locatedReview.findings,
      overall_correctness: locatedReview.overall_correctness,
      overall_explanation: locatedReview.overall_explanation,
      ...(verifiedFixes.length > 0 ? { resolved_findings: verifiedFixes } : {}),
    };

    logger.info(
      `Captured ${review.findings.length} finding(s), ${verifiedFixes.length} verified fix(es), verdict: ${review.overall_correctness}`,
    );

    const durationSeconds = (Date.now() - startTime) / 1000;
    logger.info(`Review complete (${review.findings.length} finding(s))`);

    // session.messages is the projected context and omits compacted or retried
    // attempts, so read usage from the session stats instead.
    const { tokens, cost } = session.getSessionStats();

    const metrics: ReviewMetrics = {
      inputTokens: tokens.input,
      outputTokens: tokens.output,
      cacheReadTokens: tokens.cacheRead,
      cacheWriteTokens: tokens.cacheWrite,
      totalTokens: tokens.total,
      cost,
      turns: turnCount,
      toolCalls: toolCallCount,
      ...(useCodemode ? { nestedToolCalls: nestedToolCallCount, codemodeCalls: codemodeCallCount } : {}),
      durationSeconds: Math.round(durationSeconds),
      reviewMode,
      reasoningEffort: thinkingLevel ?? "none",
      diffFiles: diffStats?.files ?? 0,
      diffAdditions: diffStats?.additions ?? 0,
      diffDeletions: diffStats?.deletions ?? 0,
      diffBytes: diffStats?.bytes ?? 0,
      diffEmbedded: embeddedDiff != null,
      reused: false,
      fastPath: singleTurn,
    };
    logger.info(`Review telemetry: ${JSON.stringify({
      project: localMode ? null : `${owner}/${repo}`,
      mr: localMode ? null : prNumber,
      headSha: headSha?.slice(0, 12) ?? null,
      model,
      outcome: "reviewed",
      reviewMode: metrics.reviewMode,
      reasoningEffort: metrics.reasoningEffort,
      fastPath: singleTurn,
      reused: false,
      diffFiles: metrics.diffFiles,
      diffAdditions: metrics.diffAdditions,
      diffDeletions: metrics.diffDeletions,
      diffBytes: metrics.diffBytes,
      turns: metrics.turns,
      toolCalls: metrics.toolCalls,
      inputTokens: metrics.inputTokens,
      cacheReadTokens: metrics.cacheReadTokens,
      cacheWriteTokens: metrics.cacheWriteTokens,
      outputTokens: metrics.outputTokens,
      cost: metrics.cost,
      findings: review.findings.length,
    })}`);

    let metricsFooter: string | null = null;
    if (includeMetricsFooter) {
      metricsFooter = formatMetricsMarkdown(metrics);
    }

    const cacheMarker = reviewCacheKey
      ? buildReviewCacheMarker(reviewCacheKey, review, workspacePath)
      : null;

    return {
      review,
      metricsFooter,
      headSha,
      metrics,
      workspacePath,
      cacheMarker,
      reusedReview: false,
      range: {
        headSha,
        targetBranch,
        // A rebased GitLab MR is reviewed against the MR base, not the old SHA.
        baseSha: previousReviewSha && !(platform === "gitlab" && reviewMode === "snapshot")
          ? previousReviewSha
          : reviewBaseSha,
      },
      context,
    };
  } finally {
    activeSession?.dispose();
    reviewToolset?.dispose();

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
