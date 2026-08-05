import { existsSync } from "node:fs";
import { join } from "node:path";
import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import type { AgentSession, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import type { Api, Model } from "@earendil-works/pi-ai";
import { logger } from "./utils/logger.js";
import { exec } from "./utils/exec.js";
import {
  fetchGithubPrInfo,
  normalizeGithubMetadata,
} from "./github.js";
import {
  fetchGitlabMrInfo,
} from "./gitlab.js";
import {
  fetchGiteaPrInfo,
} from "./gitea.js";
import { setupWorkspace, cleanupWorkspace } from "./workspace.js";
import { buildPrReviewPrompt } from "./prompt.js";
import {
  buildBedrockArnModel,
  extractBedrockArnRegion,
  getDefaultReasoningEffortForModel,
  parseModelString,
  qualifiesForSingleTurnReview,
  selectReasoningEffort,
} from "./model.js";
import { formatMetricsMarkdown, printMetrics } from "./metrics.js";
import { SUBMIT_REVIEW_SCHEMA, validateReviewOutput } from "./review.js";
import { resolveReviewLocations } from "./resolve-location.js";
import { buildReviewSystemPrompt } from "./system-prompt.js";
import {
  loadDefaultReviewInstructions,
  validateReviewInstructions,
} from "./review-instructions.js";
import { detectPlatform, parsePrUrl } from "./platform.js";
import {
  filterEmbeddedDiff,
  findLatestReviewBase,
  getChangedFiles,
  getDiffStats,
  type DiffStats,
  type ReviewDiffMode,
} from "./review-diff.js";
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
  ReviewMetrics,
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
}): Promise<{
  review: ReviewOutput;
  metricsFooter: string | null;
  headSha: string | null;
  metrics: ReviewMetrics;
  workspacePath: string;
  cacheMarker: string | null;
  reusedReview: boolean;
}> {
  const {
    prUrl,
    model = "anthropic/claude-sonnet-4-5-20250929",
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
  const parsed = parseModelString(model);

  // Snapshot env vars we may mutate, restore in finally block.
  const envSnapshot: Record<string, string | undefined> = {
    AWS_REGION: process.env.AWS_REGION,
    AWS_DEFAULT_REGION: process.env.AWS_DEFAULT_REGION,
  };

  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsPath: null,
  });
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
    if (parsed.provider === "openrouter") {
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

    // Get HEAD SHA for embedding in posted comments (skip in local mode — no posting)
    let headSha: string | null = null;
    if (!localMode) {
      const { stdout: headShaRaw } = await exec("git", ["rev-parse", "HEAD"], { cwd: workspacePath });
      headSha = headShaRaw.trim();
    }

    // A successful Hodor summary contains a compressed, validated copy of the
    // structured result. Reuse it for an identical review identity so pipeline
    // retries can regenerate artifacts and retry delivery without another LLM
    // invocation. Explicit --full reviews always bypass this fast path.
    let reviewCacheKey: string | null = null;
    if (!localMode && !full && headSha) {
      reviewCacheKey = getReviewCacheKey({
        headSha,
        model,
        requestedReasoningEffort: reasoningEffort,
        reviewInstructions: effectiveReviewInstructions,
        additionalInstructions: effectiveAdditionalInstructions,
      });
      const cachedReview = findCachedReview(mrMetadata?.Notes, reviewCacheKey);
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
        printMetrics(metrics);
        return {
          review: cachedReview,
          metricsFooter: includeMetricsFooter ? formatMetricsMarkdown(metrics) : null,
          headSha,
          metrics,
          workspacePath,
          cacheMarker: null,
          reusedReview: true,
        };
      }
    }

    // Prefer the latest reviewed commit. Preserve three-dot semantics while it
    // is an ancestor; after a force-push/rebase, use a direct snapshot delta.
    const previousReviewBase = full || localMode
      ? null
      : await findLatestReviewBase(mrMetadata?.Notes, workspacePath);
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
    let reviewDiff: string | null = null;
    let diffStats: DiffStats | null = null;
    let changedFiles: string[] = [];
    try {
      const diffArgs = previousReviewSha
        ? previousReviewBase?.mode === "snapshot"
          ? ["--no-pager", "diff", previousReviewSha, "HEAD"]
          : ["--no-pager", "diff", `${previousReviewSha}...HEAD`]
        : diffBaseSha
          ? ["--no-pager", "diff", diffBaseSha, "HEAD"]
          : localMode
            ? ["--no-pager", "diff", targetBranch]  // includes uncommitted changes
            : ["--no-pager", "diff", `origin/${targetBranch}...HEAD`];
      const { stdout: rawDiff } = await exec("git", diffArgs, { cwd: workspacePath });
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
        logger.info(`Diff too large to embed (${Buffer.byteLength(filteredDiff, "utf-8")} bytes filtered, ${Buffer.byteLength(rawDiff, "utf-8")} bytes raw), using command mode`);
      }
    } catch (err) {
      logger.warn(`Failed to pre-fetch diff, falling back to command mode: ${err}`);
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

    // Build the dynamic review task sent as the first user message.
    const prompt = buildPrReviewPrompt({
      prUrl: prUrl ?? `local diff (against ${targetBranch})`,
      platform,
      targetBranch,
      diffBaseSha,
      mrMetadata,
      embeddedDiff,
      previousReviewSha,
      reviewDiffMode: reviewMode,
      changedFiles,
      localMode,
      singleTurn,
    });

    const startTime = Date.now();
    const settingsManager = SettingsManager.inMemory({
      compaction: { enabled: true },
    });
    const skillPaths = [join(workspacePath, ".agents", "skills")]
      .filter((p) => existsSync(p));
    const resourceLoader = new DefaultResourceLoader({
      cwd: workspacePath,
      agentDir: getAgentDir(),
      settingsManager,
      systemPromptOverride: () => composedSystemPrompt,
      appendSystemPromptOverride: () => [],
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
      // pi v0.74 filters customTools through the same allowlist as built-ins
      // (see _refreshToolRegistry in @earendil-works/pi-coding-agent's
      // agent-session.ts). The submit_review custom tool must be named here
      // or the LLM never sees it and the agent loop exits without calling it.
      tools: singleTurn
        ? ["submit_review"]
        : ["read", "bash", "grep", "find", "ls", "submit_review"],
      customTools: [submitReviewTool],
      modelRuntime,
      sessionManager: SessionManager.inMemory(),
      settingsManager,
      resourceLoader,
    });
    activeSession = session;

    // Inject Bedrock cost allocation tags into stream requests
    if (bedrockTags && parsed.provider === "amazon-bedrock") {
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
    const { review, stats: locationStats } = resolveReviewLocations(rawReview, {
      workspacePath,
      diffText: embeddedDiff,
    });
    if (locationStats.corrected > 0 || locationStats.unmatched > 0) {
      logger.info(
        `Location resolution: ${locationStats.corrected} corrected, ${locationStats.confirmed} confirmed, ` +
          `${locationStats.unmatched} unmatched, ${locationStats.noSnippet} without snippet`,
      );
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

    const allMessages = session.messages as AssistantMsg[];

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
      reviewMode,
      reasoningEffort: thinkingLevel ?? "none",
      diffFiles: diffStats?.files ?? 0,
      diffAdditions: diffStats?.additions ?? 0,
      diffDeletions: diffStats?.deletions ?? 0,
      diffBytes: diffStats?.bytes ?? 0,
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
    printMetrics(metrics);

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
    };
  } finally {
    activeSession?.dispose();

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
