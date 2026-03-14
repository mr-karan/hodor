import { existsSync } from "node:fs";
import { join } from "node:path";
import type { Model } from "@mariozechner/pi-ai";
import type { ToolDefinition } from "@mariozechner/pi-coding-agent";
import { logger } from "./utils/logger.js";
import { exec } from "./utils/exec.js";
import {
  fetchGithubPrInfo,
  normalizeGithubMetadata,
} from "./github.js";
import {
  fetchGitlabMrInfo,
  postGitlabMrComment,
} from "./gitlab.js";
import { setupWorkspace, cleanupWorkspace } from "./workspace.js";
import { buildPrReviewPrompt } from "./prompt.js";
import { parseModelString, mapReasoningEffort } from "./model.js";
import { formatMetricsMarkdown, printMetrics } from "./metrics.js";
import { SUBMIT_REVIEW_SCHEMA, validateReviewOutput } from "./review.js";
import { REVIEW_SYSTEM_PROMPT } from "./system-prompt.js";
import type {
  Platform,
  ParsedPrUrl,
  ReviewMetrics,
  PostCommentResult,
  MrMetadata,
  ReviewOutput,
} from "./types.js";

export interface AgentProgressEvent {
  type: "tool_start" | "tool_end" | "turn_start" | "turn_end" | "agent_start" | "agent_end" | "assistant_text" | "assistant_thinking";
  toolName?: string;
  toolArgs?: string;
  isError?: boolean;
  turnIndex?: number;
  result?: string;
  fullOutputPath?: string;
  truncated?: boolean;
  message?: string;
  stopReason?: string;
  toolCount?: number;
}

interface AssistantContentBlockLike {
  type?: string;
  text?: string;
  thinking?: string;
}

interface AssistantMessageLike {
  role?: string;
  content?: AssistantContentBlockLike[];
  stopReason?: string;
}

interface ToolResultPayload {
  text: string;
  fullOutputPath?: string;
  truncated: boolean;
}

function asAssistantMessage(value: unknown): AssistantMessageLike | null {
  if (!value || typeof value !== "object") return null;
  const message = value as AssistantMessageLike;
  return message.role === "assistant" && Array.isArray(message.content)
    ? message
    : null;
}

export function extractAssistantBlobs(
  message: unknown,
  type: "text" | "thinking",
): string[] {
  const assistantMessage = asAssistantMessage(message);
  if (!assistantMessage) return [];
  const content = assistantMessage.content ?? [];

  return content
    .filter((block) => block.type === type)
    .map((block) => (type === "text" ? block.text : block.thinking) ?? "")
    .map((blob) => blob.trim())
    .filter((blob) => blob.length > 0);
}

export function formatTurnSummary(message: unknown, toolResults: unknown[]): string {
  const assistantMessage = asAssistantMessage(message);
  const summaryParts: string[] = [];

  if (assistantMessage?.stopReason) {
    summaryParts.push(`stop=${assistantMessage.stopReason}`);
  }
  if (toolResults.length > 0) {
    summaryParts.push(`tool_results=${toolResults.length}`);
  }
  const textBlocks = extractAssistantBlobs(message, "text");
  if (textBlocks.length > 0) {
    summaryParts.push(`text_blobs=${textBlocks.length}`);
  }

  return summaryParts.length > 0 ? summaryParts.join(", ") : "no assistant output";
}

function buildMissingAuthError(provider: string, model: string): Error {
  if (provider === "openai-codex") {
    return new Error(
      `No authentication found for "${model}". Hodor can reuse pi credentials from ~/.pi/agent/auth.json. ` +
      `Run "pi", use "/login", select "ChatGPT Plus/Pro (Codex Subscription)", then retry.`,
    );
  }

  return new Error(
    `No authentication found for "${model}". Hodor checks LLM_API_KEY, provider-specific environment variables, ` +
    `and pi auth storage (~/.pi/agent/auth.json).`,
  );
}

export function detectPlatform(prUrl: string): Platform {
  const url = new URL(prUrl);
  const hostname = url.hostname;
  if (prUrl.includes("/-/merge_requests/") || hostname.includes("gitlab")) {
    return "gitlab";
  }
  if (prUrl.includes("/pull/") || hostname.includes("github")) {
    return "github";
  }
  throw new Error(
    `Cannot detect platform for URL: ${prUrl}. Expected a GitHub pull request (/pull/) or GitLab merge request (/-/merge_requests/) URL.`,
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
    `Invalid PR/MR URL format: ${prUrl}. Expected GitHub pull request or GitLab merge request URL.`,
  );
}

export async function postReviewComment(opts: {
  prUrl: string;
  reviewText: string;
  model?: string | null;
  metricsFooter?: string | null;
}): Promise<PostCommentResult> {
  const { prUrl, reviewText, model, metricsFooter } = opts;
  const platform = detectPlatform(prUrl);
  logger.info(`Posting comment to ${platform} PR/MR: ${prUrl}`);

  let parsed: ParsedPrUrl;
  try {
    parsed = parsePrUrl(prUrl);
  } catch (err) {
    return { success: false, error: String(err) };
  }

  let body = reviewText;
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

export async function reviewPr(opts: {
  prUrl: string;
  model?: string;
  reasoningEffort?: string;
  customPrompt?: string | null;
  promptFile?: string | null;
  cleanup?: boolean;
  workspaceDir?: string | null;
  includeMetricsFooter?: boolean;
  onEvent?: (event: AgentProgressEvent) => void;
}): Promise<{ review: ReviewOutput; metricsFooter: string | null }> {
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
  } = opts;

  logger.info(`Starting PR review for: ${prUrl}`);

  // Parse PR URL
  const { owner, repo, prNumber, host } = parsePrUrl(prUrl);
  const platform = detectPlatform(prUrl);
  logger.info(`Platform: ${platform}, Repo: ${owner}/${repo}, PR: ${prNumber}, Host: ${host}`);

  // --- Preflight: validate model + credentials before any expensive I/O ---
  const parsed = parseModelString(model);
  const thinkingLevel = mapReasoningEffort(reasoningEffort);

  // Note: For bedrock, we don't preflight-check AWS credentials because the
  // SDK resolves them from many sources (env vars, IMDS, ECS task role, IRSA,
  // ~/.aws/credentials, etc.) and we can't reliably detect all of them.
  // If credentials are missing, the SDK will fail with a clear error.

  // Snapshot env vars we may mutate, restore in finally block
  const envSnapshot: Record<string, string | undefined> = {
    AWS_REGION: process.env.AWS_REGION,
  };

  // Import pi SDK
  const {
    AuthStorage,
    ModelRegistry,
    createAgentSession,
    DefaultResourceLoader,
    SessionManager,
    SettingsManager,
    createReadTool,
    createBashTool,
    createGrepTool,
    createFindTool,
    createLsTool,
  } = await import("@mariozechner/pi-coding-agent");

  const authStorage = AuthStorage.create();
  if (process.env.LLM_API_KEY && parsed.provider !== "amazon-bedrock") {
    authStorage.setRuntimeApiKey(parsed.provider, process.env.LLM_API_KEY);
  }
  const modelRegistry = new ModelRegistry(authStorage);

  // Resolve model — use registry for known models, construct manually for custom ARNs
  let piModel: Model<any>;
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
    } as Model<any>;
    logger.info(`Custom bedrock ARN model — region: ${region}`);
  } else {
    const resolvedModel = modelRegistry.find(parsed.provider, parsed.modelId);
    if (!resolvedModel) {
      throw new Error(
        `Unsupported model "${model}".`,
      );
    }
    piModel = resolvedModel;
  }

  if (parsed.provider !== "amazon-bedrock") {
    const providerApiKey = await modelRegistry.getApiKey(piModel);
    if (!providerApiKey) {
      throw buildMissingAuthError(parsed.provider, model);
    }
    if (!process.env.LLM_API_KEY && modelRegistry.isUsingOAuth(piModel)) {
      logger.info(`Using pi auth storage credentials for ${parsed.provider}`);
    }
  }
  logger.info("Preflight OK — model and credentials validated");

  // --- End preflight ---

  // Setup workspace
  const { workspace, targetBranch, diffBaseSha, isTemporary } = await setupWorkspace({
    platform,
    owner,
    repo,
    prNumber: String(prNumber),
    host,
    workingDir: workspaceDir ?? undefined,
    reuse: workspaceDir != null,
  });

  const workspacePath = workspace;

  try {
    // Fetch PR metadata
    let mrMetadata: MrMetadata | null = null;
    if (platform === "gitlab") {
      try {
        mrMetadata = await fetchGitlabMrInfo(owner, repo, prNumber, host, {
          includeComments: true,
        });
      } catch (err) {
        logger.warn(`Failed to fetch GitLab metadata: ${err}`);
      }
    } else if (platform === "github") {
      try {
        const githubRaw = await fetchGithubPrInfo(owner, repo, prNumber);
        mrMetadata = normalizeGithubMetadata(githubRaw);
      } catch (err) {
        logger.warn(`Failed to fetch GitHub metadata: ${err}`);
      }
    }

    // Build prompt (always uses JSON template; rendered to markdown post-hoc)
    const prompt = buildPrReviewPrompt({
      prUrl,
      platform,
      targetBranch,
      diffBaseSha,
      mrMetadata,
      customInstructions: customPrompt,
      customPromptFile: promptFile,
    });

    const startTime = Date.now();
    const settingsManager = SettingsManager.inMemory({
      compaction: { enabled: false },
    });
    const skillPaths = [
      join(workspacePath, ".pi", "skills"),
      join(workspacePath, ".hodor", "skills"),
    ].filter((p) => existsSync(p));
    const resourceLoader = new DefaultResourceLoader({
      cwd: workspacePath,
      settingsManager,
      systemPrompt: REVIEW_SYSTEM_PROMPT,
      appendSystemPrompt: "",
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
      authStorage,
      modelRegistry,
      model: piModel,
      thinkingLevel,
      tools: [
        createReadTool(workspacePath),
        createBashTool(workspacePath),
        createGrepTool(workspacePath),
        createFindTool(workspacePath),
        createLsTool(workspacePath),
      ],
      customTools: [submitReviewTool],
      sessionManager: SessionManager.inMemory(),
      settingsManager,
      resourceLoader,
    });

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

    /** Extract text content and metadata from tool result */
    function formatToolResult(result: unknown): ToolResultPayload {
      if (typeof result === "string") {
        return { text: result, truncated: false };
      }
      const obj = result as Record<string, unknown> | undefined;
      if (!obj) return { text: "", truncated: false };
      // pi-sdk wraps results as {content: [{type: "text", text: "..."}]}
      const content = obj.content as Array<{ type?: string; text?: string }> | undefined;
      const details = obj.details as Record<string, unknown> | undefined;
      const fullOutputPath =
        typeof details?.fullOutputPath === "string" ? details.fullOutputPath : undefined;
      const truncated = Boolean(details?.truncation) || Boolean(fullOutputPath);
      if (Array.isArray(content)) {
        return {
          text: content
            .filter((c) => c.type === "text" && c.text)
            .map((c) => c.text)
            .join("\n"),
          fullOutputPath,
          truncated,
        };
      }
      return {
        text: JSON.stringify(result, null, 2) ?? "",
        fullOutputPath,
        truncated,
      };
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
          onEvent?.({
            type: "turn_end",
            turnIndex: turnCount,
            message: formatTurnSummary(event.message, event.toolResults),
            stopReason:
              typeof (event.message as { stopReason?: unknown })?.stopReason === "string"
                ? (event.message as { stopReason: string }).stopReason
                : undefined,
            toolCount: event.toolResults.length,
          });
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
          {
            const toolResult = formatToolResult(event.result);
          onEvent?.({
            type: "tool_end",
            toolName: event.toolName,
            isError: event.isError,
            result: toolResult.text,
            fullOutputPath: toolResult.fullOutputPath,
            truncated: toolResult.truncated,
          });
          }
          break;
        case "message_update": {
          const msgEvent = (event as Record<string, unknown>).assistantMessageEvent as
            { type: string; content?: string } | undefined;
          if (!msgEvent?.content) break;
          if (msgEvent.type === "text_end") {
            onEvent?.({ type: "assistant_text", message: msgEvent.content.trim() });
          } else if (msgEvent.type === "thinking_end") {
            onEvent?.({ type: "assistant_thinking", message: msgEvent.content.trim() });
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

    return { review, metricsFooter };
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
