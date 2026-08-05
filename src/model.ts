import { getEnvApiKey } from "@earendil-works/pi-ai/compat";
import { getBuiltinProviders } from "@earendil-works/pi-ai/providers/all";
import type { Api, Model, ThinkingLevel } from "@earendil-works/pi-ai";
import type { DiffStats, ReviewDiffMode } from "./review-diff.js";

export interface ParsedModel {
  provider: string;
  modelId: string;
  /**
   * Registry model id whose descriptor should back a custom Bedrock ARN.
   * Application inference profile ARNs do not contain the model name, and
   * pi-ai gates prompt caching, adaptive thinking, and cost accounting on
   * substring matches against the model id/name. Without this hint those
   * capabilities silently switch off.
   */
  baseModelId?: string;
}

const PROVIDER_ALIASES: Record<string, string> = {
  bedrock: "amazon-bedrock",
};

/**
 * Parse a model string like "anthropic/claude-sonnet-4-5" into { provider, modelId }.
 * Handles bare names like "claude-sonnet-4-5" or "gpt-5" via auto-detection.
 */
export function parseModelString(model: string): ParsedModel {
  const trimmed = model.trim();
  if (!trimmed) throw new Error("Model name must be provided");

  const parts = trimmed.split("/");

  // Explicit provider prefix. Hodor delegates provider/model support to pi-ai's
  // registry instead of maintaining its own curated allow-list. Keep `bedrock`
  // as a friendlier alias for pi-ai's `amazon-bedrock` provider name.
  if (parts.length >= 2) {
    const first = parts[0].toLowerCase();
    const provider = PROVIDER_ALIASES[first] ?? first;
    const knownProviders = new Set<string>(getBuiltinProviders());

    if (provider === "amazon-bedrock") {
      // Strip optional "converse/" prefix from model ID for backwards compatibility.
      let modelId = parts.slice(1).join("/");
      if (modelId.startsWith("converse/")) {
        modelId = modelId.slice("converse/".length);
      }
      // "<arn>@<base-model-id>" names the registry model backing an inference
      // profile ARN. "@" cannot appear in a Bedrock ARN, so it is unambiguous.
      const separator = modelId.indexOf("@");
      if (separator !== -1) {
        const baseModelId = modelId.slice(separator + 1).trim();
        modelId = modelId.slice(0, separator).trim();
        if (!modelId || !baseModelId) {
          throw new Error(
            `Invalid bedrock model "${trimmed}". Use "bedrock/<arn>@<base-model-id>", ` +
              `for example "bedrock/arn:aws:bedrock:ap-south-1:123:application-inference-profile/abc@global.anthropic.claude-opus-5".`,
          );
        }
        return { provider, modelId, baseModelId };
      }
      return { provider, modelId };
    }

    if (knownProviders.has(provider)) {
      return { provider, modelId: parts.slice(1).join("/") };
    }

    // OpenRouter adds new model slugs frequently. Allow it even if the installed
    // pi-ai registry ever lags; agent.ts has a conservative OpenRouter fallback.
    if (provider === "openrouter") {
      return { provider, modelId: parts.slice(1).join("/") };
    }

    throw new Error(
      `Unsupported provider "${first}". Use a pi-ai provider prefix such as anthropic/, openai/, openrouter/, google/, mistral/, xai/, or bedrock/.`,
    );
  }

  // Auto-detect provider from bare model name
  const lower = trimmed.toLowerCase();
  if (lower.includes("claude") || lower.includes("anthropic")) {
    return { provider: "anthropic", modelId: trimmed };
  }
  if (
    lower.startsWith("gpt") ||
    lower.startsWith("o1") ||
    lower.startsWith("o3") ||
    lower.startsWith("o4") ||
    lower.includes("openai")
  ) {
    return { provider: "openai", modelId: trimmed };
  }

  // Default to anthropic for unknown models
  return { provider: "anthropic", modelId: trimmed };
}

/** Region from arn:aws:bedrock:<region>:<account>:..., falling back to us-east-1. */
export function extractBedrockArnRegion(arn: string): string {
  const parts = arn.split(":");
  return parts.length >= 4 && parts[3] ? parts[3] : "us-east-1";
}

/**
 * Build the model descriptor for a custom Bedrock ARN.
 *
 * pi-ai decides prompt caching, adaptive thinking, and per-token cost by
 * substring-matching the model id and name. An application inference profile
 * ARN — the only mechanism AWS bills against cost allocation tags — carries
 * neither the vendor nor the model version, so a hand-rolled descriptor
 * silently loses caching, drops the thinking config, and reports zero cost.
 * Inheriting the registry descriptor and overriding only the id keeps every
 * capability matcher working behind the opaque ARN.
 */
export function buildBedrockArnModel(opts: {
  arn: string;
  baseModel?: Model<Api> | null;
  region?: string;
}): Model<Api> {
  const region = opts.region ?? extractBedrockArnRegion(opts.arn);
  const baseUrl = `https://bedrock-runtime.${region}.amazonaws.com`;

  if (opts.baseModel) {
    return {
      ...opts.baseModel,
      id: opts.arn,
      // Keep the registry id as the name so pi-ai's capability matchers still
      // see the vendor and model version behind the ARN.
      name: opts.baseModel.id,
      baseUrl,
    } as Model<Api>;
  }

  return {
    id: opts.arn,
    name: opts.arn,
    api: "bedrock-converse-stream",
    provider: "amazon-bedrock",
    baseUrl,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 200000,
    maxTokens: 16384,
  } as Model<Api>;
}

/**
 * Map reasoning effort strings to pi-ai thinking levels.
 * Returns undefined for no reasoning.
 */
export function mapReasoningEffort(
  effort: string | undefined,
): ThinkingLevel | undefined {
  if (!effort) return undefined;
  switch (effort.toLowerCase()) {
    case "minimal":
      return "minimal";
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
    case "xhigh":
      return "xhigh";
    case "max":
      return "max";
    default:
      return undefined;
  }
}

function normalizeModelMatchValue(value: string): string {
  return value.toLowerCase().replace(/[\s_.:/]+/g, "-");
}

export function getDefaultReasoningEffortForModel(model: {
  id?: string;
  name?: string;
}): ThinkingLevel | undefined {
  const values = [model.id, model.name].filter((value): value is string => Boolean(value));
  const isAdaptiveOpus = values
    .map(normalizeModelMatchValue)
    .some((value) => value.includes("opus-4-7") || value.includes("opus-4-8") || value.includes("opus-5"));

  return isAdaptiveOpus ? "xhigh" : undefined;
}

const HIGH_RISK_PATH_RE = /(?:^|\/)(?:migrations?|schema|auth|security|permissions?|crypto|iam)(?:\/|\.|$)|\.tf$/im;
const HIGH_RISK_CHANGE_RE = /^\+.*\b(?:authorization|authentication|permission|transaction|mutex|semaphore|encrypt|decrypt|credential|secret)\b/im;

/** Diffs touching security-sensitive paths or semantics keep full investigation budget. */
export function isHighRiskDiff(diff: string): boolean {
  return HIGH_RISK_PATH_RE.test(diff) || HIGH_RISK_CHANGE_RE.test(diff);
}

const SINGLE_TURN_MAX_FILES = 5;
const SINGLE_TURN_MAX_CHANGED_LINES = 50;

/**
 * Whether the embedded diff is the whole story, so the review can be decided in
 * one turn without exploratory tool calls.
 *
 * Tiny reviews spend most of their cost re-sending the conversation on each
 * exploratory turn rather than on the diff itself, so removing those turns
 * removes both the repeated context and the reasoning that drives it. High-risk
 * diffs keep tool access regardless of size.
 */
export function qualifiesForSingleTurnReview(opts: {
  diff?: string | null;
  stats?: DiffStats | null;
  embedded: boolean;
}): boolean {
  if (!opts.embedded || !opts.stats) return false;
  const { files, additions, deletions } = opts.stats;
  if (files === 0 || files > SINGLE_TURN_MAX_FILES) return false;
  if (additions + deletions > SINGLE_TURN_MAX_CHANGED_LINES) return false;
  if (opts.diff && isHighRiskDiff(opts.diff)) return false;
  return true;
}

export function selectReasoningEffort(opts: {
  requested?: string;
  modelDefault?: ThinkingLevel;
  mode: ReviewDiffMode;
  forcedFull?: boolean;
  diff?: string | null;
  stats?: DiffStats | null;
}): ThinkingLevel | undefined {
  const requested = mapReasoningEffort(opts.requested);
  if (requested) return requested;

  const modelDefault = opts.modelDefault;
  if (modelDefault !== "xhigh") return modelDefault;
  if (opts.forcedFull) return modelDefault;

  if (opts.diff && isHighRiskDiff(opts.diff)) return modelDefault;

  if (opts.mode === "incremental" || opts.mode === "snapshot") return "high";

  const changedLines = (opts.stats?.additions ?? 0) + (opts.stats?.deletions ?? 0);
  if (opts.stats && opts.stats.files <= 10 && changedLines <= 500) return "high";

  return modelDefault;
}

/**
 * Get API key with provider-aware selection.
 *
 * Priority:
 * 1. LLM_API_KEY (universal override)
 * 2. Provider-specific key known by pi-ai (ANTHROPIC_API_KEY, OPENAI_API_KEY,
 *    OPENROUTER_API_KEY, etc.)
 *
 * Returns null for bedrock (uses AWS credentials).
 */
export function getApiKey(model?: string): string | null {
  // Priority 1: Universal override
  const llmKey = process.env.LLM_API_KEY;
  if (llmKey) return llmKey;

  // Priority 2: Provider-specific
  if (model) {
    const { provider } = parseModelString(model);
    if (provider === "amazon-bedrock") return null;
    const key = getEnvApiKey(provider);
    if (key) return key;
  }

  throw new Error(
    model
      ? `No API key found for provider "${parseModelString(model).provider}". Set the provider-specific environment variable or LLM_API_KEY.`
      : "No LLM API key found. Please set LLM_API_KEY or a provider-specific environment variable.",
  );
}
