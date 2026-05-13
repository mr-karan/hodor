import { getEnvApiKey, getProviders } from "@earendil-works/pi-ai";

export interface ParsedModel {
  provider: string;
  modelId: string;
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
    const knownProviders = new Set<string>(getProviders());

    if (provider === "amazon-bedrock") {
      // Strip optional "converse/" prefix from model ID for backwards compatibility.
      let modelId = parts.slice(1).join("/");
      if (modelId.startsWith("converse/")) {
        modelId = modelId.slice("converse/".length);
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

/**
 * Map reasoning effort strings to pi-ai thinking levels.
 * Returns undefined for no reasoning.
 */
export function mapReasoningEffort(
  effort: string | undefined,
): "low" | "medium" | "high" | undefined {
  if (!effort) return undefined;
  switch (effort.toLowerCase()) {
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
    case "xhigh":
      return "high";
    default:
      return undefined;
  }
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
