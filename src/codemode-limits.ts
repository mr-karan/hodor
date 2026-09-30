import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { isRecord } from "./utils/json.js";

/** Codemode's options line: an optional first line of the script. */
const OPTIONS_PREFIX = "// @options:";

/**
 * Codemode applies no timeout unless the script asks for one, so a
 * prompt-injected `while (true) {}` would hold the review until the CI job
 * timeout. Hodor enforces these caps on every script.
 */
export const CODEMODE_TIMEOUT_MS = 120_000;
/** Codemode's own default. Capping here stops a script from raising it. */
export const CODEMODE_MAX_OUTPUT_TOKENS = 10_000;

export interface CodemodeLimits {
  timeoutMs: number;
  maxOutputTokens: number;
}

const DEFAULT_LIMITS: CodemodeLimits = {
  timeoutMs: CODEMODE_TIMEOUT_MS,
  maxOutputTokens: CODEMODE_MAX_OUTPUT_TOKENS,
};

function cap(requested: unknown, limit: number): number {
  return typeof requested === "number" && Number.isSafeInteger(requested) && requested > 0 && requested < limit
    ? requested
    : limit;
}

/**
 * Rewrite a codemode script so its `timeout_ms` and `max_output_tokens`
 * never exceed Hodor's limits. Lower values the model chose are kept. A
 * malformed options line is left as is, so codemode rejects the script.
 */
export function capCodemodeOptions(code: string, limits: CodemodeLimits = DEFAULT_LIMITS): string {
  const newline = code.indexOf("\n");
  const firstLine = (newline === -1 ? code : code.slice(0, newline)).replace(/\r$/, "").trimStart();

  if (!firstLine.startsWith(OPTIONS_PREFIX)) {
    const options = { timeout_ms: limits.timeoutMs, max_output_tokens: limits.maxOutputTokens };
    return `${OPTIONS_PREFIX} ${JSON.stringify(options)}\n${code}`;
  }

  let requested: unknown;
  try {
    requested = JSON.parse(firstLine.slice(OPTIONS_PREFIX.length).trim());
  } catch {
    return code;
  }
  if (!isRecord(requested)) return code;

  const options = {
    ...requested,
    timeout_ms: cap(requested.timeout_ms, limits.timeoutMs),
    max_output_tokens: cap(requested.max_output_tokens, limits.maxOutputTokens),
  };
  const rest = newline === -1 ? "" : code.slice(newline);
  return `${OPTIONS_PREFIX} ${JSON.stringify(options)}${rest}`;
}

/** An inline extension that applies {@link capCodemodeOptions} to every codemode call. */
export function createCodemodeLimitsExtension(limits: CodemodeLimits = DEFAULT_LIMITS): ExtensionFactory {
  return (pi) => {
    pi.on("tool_call", (event) => {
      if (event.toolName !== "codemode") return;
      const code = event.input.code;
      if (typeof code === "string") event.input.code = capCodemodeOptions(code, limits);
    });
  };
}
