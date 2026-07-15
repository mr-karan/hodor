import { Value } from "@sinclair/typebox/value";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { SUBMIT_REVIEW_SCHEMA, validateReviewOutput } from "./review.js";
import type { ReviewOutput } from "./types.js";

export const SUBMIT_REVIEW_RECOVERY_ATTEMPTS = 2;

export function buildSubmitReviewRecoveryPrompt(attempt: number, maxAttempts: number): string {
  const finalAttempt =
    attempt >= maxAttempts
      ? "\nThis is the final automatic recovery attempt; do not end the turn without calling `submit_review`."
      : "";

  return [
    "Your previous assistant turn ended without a valid `submit_review` tool call, so Hodor cannot capture the review.",
    "Continue from the existing review context. Use only read-only tools and only the changed files/diff already identified.",
    "If more evidence is needed, inspect the relevant diff or file context now.",
    "When analysis is complete, call `submit_review` exactly once. Do not write the review as normal text.",
    "If there are no findings, call `submit_review` with `\"findings\": []` and `\"overall_correctness\": \"patch is correct\"`.",
    finalAttempt,
  ].filter(Boolean).join("\n");
}

export function parseReviewFromAssistantText(text: string): ReviewOutput | null {
  for (const candidate of getJsonCandidates(text)) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      if (!Value.Check(SUBMIT_REVIEW_SCHEMA, parsed)) continue;
      return validateReviewOutput(parsed as ReviewOutput);
    } catch {
      // Assistant text often contains prose around the payload; try the next candidate.
    }
  }
  return null;
}

export function summarizeLastAssistantMessage(session: AgentSession): string {
  const messages = session.messages as unknown as Array<Record<string, unknown>>;
  const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
  if (!lastAssistant) return "no assistant message";

  const stopReason =
    typeof lastAssistant.stopReason === "string" ? lastAssistant.stopReason : "unknown";
  const errorMessage =
    typeof lastAssistant.errorMessage === "string"
      ? `, error=${JSON.stringify(truncateForLog(lastAssistant.errorMessage, 300))}`
      : "";
  const content = Array.isArray(lastAssistant.content)
    ? lastAssistant.content
      .map((item) => {
        const block = item as Record<string, unknown>;
        const type = typeof block.type === "string" ? block.type : "unknown";
        return type === "toolCall" && typeof block.name === "string"
          ? `toolCall:${block.name}`
          : type;
      })
      .join(",")
    : "unknown";
  const rawText = session.getLastAssistantText()?.trim();
  const textSummary = rawText
    ? `, text=${JSON.stringify(truncateForLog(rawText.replace(/\s+/g, " "), 500))}`
    : "";

  return `stopReason=${stopReason}, content=[${content || "none"}]${errorMessage}${textSummary}`;
}

function getJsonCandidates(text: string): string[] {
  const candidates: string[] = [];
  const seen = new Set<string>();
  const add = (value: string): void => {
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) return;
    seen.add(trimmed);
    candidates.push(trimmed);
  };

  add(text);
  for (const match of text.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)) {
    add(match[1] ?? "");
  }
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    add(text.slice(firstBrace, lastBrace + 1));
  }
  return candidates;
}

function truncateForLog(text: string, maxLength: number): string {
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1)}…`;
}
