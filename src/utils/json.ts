import { logger } from "./logger.js";

/** Narrow untyped JSON to a plain object. Arrays and null are rejected. */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Parse concatenated JSON arrays from `glab api --paginate` or
 * `gh api --paginate`. Both print `[...][...][...]`: one array per page, no
 * delimiter.
 * We track bracket depth (respecting strings/escapes) to find each
 * top-level array, parse them individually, and merge with flat().
 */
export function parsePaginatedJsonArrays(raw: string): Array<Record<string, unknown>> {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const chunks: string[] = [];
  let depth = 0;
  let inString = false;
  let escaped = false;
  let start = -1;

  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\" && inString) {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;

    if (ch === "[") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "]") {
      depth--;
      if (depth === 0 && start >= 0) {
        chunks.push(trimmed.slice(start, i + 1));
        start = -1;
      }
    }
  }

  const results: Array<Record<string, unknown>> = [];
  for (const chunk of chunks) {
    try {
      const parsed = JSON.parse(chunk) as Array<Record<string, unknown>>;
      if (Array.isArray(parsed)) results.push(...parsed);
    } catch (err) {
      logger.warn(
        `Skipping malformed pagination chunk: ${err instanceof Error ? err.message : err}`,
      );
    }
  }
  return results;
}
