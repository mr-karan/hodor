import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { logger } from "./utils/logger.js";
import type { TrackedTree } from "./review-tools.js";
import type { ReviewOutput } from "./types.js";

/**
 * Snippet-based line resolution.
 *
 * The review agent emits line numbers (`code_location.line_range`) by counting
 * lines in a raw diff — the thing LLMs are worst at. Wrong numbers silently
 * drop GitLab inline comments and misplace `suggestion:` blocks (see
 * docs/SNIPPET_LINE_RESOLUTION.md).
 *
 * Instead, the agent also quotes the verbatim code it is commenting on
 * (`existing_code`). Here we locate that snippet in the on-disk file (the PR
 * branch is already checked out) and correct the line range to where the code
 * actually lives. When the snippet is absent or cannot be matched we keep the
 * model's range unchanged — the resolver is pure upside, never a regression.
 */

const MAX_RESOLVE_BYTES = 2 * 1024 * 1024; // skip resolution for files > 2MB

export interface ResolveStats {
  total: number;
  noSnippet: number; // finding had no existing_code
  confirmed: number; // snippet matched at the model's range (no change)
  corrected: number; // snippet matched at a different range (range fixed)
  unmatched: number; // had a snippet but no match — kept model's range
}

interface IndexedLine {
  lineNum: number; // 1-indexed line number in the original file
  content: string; // normalized content
}

/**
 * Normalize a line for matching: strip CR, trim outer whitespace, strip a
 * single leading diff marker (`+`/`-`) the model may have accidentally copied,
 * then trim again. Mirrors open-code-review's `normalizeLine`.
 */
function normalizeLine(line: string): string {
  let s = line.replace(/\r$/, "").trim();
  if (s.startsWith("+") || s.startsWith("-")) {
    s = s.slice(1).trim();
  }
  return s;
}

/** Split text into normalized, non-blank lines (blanks are dropped). */
function normalizeSnippet(code: string): string[] {
  return code
    .split("\n")
    .map(normalizeLine)
    .filter((l) => l.length > 0);
}

/**
 * Index a file's non-blank lines with their original 1-indexed line numbers.
 * Dropping blank lines on both sides lets a snippet match across internal
 * blank lines — a small improvement over open-code-review, which required the
 * snippet to be blank-line-free.
 */
function indexFile(content: string): IndexedLine[] {
  const result: IndexedLine[] = [];
  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const normalized = normalizeLine(lines[i]);
    if (normalized.length > 0) {
      result.push({ lineNum: i + 1, content: normalized });
    }
  }
  return result;
}

/** All [start,end] ranges (1-indexed, inclusive) where `target` matches a consecutive run. */
function findMatches(
  fileLines: IndexedLine[],
  target: string[],
): Array<{ start: number; end: number }> {
  const matches: Array<{ start: number; end: number }> = [];
  if (target.length === 0 || fileLines.length < target.length) return matches;

  for (let i = 0; i <= fileLines.length - target.length; i++) {
    let ok = true;
    for (let j = 0; j < target.length; j++) {
      if (fileLines[i + j].content !== target[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      matches.push({ start: fileLines[i].lineNum, end: fileLines[i + target.length - 1].lineNum });
    }
  }
  return matches;
}

export interface ResolveInput {
  existingCode?: string;
  fileContent: string;
  modelRange: { start: number; end: number };
  /** New-side line numbers touched by the diff, for disambiguation. */
  changedLines?: Set<number>;
}

export interface ResolveResult {
  start: number;
  end: number;
  /** "no-snippet" | "unmatched" => kept model range; "confirmed"/"corrected" => snippet matched. */
  status: "no-snippet" | "unmatched" | "confirmed" | "corrected";
}

/**
 * Pure core: resolve a single finding's line range from its snippet. No I/O.
 */
export function resolveLineRange(input: ResolveInput): ResolveResult {
  const { existingCode, fileContent, modelRange, changedLines } = input;
  const keep = (status: ResolveResult["status"]): ResolveResult => ({
    start: modelRange.start,
    end: modelRange.end,
    status,
  });

  const target = existingCode ? normalizeSnippet(existingCode) : [];
  if (target.length === 0) return keep("no-snippet");

  const matches = findMatches(indexFile(fileContent), target);
  if (matches.length === 0) return keep("unmatched");

  let chosen: { start: number; end: number };
  if (matches.length === 1) {
    chosen = matches[0];
  } else {
    // Prefer a match that overlaps the diff; otherwise the one nearest the
    // model's emitted start line.
    const overlapping =
      changedLines && changedLines.size > 0
        ? matches.filter((m) => {
            for (let l = m.start; l <= m.end; l++) {
              if (changedLines.has(l)) return true;
            }
            return false;
          })
        : [];
    const pool = overlapping.length > 0 ? overlapping : matches;
    chosen = pool.reduce((best, m) =>
      Math.abs(m.start - modelRange.start) < Math.abs(best.start - modelRange.start) ? m : best,
    );
  }

  const status = chosen.start === modelRange.start && chosen.end === modelRange.end ? "confirmed" : "corrected";
  return { start: chosen.start, end: chosen.end, status };
}

/**
 * Parse a unified diff into a map of file path -> set of new-side line numbers
 * within its hunks (added + context lines — both are valid comment anchors).
 * Used only to disambiguate multiple snippet matches.
 */
export function parseChangedLines(diffText: string): Map<string, Set<number>> {
  const byFile = new Map<string, Set<number>>();
  if (!diffText) return byFile;

  let current: Set<number> | null = null;
  let newLine = 0;
  const hunkRe = /^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@/;

  for (const line of diffText.split("\n")) {
    if (line.startsWith("+++ ")) {
      // "+++ b/path/to/file" (or "+++ /dev/null")
      let path = line.slice(4).trim();
      if (path.startsWith("b/")) path = path.slice(2);
      if (path === "/dev/null") {
        current = null;
      } else {
        current = byFile.get(path) ?? new Set<number>();
        byFile.set(path, current);
      }
      continue;
    }
    const m = line.match(hunkRe);
    if (m) {
      newLine = parseInt(m[1], 10);
      continue;
    }
    if (!current) continue;
    if (line.startsWith("diff --git ") || line.startsWith("--- ")) continue;
    if (line.startsWith("+")) {
      current.add(newLine);
      newLine++;
    } else if (line.startsWith("-")) {
      // deleted line — does not advance the new-side counter
    } else if (line.startsWith("\\")) {
      // "\ No newline at end of file"
    } else {
      // context line — part of the hunk's new side, usable as a comment anchor
      current.add(newLine);
      newLine++;
    }
  }
  return byFile;
}

/**
 * Resolve line ranges for every finding in a review against the checked-out
 * workspace. Returns a new review with corrected ranges plus resolution stats.
 * Findings are never dropped — on any miss or read error the model's range is kept.
 *
 * The model supplies absolute_file_path, so a malformed or injected payload
 * must not read arbitrary files: only tracked regular files inside the
 * repository (after symlink resolution) are read.
 */
export function resolveReviewLocations(
  review: ReviewOutput,
  opts: { trackedTree: TrackedTree; diffText?: string | null },
): { review: ReviewOutput; stats: ResolveStats } {
  const stats: ResolveStats = { total: 0, noSnippet: 0, confirmed: 0, corrected: 0, unmatched: 0 };
  if (review.findings.length === 0) return { review, stats };

  const { trackedTree } = opts;
  const changedByFile = opts.diffText ? parseChangedLines(opts.diffText) : new Map<string, Set<number>>();
  const fileCache = new Map<string, { content: string } | { content: null; reason: string }>();

  const readTrackedFile = (path: string): { content: string } | { content: null; reason: string } => {
    const cached = fileCache.get(path);
    if (cached) return cached;
    let result: { content: string } | { content: null; reason: string };
    try {
      const entry = trackedTree.resolve(path, "file");
      result = entry.size <= MAX_RESOLVE_BYTES
        ? { content: readFileSync(entry.absolutePath, "utf-8") }
        : { content: null, reason: `file is larger than ${MAX_RESOLVE_BYTES} bytes` };
    } catch (err) {
      result = { content: null, reason: err instanceof Error ? err.message : String(err) };
    }
    fileCache.set(path, result);
    return result;
  };

  const findings = review.findings.map((finding) => {
    stats.total++;
    const { existing_code: existingCode, code_location: loc } = finding;
    if (!existingCode) {
      stats.noSnippet++;
      return finding;
    }

    const file = readTrackedFile(loc.absolute_file_path);
    if (file.content === null) {
      stats.unmatched++;
      logger.warn(
        `Location resolution: cannot use ${loc.absolute_file_path} for "${finding.title}" (${file.reason}); keeping model range`,
      );
      return finding;
    }
    const fileContent = file.content;

    // Diff paths are repository-relative; absolute_file_path is workspace-absolute.
    const relPath = trackedTree.toRelative(resolve(trackedTree.root, loc.absolute_file_path));
    const changedLines = relPath === null ? undefined : changedByFile.get(relPath);

    const result = resolveLineRange({
      existingCode,
      fileContent,
      modelRange: loc.line_range,
      changedLines,
    });

    if (result.status === "confirmed") stats.confirmed++;
    else if (result.status === "corrected") stats.corrected++;
    else stats.unmatched++;

    if (result.status === "corrected") {
      logger.info(
        `Location resolution: corrected "${finding.title}" ${loc.line_range.start}-${loc.line_range.end} -> ${result.start}-${result.end}`,
      );
      return {
        ...finding,
        code_location: { ...loc, line_range: { start: result.start, end: result.end } },
      };
    }
    if (result.status === "unmatched") {
      logger.warn(
        `Location resolution: snippet not found for "${finding.title}" in ${loc.absolute_file_path}; keeping model range ${loc.line_range.start}-${loc.line_range.end}`,
      );
    }
    return finding;
  });

  return { review: { ...review, findings }, stats };
}
