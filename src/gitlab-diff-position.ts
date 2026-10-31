import { isRecord } from "./utils/json.js";

/**
 * GitLab accepts a text diff note only when its old/new line pair names a line
 * of the MR diff: added lines carry only a new line, unchanged lines carry
 * both. A pair that matches no diff line can be dropped at publish time while
 * the API still reports success, so positions come from GitLab's own diff and
 * a line that does not map is rejected instead of guessed.
 */
export type DiffLineAnchor =
  | { kind: "added"; newLine: number }
  | { kind: "context"; oldLine: number; newLine: number };

export interface GitlabLinePosition {
  oldPath: string;
  newPath: string;
  anchor: DiffLineAnchor;
}

export type GitlabFileDiff =
  | { oldPath: string; newPath: string; anchors: ReadonlyMap<number, DiffLineAnchor> }
  | { oldPath: string; newPath: string; unavailable: string };

export type LinePositionResult =
  | { ok: true; position: GitlabLinePosition }
  | { ok: false; reason: string };

const HUNK_HEADER_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;

/**
 * Index the new-side lines of one file's diff text, as the GitLab API returns
 * it. Hunk lengths come from the headers, so content such as `--- x` is never
 * mistaken for a file header. Returns null for any malformed hunk.
 */
export function parseDiffAnchors(diff: string): Map<number, DiffLineAnchor> | null {
  const anchors = new Map<number, DiffLineAnchor>();
  const lines = diff.split("\n");
  if (lines.at(-1) === "") lines.pop();
  let i = 0;
  // `unidiff` output starts with file headers; skip everything before the first hunk.
  while (i < lines.length && !lines[i].startsWith("@@")) i++;
  while (i < lines.length) {
    const header = HUNK_HEADER_RE.exec(lines[i++]);
    if (!header) return null;
    let oldLine = Number(header[1]);
    let oldLeft = header[2] === undefined ? 1 : Number(header[2]);
    let newLine = Number(header[3]);
    let newLeft = header[4] === undefined ? 1 : Number(header[4]);
    while (oldLeft > 0 || newLeft > 0) {
      if (i >= lines.length) return null;
      const line = lines[i++];
      if (line.startsWith("\\")) continue;
      if (line.startsWith("+") && newLeft > 0) {
        if (anchors.has(newLine)) return null;
        anchors.set(newLine, { kind: "added", newLine });
        newLine++;
        newLeft--;
      } else if (line.startsWith("-") && oldLeft > 0) {
        oldLine++;
        oldLeft--;
      } else if (line.startsWith(" ") && oldLeft > 0 && newLeft > 0) {
        if (anchors.has(newLine)) return null;
        anchors.set(newLine, { kind: "context", oldLine, newLine });
        oldLine++;
        newLine++;
        oldLeft--;
        newLeft--;
      } else {
        return null;
      }
    }
    while (i < lines.length && lines[i].startsWith("\\")) i++;
  }
  return anchors;
}

/** Parse the `diffs` of a GitLab MR diff version. Throws when the shape is not a diff list. */
export function parseGitlabDiffVersion(version: unknown): GitlabFileDiff[] {
  const diffs = isRecord(version) ? version.diffs : undefined;
  if (!Array.isArray(diffs)) throw new Error("diff version has no diffs list");
  return diffs.flatMap((entry): GitlabFileDiff[] => {
    if (!isRecord(entry) || typeof entry.old_path !== "string" || typeof entry.new_path !== "string") return [];
    const paths = { oldPath: entry.old_path, newPath: entry.new_path };
    if (entry.collapsed === true || entry.too_large === true) {
      return [{ ...paths, unavailable: "GitLab withheld this file's diff" }];
    }
    if (typeof entry.diff !== "string") return [{ ...paths, unavailable: "the file diff has no text" }];
    const anchors = parseDiffAnchors(entry.diff);
    return [anchors ? { ...paths, anchors } : { ...paths, unavailable: "the file diff is malformed" }];
  });
}

/**
 * Anchor a finding's new-side line range. Both ends must be added or
 * unchanged lines of the same file diff; the note anchors on the start.
 */
export function findLinePosition(
  files: readonly GitlabFileDiff[],
  path: string,
  range: { start: number; end: number },
): LinePositionResult {
  const { start, end } = range;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 1 || end < start) {
    return { ok: false, reason: `invalid line range ${start}-${end}` };
  }
  const matches = files.filter((file) => file.newPath === path);
  if (matches.length === 0) return { ok: false, reason: `${path} is not in the MR diff` };
  if (matches.length > 1) return { ok: false, reason: `${path} appears more than once in the MR diff` };
  const file = matches[0];
  if ("unavailable" in file) return { ok: false, reason: `${path}: ${file.unavailable}` };
  const startAnchor = file.anchors.get(start);
  if (!startAnchor) return { ok: false, reason: `${path}:${start} is not an added or unchanged diff line` };
  if (!file.anchors.has(end)) return { ok: false, reason: `${path}:${end} is not an added or unchanged diff line` };
  return { ok: true, position: { oldPath: file.oldPath, newPath: file.newPath, anchor: startAnchor } };
}
