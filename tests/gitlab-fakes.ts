/** Shared pieces of the in-memory GitLab API used by publisher tests. */

interface DiffRefs {
  base_sha: string;
  head_sha: string;
  start_sha: string;
}

export const VERSION_ID = 1;

/** The MR diff version for `refs`: `path` is a new file with `lineCount` added lines. */
export function addedFileVersion(refs: DiffRefs, path: string, lineCount: number) {
  const body = Array.from({ length: lineCount }, (_, i) => `+line ${i + 1}`).join("\n");
  return {
    id: VERSION_ID,
    base_commit_sha: refs.base_sha,
    head_commit_sha: refs.head_sha,
    start_commit_sha: refs.start_sha,
    state: "collected",
    diffs: [{
      old_path: path, new_path: path, new_file: true, renamed_file: false, deleted_file: false,
      collapsed: false, too_large: false, diff: `@@ -0,0 +1,${lineCount} @@\n${body}\n`,
    }],
  };
}

/** Answer the two diff-version reads, or undefined for any other endpoint. */
export function answerVersion(endpoint: string | undefined, version: ReturnType<typeof addedFileVersion>): unknown {
  if (endpoint?.endsWith("/versions?per_page=20")) {
    const { diffs: _diffs, ...summary } = version;
    return [summary];
  }
  if (endpoint?.endsWith(`/versions/${VERSION_ID}`)) return version;
  return undefined;
}

export function isReadback(endpoint: string | undefined): boolean {
  return endpoint?.includes("/notes?sort=desc&order_by=created_at&per_page=100") ?? false;
}

/** The diff note GitLab persists when it publishes a draft created with `input`. */
export function persistedNote(id: number, input: string, author: { id: number; username: string }) {
  const draft: unknown = JSON.parse(input);
  if (typeof draft !== "object" || draft === null || !("note" in draft) || !("position" in draft)) {
    throw new Error("draft payload has no note or position");
  }
  const position = draft.position;
  if (typeof position !== "object" || position === null) throw new Error("draft payload has no position");
  return {
    id, type: "DiffNote", body: draft.note, author, system: false, resolvable: true, resolved: false,
    position: { old_line: null, new_line: null, ...position },
  };
}

/** Readback output holding a persisted note for every draft POST in `calls`. */
export function readbackOfAllDrafts(calls: unknown[][], author: { id: number; username: string }): string {
  const notes = calls.flatMap((call, index) => {
    const args = call[1];
    const options = call[2];
    if (!Array.isArray(args) || !args.some((arg) => typeof arg === "string" && arg.endsWith("/draft_notes"))) return [];
    if (typeof options !== "object" || options === null || !("input" in options) || typeof options.input !== "string") return [];
    return [persistedNote(5000 + index, options.input, author)];
  });
  return JSON.stringify(notes.reverse());
}
