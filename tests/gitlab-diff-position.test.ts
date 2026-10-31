import { describe, expect, it } from "vitest";
import {
  findLinePosition,
  parseDiffAnchors,
  parseGitlabDiffVersion,
  type GitlabFileDiff,
} from "../src/gitlab-diff-position.js";

/** Synthetic service diff with deleted authorization checks. */
const SERVICE_DIFF = [
  "@@ -3,6 +3,4 @@",
  " ",
  " def read_statement(session_client_id: int, statement_id: int, records: dict) -> dict:",
  "     statement = records[statement_id]",
  "-    if statement[\"client_id\"] != session_client_id:",
  "-        raise PermissionError(\"statement belongs to another client\")",
  "     return statement",
  "",
].join("\n");

function entries(diff: string) {
  const anchors = parseDiffAnchors(diff);
  return anchors === null ? null : [...anchors.values()];
}

describe("parseDiffAnchors", () => {
  it("gives unchanged lines both coordinates and skips deleted lines", () => {
    expect(entries(SERVICE_DIFF)).toEqual([
      { kind: "context", oldLine: 3, newLine: 3 },
      { kind: "context", oldLine: 4, newLine: 4 },
      { kind: "context", oldLine: 5, newLine: 5 },
      { kind: "context", oldLine: 8, newLine: 6 },
    ]);
  });

  it("gives added lines only a new line and tracks each hunk's own offset", () => {
    const diff = [
      "@@ -1,3 +1,4 @@ header text",
      " a",
      "+b",
      " c",
      " d",
      "@@ -20,3 +21,2 @@",
      " x",
      "-y",
      " z",
      "",
    ].join("\n");
    expect(entries(diff)).toEqual([
      { kind: "context", oldLine: 1, newLine: 1 },
      { kind: "added", newLine: 2 },
      { kind: "context", oldLine: 2, newLine: 3 },
      { kind: "context", oldLine: 3, newLine: 4 },
      { kind: "context", oldLine: 20, newLine: 21 },
      { kind: "context", oldLine: 22, newLine: 22 },
    ]);
  });

  it("handles new files, omitted counts, deletion-only hunks, and no-newline markers", () => {
    expect(entries("@@ -0,0 +1,2 @@\n+one\n+two\n\\ No newline at end of file\n")).toEqual([
      { kind: "added", newLine: 1 },
      { kind: "added", newLine: 2 },
    ]);
    expect(entries("@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n")).toEqual([
      { kind: "added", newLine: 1 },
    ]);
    expect(entries("@@ -4,2 +3,0 @@\n-gone\n-gone too\n@@ -9,1 +7,1 @@\n same\n")).toEqual([
      { kind: "context", oldLine: 9, newLine: 7 },
    ]);
  });

  it("counts lines by the hunk header, so content that looks like a header stays content", () => {
    const diff = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n---- x\n+++ y\n same\n";
    expect(entries(diff)).toEqual([
      { kind: "added", newLine: 1 },
      { kind: "context", oldLine: 2, newLine: 2 },
    ]);
  });

  it.each([
    ["a hunk shorter than its header", "@@ -1,3 +1,3 @@\n a\n b\n"],
    ["a line after the hunk ends", "@@ -1,1 +1,1 @@\n a\n b\n"],
    ["an unprefixed line inside a hunk", "@@ -1,2 +1,2 @@\n a\n\n"],
    ["more added lines than the header allows", "@@ -1,1 +1,1 @@\n+a\n+b\n"],
    ["overlapping hunks", "@@ -1,1 +1,1 @@\n a\n@@ -1,1 +1,1 @@\n a\n"],
  ])("rejects %s", (_name, diff) => {
    expect(parseDiffAnchors(diff)).toBeNull();
  });

  it("returns no anchors for an empty diff", () => {
    expect(entries("")).toEqual([]);
  });
});

describe("parseGitlabDiffVersion", () => {
  const file = { old_path: "f.ts", new_path: "f.ts", collapsed: false, too_large: false };

  it("marks withheld, textless, and malformed file diffs unavailable", () => {
    const files = parseGitlabDiffVersion({
      diffs: [
        { ...file, diff: "@@ -0,0 +1 @@\n+a\n" },
        { ...file, new_path: "collapsed.ts", collapsed: true, diff: "" },
        { ...file, new_path: "large.ts", too_large: true, diff: "" },
        { ...file, new_path: "missing.ts" },
        { ...file, new_path: "bad.ts", diff: "@@ -1,2 +1,2 @@\n a\n" },
        { new_path: "no-old-path.ts", diff: "" },
        "not a diff",
      ],
    });
    expect(files.map((entry) => ("unavailable" in entry ? `${entry.newPath}: ${entry.unavailable}` : entry.newPath))).toEqual([
      "f.ts",
      "collapsed.ts: GitLab withheld this file's diff",
      "large.ts: GitLab withheld this file's diff",
      "missing.ts: the file diff has no text",
      "bad.ts: the file diff is malformed",
    ]);
  });

  it.each([null, {}, { diffs: "x" }])("throws for a version without a diffs list: %j", (version) => {
    expect(() => parseGitlabDiffVersion(version)).toThrow(/no diffs list/);
  });
});

describe("findLinePosition", () => {
  const service = parseGitlabDiffVersion({
    diffs: [
      { old_path: "service.py", new_path: "service.py", diff: SERVICE_DIFF },
      { old_path: "old.ts", new_path: "new.ts", diff: "@@ -1,1 +1,2 @@\n a\n+b\n" },
      { old_path: "huge.ts", new_path: "huge.ts", collapsed: true, diff: "" },
    ],
  });

  it("anchors on the start line when both ends map", () => {
    expect(findLinePosition(service, "service.py", { start: 5, end: 6 })).toEqual({
      ok: true,
      position: { oldPath: "service.py", newPath: "service.py", anchor: { kind: "context", oldLine: 5, newLine: 5 } },
    });
    expect(findLinePosition(service, "new.ts", { start: 2, end: 2 })).toEqual({
      ok: true,
      position: { oldPath: "old.ts", newPath: "new.ts", anchor: { kind: "added", newLine: 2 } },
    });
  });

  it.each([
    ["service.py", 1, 1, "service.py:1 is not an added or unchanged diff line"],
    ["service.py", 5, 7, "service.py:7 is not an added or unchanged diff line"],
    ["service.py", 6, 5, "invalid line range 6-5"],
    ["service.py", 0, 3, "invalid line range 0-3"],
    ["service.py", 3.5, 4, "invalid line range 3.5-4"],
    ["old.ts", 1, 1, "old.ts is not in the MR diff"],
    ["huge.ts", 1, 1, "huge.ts: GitLab withheld this file's diff"],
  ])("rejects %s:%d-%d", (path, start, end, reason) => {
    expect(findLinePosition(service, path, { start, end })).toEqual({ ok: false, reason });
  });

  it("rejects a path that appears twice instead of choosing one", () => {
    const twice: GitlabFileDiff[] = [...service, ...service];
    expect(findLinePosition(twice, "service.py", { start: 5, end: 5 })).toEqual({
      ok: false,
      reason: "service.py appears more than once in the MR diff",
    });
  });
});
