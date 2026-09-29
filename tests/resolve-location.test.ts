import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  resolveLineRange,
  parseChangedLines,
  resolveReviewLocations,
} from "../src/resolve-location.js";
import { TrackedTree } from "../src/review-tools.js";
import type { ReviewOutput } from "../src/types.js";

const FILE = [
  "function add(a, b) {", // 1
  "  return a + b;", //       2
  "}", //                     3
  "", //                      4
  "function sub(a, b) {", //  5
  "  return a - b;", //       6
  "}", //                     7
  "", //                      8
  "function add2(a, b) {", // 9
  "  return a + b;", //       10
  "}", //                     11
].join("\n");

describe("resolveLineRange", () => {
  it("returns model range unchanged when no snippet is provided", () => {
    const r = resolveLineRange({ fileContent: FILE, modelRange: { start: 5, end: 6 } });
    expect(r).toEqual({ start: 5, end: 6, status: "no-snippet" });
  });

  it("confirms a model range that already matches the snippet", () => {
    const r = resolveLineRange({
      existingCode: "function sub(a, b) {\n  return a - b;",
      fileContent: FILE,
      modelRange: { start: 5, end: 6 },
    });
    expect(r).toEqual({ start: 5, end: 6, status: "confirmed" });
  });

  it("corrects an off-by-N model range using the snippet (the headline case)", () => {
    const r = resolveLineRange({
      existingCode: "function sub(a, b) {\n  return a - b;\n}",
      fileContent: FILE,
      modelRange: { start: 2, end: 4 }, // model guessed wrong
    });
    expect(r).toEqual({ start: 5, end: 7, status: "corrected" });
  });

  it("matches despite leading-indentation and trailing-whitespace differences", () => {
    const r = resolveLineRange({
      existingCode: "      function sub(a, b) {   \n\treturn a - b;  ",
      fileContent: FILE,
      modelRange: { start: 1, end: 2 },
    });
    expect(r.status).toBe("corrected");
    expect(r).toMatchObject({ start: 5, end: 6 });
  });

  it("strips stray diff +/- markers from the snippet before matching", () => {
    const r = resolveLineRange({
      existingCode: "+function sub(a, b) {\n+  return a - b;",
      fileContent: FILE,
      modelRange: { start: 1, end: 1 },
    });
    expect(r).toMatchObject({ start: 5, end: 6, status: "corrected" });
  });

  it("matches across internal blank lines", () => {
    const r = resolveLineRange({
      existingCode: "}\n\nfunction sub(a, b) {",
      fileContent: FILE,
      modelRange: { start: 1, end: 1 },
    });
    // "}" at line 3, "function sub" at line 5 — blank line 4 spanned
    expect(r).toMatchObject({ start: 3, end: 5, status: "corrected" });
  });

  it("keeps model range when the snippet cannot be found", () => {
    const r = resolveLineRange({
      existingCode: "function multiply(a, b) {",
      fileContent: FILE,
      modelRange: { start: 5, end: 5 },
    });
    expect(r).toEqual({ start: 5, end: 5, status: "unmatched" });
  });

  it("disambiguates duplicate matches by diff overlap", () => {
    // "  return a + b;" appears at line 2 and line 10
    const r = resolveLineRange({
      existingCode: "  return a + b;",
      fileContent: FILE,
      modelRange: { start: 1, end: 1 },
      changedLines: new Set([9, 10, 11]),
    });
    expect(r).toMatchObject({ start: 10, end: 10, status: "corrected" });
  });

  it("disambiguates duplicate matches by proximity when no diff info", () => {
    const r = resolveLineRange({
      existingCode: "  return a + b;",
      fileContent: FILE,
      modelRange: { start: 9, end: 9 }, // closer to line 10 than line 2
    });
    expect(r).toMatchObject({ start: 10, end: 10 });
  });
});

describe("parseChangedLines", () => {
  it("extracts all new-side hunk lines (added + context) per file", () => {
    const diff = [
      "diff --git a/src/foo.ts b/src/foo.ts",
      "--- a/src/foo.ts",
      "+++ b/src/foo.ts",
      "@@ -1,3 +1,4 @@",
      " const a = 1;",
      "+const b = 2;",
      " const c = 3;",
      "-const d = 4;",
      "+const d = 40;",
      " const e = 5;",
    ].join("\n");
    const map = parseChangedLines(diff);
    expect(map.has("src/foo.ts")).toBe(true);
    // new lines: 1 ctx, 2 added, 3 ctx, 4 added, 5 ctx => {1,2,3,4,5} (deleted line never advances new side)
    expect([...map.get("src/foo.ts")!].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
  });

  it("ignores deleted files (+++ /dev/null)", () => {
    const diff = ["diff --git a/x b/x", "--- a/x", "+++ /dev/null", "@@ -1 +0,0 @@", "-gone"].join("\n");
    expect(parseChangedLines(diff).size).toBe(0);
  });

  it("returns an empty map for empty input", () => {
    expect(parseChangedLines("").size).toBe(0);
  });
});

describe("resolveReviewLocations", () => {
  let base: string;
  let dir: string;
  let filePath: string;
  let trackedTree: TrackedTree;

  beforeEach(() => {
    base = mkdtempSync(join(tmpdir(), "hodor-resolve-test-"));
    dir = join(base, "repo");
    mkdirSync(dir);
    mkdirSync(join(base, "outside"));
    filePath = join(dir, "math.ts");
    writeFileSync(filePath, FILE, "utf-8");
    writeFileSync(join(dir, "untracked.ts"), FILE, "utf-8");
    writeFileSync(join(base, "outside", "secret.ts"), FILE, "utf-8");
    symlinkSync(join(base, "outside", "secret.ts"), join(dir, "escape.ts"));
    writeFileSync(join(dir, "huge.ts"), FILE + "x".repeat(3 * 1024 * 1024), "utf-8");
    trackedTree = new TrackedTree(dir, ["math.ts", "escape.ts", "huge.ts"]);
  });

  afterEach(() => {
    rmSync(base, { recursive: true, force: true });
  });

  const makeReview = (overrides: Partial<ReviewOutput["findings"][number]>): ReviewOutput => ({
    findings: [
      {
        title: "[P1] something",
        body: "body",
        priority: 1,
        code_location: { absolute_file_path: filePath, line_range: { start: 1, end: 1 } },
        ...overrides,
      },
    ],
    overall_correctness: "patch is incorrect",
    overall_explanation: "x",
  });

  it("corrects a finding's range from its snippet and reports stats", () => {
    const review = makeReview({
      existing_code: "function sub(a, b) {\n  return a - b;",
      code_location: { absolute_file_path: filePath, line_range: { start: 2, end: 2 } },
    });
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree, diffText: null });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 5, end: 6 });
    expect(stats).toMatchObject({ total: 1, corrected: 1, confirmed: 0, unmatched: 0, noSnippet: 0 });
  });

  it("leaves findings without a snippet untouched", () => {
    const review = makeReview({ code_location: { absolute_file_path: filePath, line_range: { start: 5, end: 6 } } });
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree, diffText: null });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 5, end: 6 });
    expect(stats.noSnippet).toBe(1);
  });

  it("keeps the model range and counts unmatched when the file cannot be read", () => {
    const review = makeReview({
      existing_code: "whatever",
      code_location: { absolute_file_path: join(dir, "missing.ts"), line_range: { start: 3, end: 4 } },
    });
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree, diffText: null });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 3, end: 4 });
    expect(stats.unmatched).toBe(1);
  });

  it("refuses to resolve files outside the workspace", () => {
    const review = makeReview({
      existing_code: "root:x:0:0",
      code_location: { absolute_file_path: "/etc/passwd", line_range: { start: 1, end: 1 } },
    });
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree, diffText: null });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 1, end: 1 });
    expect(stats.unmatched).toBe(1);
  });

  it("blocks path traversal that escapes the workspace via ..", () => {
    const review = makeReview({
      existing_code: "anything",
      code_location: {
        absolute_file_path: join(dir, "..", "..", "etc", "passwd"),
        line_range: { start: 2, end: 2 },
      },
    });
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree, diffText: null });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 2, end: 2 });
    expect(stats.unmatched).toBe(1);
  });

  it.each([
    ["an untracked file in the workspace", "untracked.ts"],
    ["a tracked symlink whose target is outside the workspace", "escape.ts"],
    ["a tracked file above the size limit", "huge.ts"],
  ])("refuses %s", (_label, name) => {
    const review = makeReview({
      existing_code: "function sub(a, b) {\n  return a - b;",
      code_location: { absolute_file_path: join(dir, name), line_range: { start: 2, end: 2 } },
    });
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree, diffText: null });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 2, end: 2 });
    expect(stats.unmatched).toBe(1);
  });

  it("uses the diff's changed lines for the matching repository path", () => {
    const review = makeReview({
      existing_code: "}",
      code_location: { absolute_file_path: filePath, line_range: { start: 1, end: 1 } },
    });
    const diffText = ["diff --git a/math.ts b/math.ts", "--- a/math.ts", "+++ b/math.ts", "@@ -7 +7 @@", "+}"].join("\n");
    const { review: out } = resolveReviewLocations(review, { trackedTree, diffText });
    expect(out.findings[0].code_location.line_range).toEqual({ start: 7, end: 7 });
  });

  it("is a no-op for an empty findings list", () => {
    const review: ReviewOutput = { findings: [], overall_correctness: "patch is correct", overall_explanation: "ok" };
    const { review: out, stats } = resolveReviewLocations(review, { trackedTree });
    expect(out).toBe(review);
    expect(stats.total).toBe(0);
  });
});
