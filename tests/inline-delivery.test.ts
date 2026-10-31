import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { postReviewStructured } from "../src/publisher.js";
import { partitionNotesByProvenance } from "../src/provenance.js";
import { buildReviewCacheMarker, findCachedReview, getReviewCacheKey } from "../src/review-cache.js";
import { getHodorReviewShaCandidates } from "../src/review-diff.js";
import { parseReviewKey, type ReviewKey } from "../src/review-key.js";
import type { NoteEntry, ReviewFinding, ReviewOutput } from "../src/types.js";

type LinePair = [number | null, number | null];
interface Position {
  base_sha: string;
  head_sha: string;
  start_sha: string;
  position_type: string;
  old_path: string;
  new_path: string;
  old_line?: number;
  new_line?: number;
}
interface State {
  nextId: number;
  pending: Record<string, string>;
  positions?: Record<string, Position | null>;
  published: Array<{ id: number; body: string; position: Position | null; resolved?: boolean }>;
  publishAttempts: Array<number | string>;
  calls: Array<{ endpoint: string; method: string }>;
  notes: NoteEntry[];
  response: "normal";
  failPublish: number[];
  failCreate: boolean;
  failSummary: boolean;
  failBulk: boolean;
  parkFirst: boolean;
  publishMode?: "persist" | "drop";
  publishAuthor?: { id: number; username: string };
  moveOnPublish?: number;
  failReadback?: boolean;
  readback?: { id?: unknown; position?: Record<string, unknown>; omit?: string[]; replace?: [string, string, string?] };
  failRefs?: boolean;
  failVersions?: boolean;
  detailPatch?: Record<string, unknown>;
  versions?: Array<Record<string, unknown>> | Record<string, unknown>;
  diffs?: Array<Record<string, unknown>>;
  validLines?: Record<string, LinePair[]>;
}

const fixture = fileURLToPath(new URL("./fixtures/gitlab-delivery/", import.meta.url));
const ROOT = "projects/acme%2Fapp/merge_requests/42";
const BASE = "a".repeat(40);
const HEAD = "b".repeat(40);
const MODEL = "anthropic/claude-opus-5-5";
const SERVICE = "fixtures/service.py";
const CI = ".gitlab-ci.yml";

/** Synthetic file diffs covering context, additions, renames, and collapsed files. */
const syntheticDiffs = [
  {
    old_path: SERVICE, new_path: SERVICE, new_file: false, renamed_file: false, deleted_file: false,
    collapsed: false, too_large: false,
    diff: [
      "@@ -3,6 +3,4 @@",
      " ",
      " def read_statement(session_client_id: int, statement_id: int, records: dict) -> dict:",
      "     statement = records[statement_id]",
      "-    if statement[\"client_id\"] != session_client_id:",
      "-        raise PermissionError(\"statement belongs to another client\")",
      "     return statement",
      "",
    ].join("\n"),
  },
  {
    old_path: CI, new_path: CI, new_file: false, renamed_file: false, deleted_file: false,
    collapsed: false, too_large: false,
    diff: [
      "@@ -10,6 +10,9 @@ workflow:",
      " ",
      " stages: [test]",
      " ",
      "+default:",
      "+  tags: [shared]",
      "+",
      " hodor-review:",
      "   extends: .hodor-review",
      "   timeout: 5m",
      "",
    ].join("\n"),
  },
  {
    old_path: "src/old_name.ts", new_path: "src/new_name.ts", new_file: false, renamed_file: true, deleted_file: false,
    collapsed: false, too_large: false,
    diff: "@@ -1,2 +1,2 @@\n const a = 1;\n-const b = 2;\n+const b = 3;\n",
  },
  {
    old_path: "src/huge.ts", new_path: "src/huge.ts", new_file: false, renamed_file: false, deleted_file: false,
    collapsed: true, too_large: false, diff: "",
  },
];

/** What GitLab's line_for_position accepts for these diffs, counted by hand from the hunks above. */
const syntheticValidLines: Record<string, LinePair[]> = {
  [SERVICE]: [[3, 3], [4, 4], [5, 5], [6, null], [7, null], [8, 6]],
  [CI]: [[10, 10], [11, 11], [12, 12], [null, 13], [null, 14], [null, 15], [13, 16], [14, 17], [15, 18]],
  "src/new_name.ts": [[1, 1], [2, null], [null, 2]],
  "src/huge.ts": [[null, 1]],
};

function finding(path: string, start: number, end: number, title: string): ReviewFinding {
  return {
    title,
    body: `Synthetic finding body for ${title}.`,
    priority: 1,
    code_location: { absolute_file_path: `/workspace/${path}`, line_range: { start, end } },
  };
}

const ownershipFinding = finding(SERVICE, 5, 6, "[P1] Restore the statement ownership check");

function reviewOf(findings: ReviewFinding[]): ReviewOutput {
  return { findings, overall_correctness: "patch is incorrect", overall_explanation: "Synthetic inline review." };
}

function cacheKeyFor() {
  return getReviewCacheKey({
    scope: { platform: "gitlab", host: "gitlab.example.com", projectPath: "acme/app", reviewNumber: 42, targetBranch: "main", baseSha: BASE },
    headSha: HEAD, model: MODEL, guidanceSnapshotSha: "c".repeat(40),
  });
}

const originalPath = process.env.PATH;
const originalState = process.env.HODOR_TEST_STATE;
let stateFile: string;

function readState(): State { return JSON.parse(readFileSync(stateFile, "utf8")); }
function change(changes: Partial<State>): void { writeFileSync(stateFile, JSON.stringify({ ...readState(), ...changes })); }
function trustedNotes(reviewKey: ReviewKey | null = null) {
  return partitionNotesByProvenance(readState().notes, { platform: "gitlab", userId: 7 }, reviewKey).hodor;
}
function summaryBody(): string { return readState().notes.at(-1)?.body ?? ""; }
function readbackCalls(): number {
  return readState().calls.filter(call => call.endpoint.startsWith(`${ROOT}/notes?sort=desc`)).length;
}

beforeEach(() => {
  const root = mkdtempSync(join(tmpdir(), "hodor-inline-delivery-"));
  stateFile = join(root, "state.json");
  process.env.HODOR_TEST_STATE = stateFile;
  process.env.PATH = `${fixture}${delimiter}${originalPath ?? ""}`;
  const state: State = {
    nextId: 501,
    pending: { "411": "A manually parked draft", "413": "<!-- hodor-review -->\nForeign default draft" },
    published: [], publishAttempts: [], calls: [], notes: [], response: "normal",
    failPublish: [], failCreate: false, failSummary: false, failBulk: false, parkFirst: false,
    diffs: syntheticDiffs, validLines: syntheticValidLines,
  };
  writeFileSync(stateFile, JSON.stringify(state));
});
afterEach(() => {
  if (originalPath === undefined) delete process.env.PATH;
  else process.env.PATH = originalPath;
  if (originalState === undefined) delete process.env.HODOR_TEST_STATE;
  else process.env.HODOR_TEST_STATE = originalState;
});

async function deliver(
  findings: ReviewFinding[],
  reviewStyle: "hybrid" | "inline" = "hybrid",
  reviewKey: ReviewKey | null = null,
) {
  const review = reviewOf(findings);
  const result = await postReviewStructured({
    prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
    review, reviewStyle, workspacePath: "/workspace", headSha: HEAD, reviewKey,
    cacheMarker: buildReviewCacheMarker(cacheKeyFor(), review, "/workspace"),
  });
  return { review, result };
}

function sentPositions(): Array<Position | null> {
  return Object.values(readState().positions ?? {});
}

const refs = { base_sha: BASE, head_sha: HEAD, start_sha: BASE, position_type: "text" };
const SECURITY = parseReviewKey("security");
const addedFinding = finding(CI, 13, 13, "[P1] Pin the default runner tag");

function expectNoCompletionMarkers(reviewKey: ReviewKey | null = null): void {
  expect(findCachedReview(trustedNotes(reviewKey), cacheKeyFor())).toBeNull();
  expect(getHodorReviewShaCandidates(trustedNotes(reviewKey))).toEqual([]);
  expect(summaryBody()).not.toContain("hodor:cache:");
  expect(summaryBody()).not.toContain("hodor:sha:");
}

describe("GitLab inline positions from the MR diff version", () => {
  it("anchors an unchanged context line with both old and new lines and caches the delivery", async () => {
    const { review, result } = await deliver([ownershipFinding]);
    expect(sentPositions()).toEqual([{ ...refs, old_path: SERVICE, new_path: SERVICE, old_line: 5, new_line: 5 }]);
    expect(result.success).toBe(true);
    expect(result.draftsPublished).toBe(true);
    expect(readState().published).toHaveLength(1);
    expect(readbackCalls()).toBe(1);
    expect(findCachedReview(trustedNotes(), cacheKeyFor())).toEqual(review);
    expect(getHodorReviewShaCandidates(trustedNotes())).toEqual([HEAD]);
    expect(summaryBody()).toContain("**Inline comments:** 1 new");
  });

  it("maps each line with its own hunk offset, side, and renamed path", async () => {
    const { result } = await deliver([
      finding(SERVICE, 6, 6, "[P1] Return after the check"),
      finding(CI, 13, 14, "[P1] Pin the default runner tag"),
      finding(CI, 16, 16, "[P1] Keep the job name stable"),
      finding("src/new_name.ts", 2, 2, "[P1] Keep the old value"),
    ]);
    expect(sentPositions()).toEqual([
      { ...refs, old_path: SERVICE, new_path: SERVICE, old_line: 8, new_line: 6 },
      { ...refs, old_path: CI, new_path: CI, new_line: 13 },
      { ...refs, old_path: CI, new_path: CI, old_line: 13, new_line: 16 },
      { ...refs, old_path: "src/old_name.ts", new_path: "src/new_name.ts", new_line: 2 },
    ]);
    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(4);
    expect(readState().published).toHaveLength(4);
    expect(readbackCalls()).toBe(1);
  });

  it.each([
    { name: "a line outside every hunk", target: finding(SERVICE, 1, 1, "[P1] Outside the hunk") },
    { name: "a range whose end leaves the hunk", target: finding(SERVICE, 5, 9, "[P1] Past the hunk") },
    { name: "a file missing from the diff", target: finding("src/other.ts", 1, 1, "[P1] Other file") },
    { name: "a collapsed file diff", target: finding("src/huge.ts", 1, 1, "[P1] Collapsed file") },
    // The cache payload itself rejects an inverted range, so only the summary is checked.
    { name: "an inverted range", target: finding(SERVICE, 6, 5, "[P1] Inverted range"), cacheable: false },
  ])("sends no draft for $name and delivers it only through the summary", async ({ target, cacheable = true }) => {
    const { review, result } = await deliver([target], "inline");
    expect(sentPositions()).toEqual([]);
    expect(readState().publishAttempts).toEqual([]);
    expect(result.inlineCreated).toBe(0);
    expect(result.inlineFailed).toBe(1);
    expect(result.summaryPosted).toBe(true);
    expect(result.success).toBe(true);
    expect(summaryBody()).toContain(target.body);
    expect(result.errors?.join("\n")).toMatch(/no GitLab diff position/);
    expect(findCachedReview(trustedNotes(), cacheKeyFor())).toEqual(cacheable ? review : null);
  });

  it.each(["hybrid", "inline"] as const)("withholds completion on initial refs failure in %s mode", async (style) => {
    change({ failRefs: true });
    const { result } = await deliver([ownershipFinding], style);
    expect(result.success).toBe(false);
    expect(result.summaryPosted).toBe(true);
    expect(result.errors?.join("\n")).toContain("diff refs failed");
    expect(summaryBody()).toContain(ownershipFinding.body);
    expect(readState().publishAttempts).toEqual([]);
    expectNoCompletionMarkers();
  });

  it("preserves both initial refs and fallback summary errors", async () => {
    change({ failRefs: true, failSummary: true });
    const { result } = await deliver([ownershipFinding]);
    expect(result.success).toBe(false);
    expect(result.errors?.join("\n")).toContain("diff refs failed");
    expect(result.errors?.join("\n")).toContain("summary failed");
    expectNoCompletionMarkers();
  });

  it("delivers at the same HEAD after initial refs recovery", async () => {
    change({ failRefs: true });
    expect((await deliver([ownershipFinding])).result.success).toBe(false);
    expectNoCompletionMarkers();
    change({ failRefs: false });
    const { review, result } = await deliver([ownershipFinding]);
    expect(result.success).toBe(true);
    expect(readState().published).toHaveLength(1);
    expect(findCachedReview(trustedNotes(), cacheKeyFor())).toEqual(review);
  });

  it.each([
    { name: "the versions request fails", state: { failVersions: true }, error: /500 versions failed/ },
    { name: "the versions response is not a list", state: { versions: { id: 1 } }, error: /no diff version matches/ },
    {
      name: "no diff version matches the MR diff refs",
      state: { versions: [{ id: 8, head_commit_sha: "d".repeat(40), base_commit_sha: BASE, start_commit_sha: BASE, state: "collected" }] },
      error: /no diff version matches/,
    },
    { name: "the version detail has other refs", state: { detailPatch: { head_commit_sha: "d".repeat(40) } }, error: /does not match the MR diff refs/ },
    { name: "the version detail has no diffs list", state: { detailPatch: { diffs: null } }, error: /no diffs list/ },
  ])("posts the full summary but claims no delivery when $name", async ({ state, error }) => {
    change(state);
    const { result } = await deliver([ownershipFinding, addedFinding]);
    expect(sentPositions()).toEqual([]);
    expect(readState().publishAttempts).toEqual([]);
    expect(result.inlineFailed).toBe(2);
    expect(result.success).toBe(false);
    expect(result.summaryPosted).toBe(true);
    expect(summaryBody()).toContain(ownershipFinding.body);
    expect(summaryBody()).toContain(addedFinding.body);
    expect(result.errors?.join("\n")).toMatch(/cannot load the MR diff/);
    expect(result.errors?.join("\n")).toMatch(error);
    expectNoCompletionMarkers();
  });

  it("retries inline delivery at the same HEAD after the diff read recovers", async () => {
    change({ failVersions: true });
    expect((await deliver([ownershipFinding])).result.success).toBe(false);
    change({ failVersions: false });
    const { review, result } = await deliver([ownershipFinding]);
    expect(result.success).toBe(true);
    expect(readState().published).toHaveLength(1);
    expect(findCachedReview(trustedNotes(), cacheKeyFor())).toEqual(review);
  });
});

describe("persisted inline note verification after per-ID publication", () => {
  it.each(["hybrid", "inline"] as const)("does not claim %s delivery when the publish succeeds but no note persists", async (reviewStyle) => {
    change({ publishMode: "drop" });
    const { result } = await deliver([ownershipFinding], reviewStyle);
    expect(readState().publishAttempts).toEqual([501]);
    expect(readState().published).toEqual([]);
    expect(result.success).toBe(false);
    expect(result.draftsPublished).toBe(false);
    expect(result.errors?.join("\n")).toMatch(/unverified after publishing draft 501 \(no note with its delivery marker/);
    expect(result.summaryPosted).toBe(true);
    expect(summaryBody()).toContain(ownershipFinding.body);
    expect(summaryBody()).not.toContain("**Inline comments:**");
    expectNoCompletionMarkers();
    expect(Object.keys(readState().pending)).toEqual(["411", "413"]);
    expect(readbackCalls()).toBe(1);
  });

  it("does not accept a matching note written by another account", async () => {
    change({ publishAuthor: { id: 8, username: "someone-else" } });
    const { result } = await deliver([ownershipFinding]);
    expect(readState().published).toHaveLength(1);
    expect(result.success).toBe(false);
    expect(result.draftsPublished).toBe(false);
    expectNoCompletionMarkers();
  });

  it("does not accept this invocation's note persisted at a different line", async () => {
    change({ moveOnPublish: 1 });
    const { result } = await deliver([ownershipFinding]);
    expect(readState().published).toHaveLength(1);
    expect(result.success).toBe(false);
    expect(result.draftsPublished).toBe(false);
    expectNoCompletionMarkers();
  });

  it("fails closed when the persisted-note readback fails", async () => {
    change({ failReadback: true });
    const { result } = await deliver([ownershipFinding]);
    expect(readState().published).toHaveLength(1);
    expect(result.success).toBe(false);
    expect(result.errors?.join("\n")).toMatch(/500 notes failed/);
    expectNoCompletionMarkers();
  });

  it("does not let an earlier resolved identical note stand in for this invocation's note", async () => {
    expect((await deliver([ownershipFinding])).result.success).toBe(true);
    const [earlier] = readState().published;
    change({ published: [{ ...earlier, resolved: true }], publishMode: "drop" });
    const { result } = await deliver([ownershipFinding]);
    expect(readState().publishAttempts).toEqual([501, 502]);
    expect(result.inlineCreated).toBe(1);
    expect(result.success).toBe(false);
    expect(result.draftsPublished).toBe(false);
    expect(summaryBody()).not.toContain("hodor:sha:");
  });
});

describe("readback that must name this invocation's finding, key, and diff version", () => {
  const DELIVERY = "<!-- hodor:delivery:[0-9a-f-]+ -->";
  const OTHER = "d".repeat(40);

  it.each([
    { name: "carries another review key", readback: { replace: [`(${DELIVERY})`, "$1\n<!-- hodor:key:other -->"] } },
    { name: "carries the wrong review key", reviewKey: SECURITY, readback: { replace: ["hodor:key:security", "hodor:key:other"] } },
    { name: "lost its review key", reviewKey: SECURITY, readback: { replace: ["<!-- hodor:key:security -->\n", ""] } },
    { name: "names another finding", readback: { replace: ["hodor:finding:[0-9a-f]{64}", `hodor:finding:${"0".repeat(64)}`] } },
    { name: "lost its finding marker", readback: { replace: ["<!-- hodor:finding:[0-9a-f]{64} -->\n", ""] } },
    { name: "only quotes the delivery marker in prose", readback: { replace: [`^[\\s\\S]*(${DELIVERY})[\\s\\S]*$`, "Prose quoting $1 only."] } },
    { name: "has the delivery marker below the review text", readback: { replace: [`^([\\s\\S]*?)(${DELIVERY})\n([\\s\\S]*)$`, "$1$3\n$2"] } },
    { name: "is on another head_sha", readback: { position: { head_sha: OTHER } } },
    { name: "is on another base_sha", readback: { position: { base_sha: OTHER } } },
    { name: "is on another start_sha", readback: { position: { start_sha: OTHER } } },
    { name: "has no diff refs", readback: { omit: ["base_sha", "start_sha", "head_sha"] } },
    { name: "is an image position", readback: { position: { position_type: "image" } } },
    { name: "has a negative fractional id", readback: { id: -3.5 } },
    { name: "has id 0", readback: { id: 0 } },
    { name: "has a string id", readback: { id: "9501" } },
    { name: "has a string new_line", readback: { position: { new_line: "5" } } },
    { name: "has new_line 0", readback: { position: { new_line: 0 } } },
    { name: "has a fractional old_line", readback: { position: { old_line: 5.5 } } },
    { name: "has a nonsense old_line on an added line", target: addedFinding, readback: { position: { old_line: "nonsense" } } },
    { name: "has a numeric path", readback: { position: { old_path: 5 } } },
  ] satisfies Array<{ name: string; readback: State["readback"]; reviewKey?: ReviewKey; target?: ReviewFinding }>)(
    "does not claim delivery when the only marked note $name",
    async ({ readback, reviewKey = null, target = ownershipFinding }) => {
      change({ readback });
      const { result } = await deliver([target], "hybrid", reviewKey);
      expect(readState().publishAttempts).toEqual([501]);
      expect(readState().published).toHaveLength(1);
      expect(result.success).toBe(false);
      expect(result.draftsPublished).toBe(false);
      expect(result.errors?.join("\n")).toMatch(/unverified after publishing draft 501/);
      expect(result.summaryPosted).toBe(true);
      expect(summaryBody()).toContain(target.body);
      expectNoCompletionMarkers(reviewKey);
      expect(Object.keys(readState().pending)).toEqual(["411", "413"]);
      expect(readbackCalls()).toBe(1);
    },
  );

  it.each([
    { name: "a keyed note under its own key", reviewKey: SECURITY },
    { name: "a note GitLab stored with CRLF line endings", readback: { replace: ["\n", "\r\n", "g"] } },
    { name: "a note whose review text GitLab rewrote", readback: { replace: ["Synthetic finding body", "Normalized finding body"] } },
    { name: "an added line with old_line absent", target: addedFinding, readback: { omit: ["old_line"] } },
    { name: "an added line with old_line null", target: addedFinding },
  ] satisfies Array<{ name: string; readback?: State["readback"]; reviewKey?: ReviewKey; target?: ReviewFinding }>)(
    "verifies and caches $name",
    async ({ readback, reviewKey = null, target = ownershipFinding }) => {
      if (readback) change({ readback });
      const { review, result } = await deliver([target], "hybrid", reviewKey);
      expect(result.success).toBe(true);
      expect(result.draftsPublished).toBe(true);
      expect(findCachedReview(trustedNotes(reviewKey), cacheKeyFor())).toEqual(review);
      expect(getHodorReviewShaCandidates(trustedNotes(reviewKey))).toEqual([HEAD]);
    },
  );
});
