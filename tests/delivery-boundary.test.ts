import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
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

interface State {
  nextId: number;
  pending: Record<string, string>;
  published: Array<{ id: number; body: string }>;
  publishAttempts: Array<number | string>;
  calls: Array<{ endpoint: string; method: string }>;
  notes: NoteEntry[];
  response: "normal" | "empty" | "invalid-json" | "invalid-first" | "null" | "string";
  failPublish: number[];
  failCreate: boolean;
  failSummary: boolean;
  failBulk: boolean;
  parkFirst: boolean;
}
const fixture = fileURLToPath(new URL("./fixtures/gitlab-delivery/", import.meta.url));
const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
const HEAD = "b".repeat(40);
const MODEL = "anthropic/claude-opus-5-5";
const finding: ReviewFinding = {
  title: "[P2] Bound input before processing",
  body: "The synthetic input must have an explicit size limit before processing.",
  priority: 2,
  code_location: { absolute_file_path: "/workspace/src/app.ts", line_range: { start: 1, end: 1 } },
};
const review: ReviewOutput = {
  findings: [finding], overall_correctness: "patch is incorrect", overall_explanation: "Synthetic delivery review.",
};
const opts = { prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42", review, workspacePath: "/workspace", headSha: HEAD };
const cacheKey = getReviewCacheKey({
  scope: { platform: "gitlab", host: "gitlab.example.com", projectPath: "acme/app", reviewNumber: 42, targetBranch: "main", baseSha: "a".repeat(40) },
  headSha: HEAD, model: MODEL, guidanceSnapshotSha: "c".repeat(40),
});
const cacheMarker = buildReviewCacheMarker(cacheKey, review, "/workspace");
const originalPath = process.env.PATH;
const originalState = process.env.HODOR_TEST_STATE;
let root: string;
let stateFile: string;
function readState(): State { return JSON.parse(readFileSync(stateFile, "utf8")); }
function change(changes: Partial<State>): void { writeFileSync(stateFile, JSON.stringify({ ...readState(), ...changes })); }
function trustedNotes(reviewKey: ReviewKey | null = null) {
  return partitionNotesByProvenance(readState().notes, { platform: "gitlab", userId: 7 }, reviewKey).hodor;
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hodor-delivery-boundary-"));
  stateFile = join(root, "state.json");
  process.env.HODOR_TEST_STATE = stateFile;
  process.env.PATH = `${fixture}${delimiter}${originalPath ?? ""}`;
  const state: State = {
    nextId: 501,
    pending: { "411": "A manually parked draft", "412": "<!-- hodor-review -->\n<!-- hodor:key:security -->\nForeign security draft", "413": "<!-- hodor-review -->\nForeign default draft" },
    published: [], publishAttempts: [], calls: [], notes: [], response: "normal",
    failPublish: [], failCreate: false, failSummary: false, failBulk: false, parkFirst: false,
  };
  writeFileSync(stateFile, JSON.stringify(state));
});
afterEach(() => {
  if (originalPath === undefined) delete process.env.PATH;
  else process.env.PATH = originalPath;
  if (originalState === undefined) delete process.env.HODOR_TEST_STATE;
  else process.env.HODOR_TEST_STATE = originalState;
});

function runCli(reviewKey: string | null = null) {
  // No caller credentials, settings, .env, or agent home enter the subprocess.
  const workspace = join(root, "workspace");
  mkdirSync(join(workspace, "src"), { recursive: true });
  writeFileSync(join(workspace, "src/app.ts"), "const value = 2;\n");
  const result = spawnSync("bun", [
    "--preload", join(fixture, "block-model.mjs"), cli, opts.prUrl,
    "--model", MODEL, "--post", "--require-delivery", "--verbose",
    ...(reviewKey ? ["--review-key", reviewKey] : []),
  ], {
    cwd: root, encoding: "utf8", timeout: 15_000,
    env: {
      PATH: process.env.PATH, HOME: join(root, "home"), XDG_CONFIG_HOME: join(root, "config"),
      PI_CODING_AGENT_DIR: join(root, "agent"), PI_OFFLINE: "1",
      LLM_API_KEY: "synthetic-unused-key", GITLAB_TOKEN: "synthetic-unused-token",
      GITLAB_CI: "true", CI_PROJECT_DIR: workspace, CI_PROJECT_PATH: "acme/app",
      CI_MERGE_REQUEST_TARGET_BRANCH_NAME: "main", CI_MERGE_REQUEST_DIFF_BASE_SHA: "a".repeat(40),
      HODOR_TEST_STATE: stateFile, HODOR_TEST_MODEL_CALLS: join(root, "model-calls"),
      HODOR_TEST_GIT_CALLS: join(root, "git-calls"),
    },
  });
  if (result.error) throw result.error;
  return { status: result.status, output: result.stdout + result.stderr };
}

function expectNoCompletionMarkers(reviewKey: ReviewKey | null = null): void {
  expect(findCachedReview(trustedNotes(reviewKey), cacheKey)).toBeNull();
  expect(getHodorReviewShaCandidates(trustedNotes(reviewKey))).toEqual([]);
  expect(readState().notes.at(-1)?.body ?? "").not.toContain("hodor:cache:");
  expect(readState().notes.at(-1)?.body ?? "").not.toContain("hodor:sha:");
}

describe("draft delivery across the real subprocess and CLI cache boundaries", () => {
  it.each([
    { parkedKey: "security", activeKey: null },
    { parkedKey: null, activeKey: "security" },
    { parkedKey: null, activeKey: null },
    { parkedKey: "security", activeKey: "security" },
  ])("isolates concurrent real subprocesses with keys $parkedKey and $activeKey", async ({ parkedKey, activeKey }) => {
    change({ parkFirst: true });
    const parked = postReviewStructured({ ...opts, reviewKey: parkedKey === null ? null : parseReviewKey(parkedKey) });
    try {
      const deadline = Date.now() + 5000;
      while (!existsSync(`${stateFile}.ready`)) {
        if (Date.now() > deadline) throw new Error("synthetic draft creation gate did not become ready");
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      const active = await postReviewStructured({ ...opts, reviewKey: activeKey === null ? null : parseReviewKey(activeKey) });
      expect(active.success).toBe(true);
      expect(readState().published.map(note => note.id)).toEqual([502]);
      expect(Object.keys(readState().pending)).toEqual(["411", "412", "413", "501"]);
    } finally {
      writeFileSync(`${stateFile}.release`, "release");
      await parked;
    }
    expect((await parked).success).toBe(true);
    expect(readState().publishAttempts).toEqual([502, 501]);
    expect(Object.keys(readState().pending)).toEqual(["411", "412", "413"]);
  }, 15_000);

  it.each([
    { response: "empty", reviewStyle: "hybrid" },
    { response: "invalid-json", reviewStyle: "hybrid" },
    { response: "null", reviewStyle: "hybrid" },
    { response: "empty", reviewStyle: "inline" },
    { response: "invalid-json", reviewStyle: "inline" },
    { response: "null", reviewStyle: "inline" },
  ] as const)("fails closed after a successful creation with $response stdout in $reviewStyle mode", async ({ response, reviewStyle }) => {
    change({ response });
    const result = await postReviewStructured({ ...opts, reviewStyle, cacheMarker });
    console.log("AMBIGUOUS_CREATE", JSON.stringify({ response, ...result }));
    expect(result.success).toBe(false);
    expect(result.inlineCreated).toBe(1);
    expect(result.inlineFailed).toBe(0);
    expect(result.draftsPublished).toBe(false);
    expect(result.summaryPosted).toBe(reviewStyle === "hybrid");
    expect(result.errors?.join("\n")).toMatch(/draft.*id/i);
    expect(readState().publishAttempts).toEqual([]);
    expect(Object.keys(readState().pending)).toEqual(["411", "412", "413", "501"]);
    expectNoCompletionMarkers();
  });

  it("does not let another published ID hide a truncated creation response", async () => {
    change({ response: "invalid-first" });
    const second: ReviewFinding = {
      ...finding, title: "[P2] Validate the response",
      code_location: { ...finding.code_location, line_range: { start: 2, end: 2 } },
    };
    const result = await postReviewStructured({ ...opts, review: { ...review, findings: [finding, second] }, cacheMarker });
    expect(result.success).toBe(false);
    expect(result.inlineCreated).toBe(2);
    expect(result.draftsPublished).toBe(false);
    expect(readState().publishAttempts).toEqual([502]);
    expect(Object.keys(readState().pending)).toEqual(["411", "412", "413", "501"]);
    expectNoCompletionMarkers();
  });

  it("only publishes a fresh owned ID after a truncated creation response", async () => {
    change({ response: "invalid-json" });
    const first = await postReviewStructured({ ...opts, cacheMarker });
    expect(first.success).toBe(false);
    change({ response: "normal" });
    const fresh = await postReviewStructured({ ...opts, cacheMarker });
    expect(fresh.success).toBe(true);
    expect(readState().publishAttempts).toEqual([502]);
    expect(Object.keys(readState().pending)).toEqual(["411", "412", "413", "501"]);
    expect(findCachedReview(trustedNotes(), cacheKey)).toEqual(review);
  });

  it.each(["hybrid", "inline"] as const)("keeps a rejected POST's delivered %s summary fallback cacheable", async reviewStyle => {
    change({ failCreate: true });
    const result = await postReviewStructured({ ...opts, reviewStyle, cacheMarker });
    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(0);
    expect(result.inlineFailed).toBe(1);
    expect(readState().notes[0]?.body).toContain(finding.body);
    expect(Object.keys(readState().pending)).toEqual(["411", "412", "413"]);
    expect(findCachedReview(trustedNotes(), cacheKey)).toEqual(review);
    expect(getHodorReviewShaCandidates(trustedNotes())).toEqual([HEAD]);
  });

  it("reports failure when a rejected POST and its summary fallback both fail", async () => {
    change({ failCreate: true, failSummary: true });
    const result = await postReviewStructured({ ...opts, cacheMarker });
    expect(result.success).toBe(false);
    expect(result.summaryPosted).toBe(false);
    expect(readState().notes).toEqual([]);
  });

  it.each([null, "security"])("prevents false cached CLI success for an incomplete %s publication and preserves completed caching", async key => {
    const reviewKey = key === null ? null : parseReviewKey(key);
    change({ failPublish: [501], failBulk: true });
    const first = await postReviewStructured({ ...opts, cacheMarker, reviewKey });
    expect(first.success).toBe(false);
    expect(first.summaryPosted).toBe(true);

    // Run the actual CLI, reviewPr cache lookup, skip flags, and publisher, not equivalent flags.
    const retried = runCli(key);
    console.log("ACTUAL_CLI_RETRY", JSON.stringify({ key, status: retried.status, reused: retried.output.includes("Reused the existing review"), blockedModel: retried.output.includes("Synthetic LLM stream blocked"), publishAttempts: readState().publishAttempts }));
    expect(retried.status).toBe(1);
    expect(retried.output).toContain("Synthetic LLM stream blocked");
    expect(retried.output).not.toContain("Reused the existing review");
    expect(existsSync(join(root, "model-calls"))).toBe(true);
    expect(readFileSync(join(root, "git-calls"), "utf8")).toContain(`"${"a".repeat(40)}","HEAD"`);
    expectNoCompletionMarkers(reviewKey);
    expect(readState().publishAttempts).toEqual([501]);

    // A fresh computation may deliver with a new owned draft; old pending IDs are never adopted.
    change({ failPublish: [], failBulk: false });
    const delivered = await postReviewStructured({ ...opts, cacheMarker, reviewKey });
    expect(delivered.success).toBe(true);
    expect(readState().published.map(note => note.id)).toEqual([502]);
    expect(Object.keys(readState().pending)).toEqual(["411", "412", "413", "501"]);
    expect(findCachedReview(trustedNotes(reviewKey), cacheKey)).toEqual(review);
    const callsBefore = readFileSync(join(root, "model-calls"), "utf8");
    const cached = runCli(key);
    console.log("ACTUAL_CLI_COMPLETED_CACHE", JSON.stringify({ key, status: cached.status, reused: cached.output.includes("Reused the existing review"), publishAttempts: readState().publishAttempts }));
    expect(cached.status).toBe(0);
    expect(cached.output).toContain("Reused the existing review");
    expect(readFileSync(join(root, "model-calls"), "utf8")).toBe(callsBefore);
    expect(readState().publishAttempts).toEqual([501, 502]);
    expect(readState().published).toHaveLength(1);
  }, 20_000);
});
