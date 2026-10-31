import { beforeEach, describe, expect, it, vi } from "vitest";
import { postReviewStructured } from "../src/publisher.js";
import { parseReviewKey, withReviewKey, type ReviewKey } from "../src/review-key.js";
import { getFindingFingerprint } from "../src/review-state.js";
import type { ReviewFinding, ReviewOutput } from "../src/types.js";

const mocks = vi.hoisted(() => ({ exec: vi.fn(), execJson: vi.fn() }));
vi.mock("../src/utils/exec.js", () => ({ exec: mocks.exec, execJson: mocks.execJson }));

const URL = "https://gitlab.example.com/acme/app/-/merge_requests/42";
const SHA = "a".repeat(40);
const SECURITY = parseReviewKey("security");
const BOT = { id: 7, username: "review-bot" };
const ROOT = "projects/acme%2Fapp/merge_requests/42";
const finding: ReviewFinding = {
  title: "[P2] Validate the identifier",
  body: "Synthetic finding.",
  priority: 2,
  code_location: {
    absolute_file_path: "/workspace/src/app.ts",
    line_range: { start: 1, end: 1 },
  },
};
const secondFinding: ReviewFinding = {
  ...finding,
  title: "[P2] Bound the input",
  code_location: {
    absolute_file_path: "/workspace/src/app.ts",
    line_range: { start: 2, end: 2 },
  },
};
const thirdFinding: ReviewFinding = {
  ...finding,
  title: "[P2] Check the result",
  code_location: {
    absolute_file_path: "/workspace/src/app.ts",
    line_range: { start: 3, end: 3 },
  },
};

function review(findings: ReviewFinding[] = [finding]): ReviewOutput {
  return {
    findings,
    overall_correctness: "patch is incorrect",
    overall_explanation: "Synthetic review.",
  };
}

function post(reviewKey: ReviewKey | null = null, findings?: ReviewFinding[]) {
  return postReviewStructured({
    prUrl: URL,
    review: review(findings),
    workspacePath: "/workspace",
    headSha: SHA,
    reviewKey,
    commitStatus: true,
  });
}

function deferred() {
  let release = () => {};
  const promise = new Promise<void>((resolve) => { release = resolve; });
  return { promise, release };
}

/** In-memory GitLab boundary: bulk publication deliberately follows the user-wide API contract. */
function gitlab() {
  const pending = new Map<number, string>([
    [90, "A human's pending draft on the bot account"],
    [91, withReviewKey("An older keyed run's pending draft", SECURITY)],
    [92, "<!-- hodor-review -->\nAn older default run's pending draft"],
  ]);
  const published = new Map<number, string>();
  const publishAttempts: number[] = [];
  const bulkAttempts: string[] = [];
  const failures = new Map<number, string>();
  let nextId = 101;
  let draftResponse: ((id: number) => unknown) | undefined;
  let afterCreate: ((id: number) => Promise<void>) | undefined;

  mocks.execJson.mockImplementation(async (_command: string, args: string[]) => {
    const endpoint = args[1];
    if (endpoint === "user") return BOT;
    if (endpoint === ROOT) return { diff_refs: { base_sha: SHA, start_sha: SHA, head_sha: SHA } };
    throw new Error(`unexpected JSON endpoint: ${endpoint}`);
  });
  mocks.exec.mockImplementation(async (_command: string, args: string[], options?: { input?: string }) => {
    const endpoint = args[1];
    if (endpoint === `${ROOT}/draft_notes`) {
      expect(args).toContain("POST");
      const payload: unknown = JSON.parse(options?.input ?? "{}");
      if (typeof payload !== "object" || payload === null || !("note" in payload) || typeof payload.note !== "string") {
        throw new Error("invalid draft payload");
      }
      const id = nextId++;
      pending.set(id, payload.note);
      await afterCreate?.(id);
      return { stdout: JSON.stringify(draftResponse ? draftResponse(id) : { id }) ?? "", stderr: "" };
    }
    if (endpoint === `${ROOT}/draft_notes/bulk_publish`) {
      bulkAttempts.push(endpoint);
      expect(args).toContain("POST");
      for (const [id, body] of pending) published.set(id, body);
      pending.clear();
    } else if (endpoint?.startsWith(`${ROOT}/draft_notes/`) && endpoint.endsWith("/publish")) {
      expect(args).toEqual(["api", endpoint, "--method", "PUT"]);
      const id = Number(endpoint.split("/").at(-2));
      publishAttempts.push(id);
      const failure = failures.get(id);
      if (failure) throw new Error(failure);
      const body = pending.get(id);
      if (body === undefined) throw new Error("404 Not Found");
      published.set(id, body);
      pending.delete(id);
    } else if (endpoint === `${ROOT}/discussions?per_page=100`) {
      return {
        stdout: JSON.stringify([...published].map(([id, body]) => ({
          id: `discussion-${id}`,
          notes: [{ id, body, author: BOT, resolvable: true, resolved: false }],
        }))),
        stderr: "",
      };
    } else if (endpoint === `${ROOT}/notes?per_page=100`) {
      return { stdout: "[]", stderr: "" };
    } else if (endpoint === `${ROOT}/notes` && args.includes("POST")) {
      return { stdout: '{"id":900}', stderr: "" };
    } else if (endpoint !== `projects/acme%2Fapp/statuses/${SHA}`) {
      throw new Error(`unexpected endpoint: ${endpoint}`);
    }
    return { stdout: "", stderr: "" };
  });
  return {
    pending, published, publishAttempts, bulkAttempts, failures,
    respondWith: (response: (id: number) => unknown) => { draftResponse = response; },
    onCreate: (callback: (id: number) => Promise<void>) => { afterCreate = callback; },
  };
}

beforeEach(() => {
  mocks.exec.mockReset();
  mocks.execJson.mockReset();
});

describe("invocation-owned GitLab draft publication", () => {
  it.each([
    { parkedKey: SECURITY, activeKey: null },
    { parkedKey: null, activeKey: SECURITY },
    { parkedKey: null, activeKey: null },
    { parkedKey: SECURITY, activeKey: SECURITY },
  ])("isolates concurrent runs with keys $parkedKey and $activeKey", async ({ parkedKey, activeKey }) => {
    const api = gitlab();
    const parked = deferred();
    const ready = deferred();
    api.onCreate(async (id) => {
      if (id === 101) {
        ready.release();
        await parked.promise;
      }
    });
    const parkedRun = post(parkedKey);
    await ready.promise;
    try {
      const activeResult = await post(activeKey);
      expect(activeResult.success).toBe(true);
      expect(activeResult.draftsPublished).toBe(true);
      expect(api.publishAttempts).toEqual([102]);
      expect([...api.published.keys()]).toEqual([102]);
      expect([...api.pending.keys()]).toEqual([90, 91, 92, 101]);
    } finally {
      parked.release();
      await parkedRun;
    }
    const parkedResult = await parkedRun;
    expect(parkedResult.success).toBe(true);
    expect(parkedResult.draftsPublished).toBe(true);
    expect(api.publishAttempts).toEqual([102, 101]);
    expect([...api.published.keys()]).toEqual([102, 101]);
    expect([...api.pending.keys()]).toEqual([90, 91, 92]);
    expect(api.bulkAttempts).toEqual([]);
  });

  it("publishes every exact returned ID and accepts numeric string IDs", async () => {
    const api = gitlab();
    api.respondWith((id) => ({ id: String(id) }));
    const result = await post(null, [finding, secondFinding]);
    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(2);
    expect(result.draftsPublished).toBe(true);
    expect(api.publishAttempts).toEqual([101, 102]);
    expect([...api.published.keys()]).toEqual([101, 102]);
    expect([...api.pending.keys()]).toEqual([90, 91, 92]);
    expect(api.bulkAttempts).toEqual([]);
  });

  it.each([undefined, null, "", "../bulk_publish", "101/publish", "1e2", 0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "reports incomplete delivery for invalid returned draft ID %s without publishing unrelated drafts", async (id) => {
      const api = gitlab();
      api.respondWith((createdId) => createdId === 101 ? { id } : { id: createdId });
      const result = await post(null, [finding, secondFinding]);
      expect(result.success).toBe(false);
      expect(result.summaryPosted).toBe(true);
      expect(result.draftsPublished).toBe(false);
      expect(result.errors?.join("\n")).toMatch(/draft.*id/i);
      expect(api.publishAttempts).toEqual([102]);
      expect([...api.pending.keys()]).toEqual([90, 91, 92, 101]);
      expect(api.bulkAttempts).toEqual([]);
    },
  );

  it.each([undefined, null, [], "invalid response", { id: false }])(
    "fails closed for a malformed creation response %j", async (response) => {
      const api = gitlab();
      api.respondWith(() => response);
      const result = await post();
      expect(result.success).toBe(false);
      expect(result.inlineCreated).toBe(1);
      expect(result.draftsPublished).toBe(false);
      expect(result.errors?.join("\n")).toMatch(/draft.*id/i);
      expect(api.publishAttempts).toEqual([]);
      expect([...api.pending.keys()]).toEqual([90, 91, 92, 101]);
      expect(api.bulkAttempts).toEqual([]);
    },
  );

  it("reports failure when no inline draft can be published without a summary", async () => {
    const api = gitlab();
    api.failures.set(101, "403 Forbidden");
    const result = await postReviewStructured({
      prUrl: URL, review: review(), reviewStyle: "inline", workspacePath: "/workspace",
    });
    expect(result.success).toBe(false);
    expect(result.summaryPosted).toBe(false);
    expect(result.draftsPublished).toBe(false);
    expect(result.error).toContain("403 Forbidden");
    expect(api.publishAttempts).toEqual([101]);
    expect([...api.pending.keys()]).toEqual([90, 91, 92, 101]);
    expect(api.bulkAttempts).toEqual([]);
  });

  it.each(["404 Not Found", "500 Internal Server Error", "request timed out"])(
    "reports partial failure (%s), attempts later IDs, and isolates a fresh invocation retry", async (message) => {
      const api = gitlab();
      api.failures.set(102, message);
      const result = await post(SECURITY, [finding, secondFinding, thirdFinding]);
      expect(result.success).toBe(false);
      expect(result.draftsPublished).toBe(false);
      expect(result.summaryPosted).toBe(true);
      expect(result.commitStatusPosted).toBe(true);
      expect(result.errors?.join("\n")).toContain(`Failed to publish draft note 102 for MR !42: ${message}`);
      expect(api.publishAttempts).toEqual([101, 102, 103]);
      expect([...api.published.keys()]).toEqual([101, 103]);
      expect([...api.pending.keys()]).toEqual([90, 91, 92, 102]);
      expect(api.bulkAttempts).toEqual([]);

      api.failures.clear();
      const retried = await post(SECURITY, [finding, secondFinding, thirdFinding]);
      expect(retried.success).toBe(true);
      expect(retried.inlineCreated).toBe(1);
      expect(retried.draftsPublished).toBe(true);
      expect(api.publishAttempts).toEqual([101, 102, 103, 104]);
      expect([...api.pending.keys()]).toEqual([90, 91, 92, 102]);
      expect([...api.published.values()].filter((body) => body.includes(getFindingFingerprint(secondFinding, "/workspace")))).toHaveLength(1);
      expect(api.bulkAttempts).toEqual([]);
    },
  );

  it.each(["summary", "inline", "hybrid"] as const)("does not publish any drafts for a %s run with no findings", async (reviewStyle) => {
    const api = gitlab();
    const result = await postReviewStructured({ prUrl: URL, review: review([]), reviewStyle });
    expect(result.success).toBe(true);
    expect(api.publishAttempts).toEqual([]);
    expect(api.bulkAttempts).toEqual([]);
    expect([...api.pending.keys()]).toEqual([90, 91, 92]);
  });
});
