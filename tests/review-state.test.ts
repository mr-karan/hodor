import { describe, expect, it } from "vitest";
import type { HodorDiscussion } from "../src/gitlab.js";
import {
  buildFixCandidates,
  buildFixedReplyBody,
  getFindingFingerprint,
  getFixedMarker,
  MAX_PROMPT_FINDING_THREADS,
  mergeReviewStateFindings,
  resolveFixedThreads,
  selectFindingThreads,
  selectVerifiedFixes,
} from "../src/review-state.js";
import type { ReviewFinding } from "../src/types.js";

const currentFinding: ReviewFinding = {
  title: "[P1] Preserve authorization",
  body: "The new path skips the ownership check.",
  priority: 1,
  code_location: {
    absolute_file_path: "/workspace/src/app.ts",
    line_range: { start: 12, end: 13 },
  },
};

const oldFinding: ReviewFinding = {
  ...currentFinding,
  title: "[P2] Keep the archive schema in sync",
  priority: 2,
};

function discussion(
  finding: ReviewFinding,
  overrides: Partial<HodorDiscussion> = {},
): HodorDiscussion {
  const fingerprint = getFindingFingerprint(finding, "/workspace");
  return fingerprintDiscussion(fingerprint, finding.title, overrides);
}

function fingerprintDiscussion(
  fingerprint: string,
  title: string,
  overrides: Partial<HodorDiscussion> = {},
): HodorDiscussion {
  return {
    discussionId: `discussion-${fingerprint.slice(0, 12)}`,
    noteId: 1,
    body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\n**${title}**\n\nBody.`,
    resolved: false,
    filePath: "src/app.ts",
    line: 12,
    humanReplies: [],
    ...overrides,
  };
}

describe("mergeReviewStateFindings", () => {
  it("normalizes current finding paths and assigns canonical fingerprints", () => {
    const { open: [finding] } = mergeReviewStateFindings([currentFinding], [], "/workspace");

    expect(finding).toMatchObject({
      fingerprint: getFindingFingerprint(currentFinding, "/workspace"),
      filePath: "src/app.ts",
      lineRange: { start: 12, end: 13 },
    });
  });

  it("retains unresolved prior findings and excludes resolved threads", () => {
    const { open } = mergeReviewStateFindings(
      [],
      [discussion(oldFinding), discussion(currentFinding, { resolved: true, noteId: 2 })],
      "/workspace",
    );

    expect(open).toEqual([
      expect.objectContaining({
        title: oldFinding.title,
        priority: 2,
        filePath: "src/app.ts",
      }),
    ]);
  });

  it("prefers the current finding when an open thread has the same fingerprint", () => {
    const stale = discussion(currentFinding);
    stale.body = stale.body.replace("Body.", "Older explanation from the previous review.");

    const { open } = mergeReviewStateFindings([currentFinding], [stale], "/workspace");

    expect(open).toHaveLength(1);
    expect(open[0].body).toBe(currentFinding.body);
    expect(open[0].lineRange).toEqual({ start: 12, end: 13 });
  });

  it("counts unresolved earlier threads as open in every review mode", () => {
    const { open, fixedAwaiting } = mergeReviewStateFindings(
      [],
      [discussion(currentFinding)],
      "/workspace",
    );

    expect(open).toHaveLength(1);
    expect(fixedAwaiting).toBe(0);
  });

  it("honors human resolution when an identical-head review is reused", () => {
    const { open } = mergeReviewStateFindings(
      [currentFinding],
      [discussion(currentFinding, { resolved: true })],
      "/workspace",
      { suppressResolvedCurrent: true },
    );

    expect(open).toEqual([]);
  });

  it("moves a thread verified fixed in this review out of the open set", () => {
    const thread = discussion(oldFinding);
    const [candidate] = buildFixCandidates([thread]);

    const { open, fixedAwaiting } = mergeReviewStateFindings([], [thread], "/workspace", {
      resolvedFindingIds: [candidate.id],
    });

    expect(open).toEqual([]);
    expect(fixedAwaiting).toBe(1);
  });

  it("keeps a thread with a trusted fixed-reply out of the open set without a new claim", () => {
    const { open, fixedAwaiting } = mergeReviewStateFindings(
      [],
      [discussion(oldFinding, { fixedAtSha: "d".repeat(40) })],
      "/workspace",
    );

    expect(open).toEqual([]);
    expect(fixedAwaiting).toBe(1);
  });

  it("reopens a fixed thread when this review reports the same finding again", () => {
    const { open, fixedAwaiting } = mergeReviewStateFindings(
      [currentFinding],
      [discussion(currentFinding, { fixedAtSha: "d".repeat(40) })],
      "/workspace",
    );

    expect(open).toHaveLength(1);
    expect(open[0].fingerprint).toBe(getFindingFingerprint(currentFinding, "/workspace"));
    expect(fixedAwaiting).toBe(0);
  });

  it("does not count a resolved thread as fixed-awaiting", () => {
    const thread = discussion(oldFinding, { resolved: true, fixedAtSha: "d".repeat(40) });

    expect(mergeReviewStateFindings([], [thread], "/workspace")).toEqual({ open: [], fixedAwaiting: 0 });
  });
});

describe("buildFixCandidates", () => {
  it("gives each open thread the first 8 hex of its fingerprint", () => {
    const thread = discussion(oldFinding);
    const fingerprint = getFindingFingerprint(oldFinding, "/workspace");

    expect(buildFixCandidates([thread])).toEqual([
      {
        id: fingerprint.slice(0, 8),
        fingerprint,
        discussionId: thread.discussionId,
        title: oldFinding.title,
        filePath: "src/app.ts",
      },
    ]);
  });

  it("lengthens colliding prefixes deterministically in steps of 4 hex", () => {
    const a = `12345678aaaa1${"0".repeat(51)}`;
    const b = `12345678aaaa2${"0".repeat(51)}`;
    const c = `12345678bbbb${"0".repeat(52)}`;
    const threads = [a, b, c].map((fingerprint) => fingerprintDiscussion(fingerprint, "[P2] Issue"));

    const ids = buildFixCandidates(threads).map((candidate) => candidate.id);
    const reversed = buildFixCandidates([...threads].reverse()).map((candidate) => candidate.id);

    expect(ids).toEqual(["12345678aaaa1000", "12345678aaaa2000", "12345678bbbb"]);
    expect([...reversed].reverse()).toEqual(ids);
  });

  it("makes ids unique against fixed threads too, so they map back to one thread", () => {
    const open = `abcdef01${"1".repeat(56)}`;
    const fixed = `abcdef01${"2".repeat(56)}`;
    const threads = [
      fingerprintDiscussion(open, "[P2] Open"),
      fingerprintDiscussion(fixed, "[P2] Fixed", { fixedAtSha: "d".repeat(40) }),
    ];

    expect(buildFixCandidates(threads).map((candidate) => candidate.id)).toEqual(["abcdef011111"]);
  });

  it("gives no id to open threads that share a full fingerprint", () => {
    const fingerprint = "a".repeat(64);
    const threads = [
      fingerprintDiscussion(fingerprint, "[P2] Same", { discussionId: "one" }),
      fingerprintDiscussion(fingerprint, "[P2] Same", { discussionId: "two" }),
    ];

    expect(buildFixCandidates(threads)).toEqual([]);
    expect(resolveFixedThreads(["aaaaaaaa"], threads).size).toBe(0);
  });

  it("excludes resolved threads and threads already marked fixed", () => {
    const threads = [
      discussion(oldFinding, { resolved: true }),
      discussion(currentFinding, { fixedAtSha: "d".repeat(40) }),
    ];

    expect(buildFixCandidates(threads)).toEqual([]);
  });
});

describe("selectVerifiedFixes", () => {
  const threads = [
    discussion(oldFinding),
    discussion(currentFinding, { filePath: "src/other.ts" }),
  ];
  const candidates = buildFixCandidates(threads);
  const [inDiff, outsideDiff] = candidates;

  it("accepts a presented id on a changed file that was not reported again", () => {
    expect(
      selectVerifiedFixes([inDiff.id], candidates, {
        changedFiles: ["src/app.ts"],
        currentFingerprints: new Set(),
      }),
    ).toEqual({ accepted: [inDiff.id], rejected: [] });
  });

  it("drops unknown ids, full fingerprints that were not presented, files outside the diff, and duplicates", () => {
    const result = selectVerifiedFixes(
      ["deadbeef", inDiff.fingerprint, outsideDiff.id, inDiff.id, inDiff.id],
      candidates,
      { changedFiles: ["src/app.ts"], currentFingerprints: new Set() },
    );

    expect(result).toEqual({
      accepted: [inDiff.id],
      rejected: [
        { id: "deadbeef", reason: "unknown id" },
        { id: inDiff.fingerprint, reason: "unknown id" },
        { id: outsideDiff.id, reason: "file not in the reviewed diff" },
        { id: inDiff.id, reason: "duplicate id" },
      ],
    });
  });

  it("drops an id whose finding this review reported again", () => {
    const result = selectVerifiedFixes([inDiff.id], candidates, {
      changedFiles: ["src/app.ts"],
      currentFingerprints: new Set([inDiff.fingerprint]),
    });

    expect(result).toEqual({
      accepted: [],
      rejected: [{ id: inDiff.id, reason: "re-reported in this review" }],
    });
  });
});

describe("fixed-reply markers", () => {
  it("round-trips the fingerprint and head SHA", () => {
    const fingerprint = "a".repeat(64);
    const sha = "0123456789abcdef0123456789abcdef01234567";
    const body = buildFixedReplyBody(fingerprint, sha);

    expect(body).toBe(
      "<!-- hodor-review -->\n" +
        `<!-- hodor:fixed:${fingerprint}:${sha} -->\n` +
        "Fixed in `01234567`. Resolve this thread if you agree.",
    );
    expect(getFixedMarker(body)).toEqual({ fingerprint, sha });
  });
});

describe("selectFindingThreads", () => {
  it("shows open threads first, then resolved threads newest first, with human replies", () => {
    const open = discussion(oldFinding, {
      humanReplies: [
        { author: "alice", body: "👍" },
        { author: "alice", body: "+1" },
        { author: "bob", body: "working on it" },
      ],
    });
    const olderResolved = discussion(currentFinding, {
      discussionId: "older",
      resolved: true,
      resolvedBy: "carol",
      updatedAt: "2026-09-01T00:00:00Z",
    });
    const newerResolved = fingerprintDiscussion("b".repeat(64), "[P3] Naming", {
      resolved: true,
      resolvedBy: "dave",
      updatedAt: "2026-09-20T00:00:00Z",
      humanReplies: [
        { author: "dave", body: "one" },
        { author: "dave", body: "two" },
        { author: "dave", body: "false positive" },
      ],
    });
    const threads = [olderResolved, newerResolved, open];

    const selected = selectFindingThreads(threads, buildFixCandidates(threads), ["src/app.ts"]);

    expect(selected).toEqual([
      {
        fixId: getFindingFingerprint(oldFinding, "/workspace").slice(0, 8),
        title: oldFinding.title,
        filePath: "src/app.ts",
        status: "open",
        replies: [{ author: "bob", body: "working on it" }],
      },
      {
        title: "[P3] Naming",
        filePath: "src/app.ts",
        status: "resolved",
        resolvedBy: "dave",
        replies: [
          { author: "dave", body: "two" },
          { author: "dave", body: "false positive" },
        ],
      },
      {
        title: currentFinding.title,
        filePath: "src/app.ts",
        status: "resolved",
        resolvedBy: "carol",
        replies: [],
      },
    ]);
  });

  it("gives no fix id to open threads on files outside the reviewed diff", () => {
    const thread = discussion(oldFinding, { filePath: "src/other.ts" });

    const [selected] = selectFindingThreads([thread], buildFixCandidates([thread]), ["src/app.ts"]);

    expect(selected.status).toBe("open");
    expect(selected.fixId).toBeUndefined();
  });

  it("marks threads with a trusted fixed-reply as fixed without an id", () => {
    const thread = discussion(oldFinding, { fixedAtSha: "d".repeat(40) });

    const [selected] = selectFindingThreads([thread], buildFixCandidates([thread]), ["src/app.ts"]);

    expect(selected).toMatchObject({ status: "fixed" });
    expect(selected.fixId).toBeUndefined();
  });

  it("keeps at most 15 threads and prefers open ones", () => {
    const resolved = Array.from({ length: 20 }, (_, index) =>
      fingerprintDiscussion(index.toString(16).padStart(2, "0").repeat(32), `[P3] Resolved ${index}`, {
        resolved: true,
        updatedAt: new Date(Date.UTC(2026, 8, index + 1)).toISOString(),
      }),
    );
    const open = fingerprintDiscussion("f".repeat(64), "[P1] Still open");
    const threads = [...resolved, open];

    const selected = selectFindingThreads(threads, buildFixCandidates(threads), ["src/app.ts"]);

    expect(MAX_PROMPT_FINDING_THREADS).toBe(15);
    expect(selected).toHaveLength(15);
    expect(selected[0].title).toBe("[P1] Still open");
    expect(selected[1].title).toBe("[P3] Resolved 19");
    expect(selected[14].title).toBe("[P3] Resolved 6");
  });
});
