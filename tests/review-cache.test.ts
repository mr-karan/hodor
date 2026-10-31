import { gunzipSync, gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  buildReviewCacheMarker,
  findCachedReview,
  getReviewCacheKey,
  type ReviewCacheScope,
} from "../src/review-cache.js";
import type { ReviewFinding, ReviewOutput } from "../src/types.js";

const scope: ReviewCacheScope = {
  platform: "gitlab",
  host: "gitlab.example.com",
  projectPath: "team/widget",
  reviewNumber: 42,
  targetBranch: "main",
  baseSha: "b".repeat(40),
};

const review: ReviewOutput = {
  findings: [{
    title: "[P1] Preserve the guard",
    body: "Removing this guard lets a null payload reach the decoder.",
    priority: 1,
    code_location: {
      absolute_file_path: "/builds/private/team/widget/src/api.ts",
      line_range: { start: 12, end: 13 },
    },
    existing_code: "decode(payload)",
  }],
  overall_correctness: "patch is incorrect",
  overall_explanation: "The null guard is required.",
};

// Synthetic stand-in for a credential the reviewed source contains.
const SOURCE_SENTINEL = "hodor-synthetic-sentinel-5f0c2a9e";

/** Decodes a posted marker the way anyone who can read the note can. */
function decodeMarker(marker: string): string {
  const encoded = marker.match(/hodor:cache:v1:([A-Za-z0-9_-]+)/)?.[1] ?? "";
  return gunzipSync(Buffer.from(encoded, "base64url")).toString("utf-8");
}

const findingWithSecretSource: ReviewFinding = {
  title: "[P1] Remove the committed API token",
  body: "The token is committed in source and ships with every build.",
  priority: 1,
  code_location: {
    absolute_file_path: "/workspace/src/config.ts",
    line_range: { start: 4, end: 4 },
  },
  existing_code: `const apiToken = "${SOURCE_SENTINEL}";`,
  suggestion: "const apiToken = process.env.API_TOKEN;",
};

describe("review cache", () => {
  it("round-trips a validated review without retaining workspace paths", () => {
    const key = getReviewCacheKey({
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      instructions: ["default review profile"],
      guidanceSnapshotSha: "c".repeat(40),
    });
    const marker = buildReviewCacheMarker(key, review, "/builds/private/team/widget");
    const cached = findCachedReview([{
      body: `<!-- hodor:sha:${"a".repeat(40)} -->\n${marker}\n<!-- hodor-review -->`,
      created_at: "2026-07-16T00:00:00Z",
      provenance: "hodor",
    }], key);

    expect(marker).not.toContain("private");
    expect(cached?.findings[0].code_location.absolute_file_path).toBe("/workspace/src/api.ts");
    expect(cached?.overall_correctness).toBe("patch is incorrect");
  });

  it("prefers the most recently updated rolling-summary cache", () => {
    const key = getReviewCacheKey({
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      instructions: ["default review profile"],
      guidanceSnapshotSha: "c".repeat(40),
    });
    const legacyReview = {
      ...review,
      overall_explanation: "Legacy cache.",
    };
    const rollingReview = {
      ...review,
      overall_explanation: "Updated rolling cache.",
    };

    const cached = findCachedReview([
      {
        body: buildReviewCacheMarker(key, legacyReview),
        created_at: "2026-07-16T00:00:00Z",
        updated_at: "2026-07-16T00:00:00Z",
        provenance: "hodor",
      },
      {
        body: buildReviewCacheMarker(key, rollingReview),
        created_at: "2026-07-15T00:00:00Z",
        updated_at: "2026-07-17T00:00:00Z",
        provenance: "hodor",
      },
    ], key);

    expect(cached?.overall_explanation).toBe("Updated rolling cache.");
  });

  it("does not reuse a result with a different review identity", () => {
    const oldKey = getReviewCacheKey({
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      instructions: ["default review profile"],
      guidanceSnapshotSha: "c".repeat(40),
    });
    const newKey = getReviewCacheKey({
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      requestedReasoningEffort: "high",
      instructions: ["default review profile"],
      guidanceSnapshotSha: "c".repeat(40),
    });
    const marker = buildReviewCacheMarker(oldKey, review);

    expect(findCachedReview([{ body: `${marker}\n<!-- hodor-review -->`, provenance: "hodor" }], newKey))
      .toBeNull();
  });

  it("changes cache identity when the explicit instructions or focus change", () => {
    const base = {
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      instructions: ["Review authentication changes."],
      guidanceSnapshotSha: "c".repeat(40),
    };

    const sameContent = getReviewCacheKey(base);
    const changedInstructions = getReviewCacheKey({
      ...base,
      instructions: ["Review authorization changes."],
    });
    const changedFocus = getReviewCacheKey({
      ...base,
      focus: "Prioritize tenant isolation.",
    });

    expect(getReviewCacheKey({ ...base })).toBe(sameContent);
    expect(changedInstructions).not.toBe(sameContent);
    expect(changedFocus).not.toBe(sameContent);
    expect(getReviewCacheKey({ ...base, guidanceSnapshotSha: "d".repeat(40) })).not.toBe(sameContent);
    expect(getReviewCacheKey({ ...base, instructions: ["First", "Second"] }))
      .not.toBe(getReviewCacheKey({ ...base, instructions: ["Second", "First"] }));
  });

  const scopeChanges: Array<[string, Partial<ReviewCacheScope>]> = [
    ["platform", { platform: "github" }],
    ["host", { host: "gitlab.other.example" }],
    ["project path", { projectPath: "team/other" }],
    ["MR number", { reviewNumber: 43 }],
    ["target branch", { targetBranch: "release" }],
    ["target base SHA", { baseSha: "c".repeat(40) }],
  ];

  it.each(scopeChanges)("changes cache identity when the %s changes", (_field, change) => {
    const base = {
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      instructions: ["default review profile"],
      guidanceSnapshotSha: "c".repeat(40),
    };

    expect(getReviewCacheKey({ ...base, scope: { ...scope, ...change } }))
      .not.toBe(getReviewCacheKey(base));
  });

  it("keeps cache identity stable for an identical retry", () => {
    const opts = {
      scope,
      headSha: "a".repeat(40),
      model: "anthropic/claude-opus-4-7",
      instructions: ["default review profile"],
      guidanceSnapshotSha: "c".repeat(40),
    };

    expect(getReviewCacheKey({ ...opts, scope: { ...scope } })).toBe(getReviewCacheKey(opts));
  });

  it("invalidates same-commit reviews for new or edited participant evidence", () => {
    const opts = { scope, headSha: "a".repeat(40), model: "test", guidanceSnapshotSha: "b".repeat(40) };
    const note = { id: 42, body: "Fixed by the new helper", author: { id: 8 } };
    const original = getReviewCacheKey({ ...opts, humanNotes: [note] });
    expect(getReviewCacheKey({ ...opts, humanNotes: [{ ...note, body: "Only partly fixed" }] })).not.toBe(original);
    expect(getReviewCacheKey({ ...opts, humanNotes: [note, { id: 43, body: "See the revocation caller" }] })).not.toBe(original);
    expect(getReviewCacheKey({ ...opts, humanNotes: [note, { id: 43, body: "resolved thread", system: true }] })).toBe(original);
    expect(getReviewCacheKey({ ...opts, humanNotes: [{ ...note, updated_at: "2026-10-07" }] })).toBe(original);
  });

  it("preserves canonical fix confirmations in cached reviews", () => {
    const fingerprint = "f".repeat(64);
    const cached = findCachedReview([{
      provenance: "hodor",
      body: buildReviewCacheMarker("key", { ...review, resolved_findings: [fingerprint] }),
    }], "key");
    expect(cached?.resolved_findings).toEqual([fingerprint]);
  });

  it("posts no source snippet in the marker and still reuses the review", () => {
    const marker = buildReviewCacheMarker("key", {
      ...review,
      findings: [findingWithSecretSource, review.findings[0]],
    });

    expect(decodeMarker(marker)).not.toContain(SOURCE_SENTINEL);
    const cached = findCachedReview([{ provenance: "hodor", body: marker }], "key");
    const { existing_code: _existingCode, ...published } = findingWithSecretSource;
    expect(cached?.findings[0]).toEqual(published);
    expect(cached?.findings[0]).not.toHaveProperty("existing_code");
    expect(cached?.findings[1].title).toBe(review.findings[0].title);
    expect(cached?.overall_correctness).toBe("patch is incorrect");
  });

  it("reuses a legacy marker without carrying its source snippet forward", () => {
    // Markers posted before snippets were dropped hold the full finding.
    const legacyMarker = `<!-- hodor:cache:v1:${gzipSync(JSON.stringify({
      key: "key",
      review: { ...review, findings: [findingWithSecretSource] },
    })).toString("base64url")} -->`;

    const cached = findCachedReview([{ provenance: "hodor", body: legacyMarker }], "key");

    expect(cached?.findings).toHaveLength(1);
    expect(cached?.findings[0]).not.toHaveProperty("existing_code");
    expect(cached?.findings[0].suggestion).toBe(findingWithSecretSource.suggestion);
    expect(cached?.findings[0].code_location).toEqual(findingWithSecretSource.code_location);
    // A reused review is re-posted with a marker rebuilt from the decoded review.
    expect(decodeMarker(buildReviewCacheMarker("key", cached ?? review))).not.toContain(SOURCE_SENTINEL);
  });

  it("ignores malformed cache markers", () => {
    expect(findCachedReview([{
      body: "<!-- hodor:cache:v1:not-valid-gzip -->\n<!-- hodor-review -->",
      provenance: "hodor",
    }], "key")).toBeNull();
  });
});
