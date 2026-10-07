import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  findLatestReviewBase,
  getReviewDiffArgs,
  getChangedFiles,
  getChangedPaths,
  getDiffStats,
  resolveReviewBaseSha,
} from "../src/review-diff.js";
import type { TrustedHodorNote } from "../src/types.js";
import { exec } from "../src/utils/exec.js";

vi.mock("../src/utils/exec.js", () => ({ exec: vi.fn() }));

const sha = "1".repeat(40);
const notes: TrustedHodorNote[] = [{
  body: `<!-- hodor:sha:${sha} -->\n<!-- hodor-review -->`,
  created_at: "2026-07-16T00:00:00Z",
  provenance: "hodor",
}];

describe("changed paths", () => {
  it("preserves rename, copy, deletion, and unusual filename paths", () => {
    expect(getChangedPaths("R100\0old path.ts\0new path.ts\0C100\0new path.ts\0copy\npath.ts\0D\0gone.ts\0"))
      .toEqual(["old path.ts", "new path.ts", "copy\npath.ts", "gone.ts"]);
  });

  it("handles an empty change set", () => {
    expect(getChangedPaths("")).toEqual([]);
  });

  it("rejects incomplete path records", () => {
    expect(() => getChangedPaths("R100\0old.ts\0")).toThrow(/Invalid changed-path/);
  });
});

describe("findLatestReviewBase", () => {
  beforeEach(() => vi.mocked(exec).mockReset());

  it("uses incremental mode while the reviewed commit remains an ancestor", async () => {
    vi.mocked(exec)
      .mockResolvedValueOnce({ stdout: "commit\n", stderr: "" })
      .mockResolvedValueOnce({ stdout: "", stderr: "" });

    await expect(findLatestReviewBase(notes, "/workspace")).resolves.toEqual({
      sha,
      mode: "incremental",
    });
    expect(vi.mocked(exec).mock.calls[1]?.[1]).toEqual([
      "merge-base", "--is-ancestor", sha, "HEAD",
    ]);
  });

  it("uses a snapshot delta when history was rewritten", async () => {
    vi.mocked(exec)
      .mockResolvedValueOnce({ stdout: "commit\n", stderr: "" })
      .mockRejectedValueOnce(new Error("not an ancestor"));

    await expect(findLatestReviewBase(notes, "/workspace")).resolves.toEqual({
      sha,
      mode: "snapshot",
    });
  });
});

describe("resolveReviewBaseSha", () => {
  beforeEach(() => vi.mocked(exec).mockReset());

  it("prefers the known MR diff base without running git", async () => {
    await expect(resolveReviewBaseSha("/workspace", "main", "2".repeat(40)))
      .resolves.toBe("2".repeat(40));
    expect(exec).not.toHaveBeenCalled();
  });

  it("uses the merge base with the target branch", async () => {
    vi.mocked(exec).mockResolvedValueOnce({ stdout: `${"3".repeat(40)}\n`, stderr: "" });

    await expect(resolveReviewBaseSha("/workspace", "main", null)).resolves.toBe("3".repeat(40));
    expect(vi.mocked(exec).mock.calls[0]?.[1]).toEqual(["merge-base", "HEAD", "origin/main"]);
  });

  it("returns null when no base can be computed", async () => {
    vi.mocked(exec).mockRejectedValueOnce(new Error("no merge base"));

    await expect(resolveReviewBaseSha("/workspace", "main", null)).resolves.toBeNull();
  });
});

describe("getReviewDiffArgs", () => {
  it("uses the current GitLab MR base after a rewritten review history", () => {
    expect(getReviewDiffArgs({
      platform: "gitlab",
      targetBranch: "main",
      diffBaseSha: "2".repeat(40),
      previousReviewSha: "1".repeat(40),
      reviewDiffMode: "snapshot",
    })).toEqual([
      "--no-pager", "diff", "2".repeat(40), "HEAD",
    ]);
  });

  it("keeps incremental diffs anchored to the previous review", () => {
    expect(getReviewDiffArgs({
      platform: "gitlab",
      targetBranch: "main",
      diffBaseSha: "2".repeat(40),
      previousReviewSha: "1".repeat(40),
      reviewDiffMode: "incremental",
    })).toEqual([
      "--no-pager", "diff", `${"1".repeat(40)}...HEAD`,
    ]);
  });
});

describe("diff metadata", () => {
  const diff = [
    "diff --git a/src/a.ts b/src/a.ts",
    "--- a/src/a.ts",
    "+++ b/src/a.ts",
    "-const oldValue = 1;",
    "+const newValue = 2;",
    "diff --git a/src/b.ts b/src/b.ts",
    "--- a/src/b.ts",
    "+++ b/src/b.ts",
    "+export const enabled = true;",
  ].join("\n");

  it("counts reviewed files and changed lines", () => {
    expect(getDiffStats(diff)).toEqual({
      files: 2,
      additions: 2,
      deletions: 1,
      bytes: Buffer.byteLength(diff),
    });
    expect(getChangedFiles(diff)).toEqual(["src/a.ts", "src/b.ts"]);
  });
});
