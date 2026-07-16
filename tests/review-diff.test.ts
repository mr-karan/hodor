import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  findLatestReviewBase,
  getChangedFiles,
  getDiffStats,
} from "../src/review-diff.js";
import { exec } from "../src/utils/exec.js";

vi.mock("../src/utils/exec.js", () => ({ exec: vi.fn() }));

const sha = "1".repeat(40);
const notes = [{
  body: `<!-- hodor:sha:${sha} -->\n<!-- hodor-review -->`,
  created_at: "2026-07-16T00:00:00Z",
}];

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
