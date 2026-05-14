import { describe, it, expect, vi } from "vitest";
import {
  detectPlatform,
  filterEmbeddedDiff,
  getHodorReviewShaCandidates,
  parsePrUrl,
} from "../src/agent.js";
import { formatMetricsMarkdown } from "../src/metrics.js";
import type { ReviewMetrics } from "../src/types.js";

// Mock the exec module at the module level
let capturedArgs: string[] = [];
vi.mock("../src/utils/exec.js", () => ({
  exec: vi.fn(async (_cmd: string, args: string[]) => {
    capturedArgs = args;
    return { stdout: "", stderr: "" };
  }),
  execJson: vi.fn(async () => ({})),
}));

describe("detectPlatform", () => {
  it.each([
    ["https://github.com/foo/bar/pull/42", "github"],
    ["https://gitlab.com/foo/bar/-/merge_requests/3", "gitlab"],
    ["https://gitlab.example.dev/group/repo/-/merge_requests/19", "gitlab"],
    ["https://gitea.example.com/foo/bar/pulls/5", "gitea"],
    ["https://codeberg.org/foo/bar/pulls/10", "gitea"],
    ["https://forgejo.example.org/foo/bar/pulls/1", "gitea"],
    ["https://code.mycompany.com/team/project/pulls/77", "gitea"],
  ] as const)("detects platform for %s", (url, expected) => {
    expect(detectPlatform(url)).toBe(expected);
  });

  it("throws on unrecognized platform", () => {
    expect(() => detectPlatform("https://bitbucket.org/foo/bar/pull-requests/1")).toThrow(
      /Cannot detect platform/,
    );
  });
});

describe("parsePrUrl", () => {
  it("parses GitHub PR URL", () => {
    const result = parsePrUrl("https://github.com/octo/hoDor/pull/123");
    expect(result.owner).toBe("octo");
    expect(result.repo).toBe("hoDor");
    expect(result.prNumber).toBe(123);
    expect(result.host).toBe("github.com");
  });

  it("parses GitLab MR URL with subgroups", () => {
    const url =
      "https://gitlab.example.com/org/security/tools/hodor/-/merge_requests/99";
    const result = parsePrUrl(url);
    expect(result.owner).toBe("org/security/tools");
    expect(result.repo).toBe("hodor");
    expect(result.prNumber).toBe(99);
    expect(result.host).toBe("gitlab.example.com");
  });

  it("parses Gitea PR URL", () => {
    const result = parsePrUrl("https://gitea.example.com/acme/widget/pulls/42");
    expect(result.owner).toBe("acme");
    expect(result.repo).toBe("widget");
    expect(result.prNumber).toBe(42);
    expect(result.host).toBe("gitea.example.com");
  });

  it("parses Codeberg PR URL", () => {
    const result = parsePrUrl("https://codeberg.org/user/repo/pulls/7");
    expect(result.owner).toBe("user");
    expect(result.repo).toBe("repo");
    expect(result.prNumber).toBe(7);
    expect(result.host).toBe("codeberg.org");
  });

  it("throws on invalid URL", () => {
    expect(() => parsePrUrl("https://gitlab.com/foo/bar/issues/1")).toThrow();
  });

  it("throws on non-numeric PR number", () => {
    expect(() => parsePrUrl("https://github.com/foo/bar/pull/abc")).toThrow(
      /Invalid PR number/,
    );
  });

  it("throws on non-numeric MR number", () => {
    expect(() =>
      parsePrUrl("https://gitlab.com/foo/bar/-/merge_requests/xyz"),
    ).toThrow(/Invalid MR number/);
  });

  it("throws on non-numeric Gitea PR number", () => {
    expect(() =>
      parsePrUrl("https://gitea.example.com/foo/bar/pulls/abc"),
    ).toThrow(/Invalid PR number/);
  });
});

describe("formatMetricsMarkdown", () => {
  it("includes expected fields", () => {
    const metrics: ReviewMetrics = {
      inputTokens: 1000,
      outputTokens: 80,
      cacheReadTokens: 900,
      cacheWriteTokens: 0,
      totalTokens: 1080,
      cost: 1.2345,
      turns: 3,
      toolCalls: 8,
      durationSeconds: 125,
    };

    const markdown = formatMetricsMarkdown(metrics);

    expect(markdown).toContain("**Review Metrics**");
    expect(markdown).toContain("3 turns");
    expect(markdown).toContain("8 tool calls");
    expect(markdown).toContain("2m 5s");
    expect(markdown).toContain("in `1.9K`"); // totalInput = inputTokens (1000) + cacheReadTokens (900)
    expect(markdown).toContain("cached `900`");
    expect(markdown).toContain("out `80`");
    expect(markdown).toContain("Cost: `$1.2345`");
  });

  it("omits cache when zero", () => {
    const metrics: ReviewMetrics = {
      inputTokens: 500,
      outputTokens: 100,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      totalTokens: 600,
      cost: 0,
      turns: 1,
      toolCalls: 2,
      durationSeconds: 30,
    };

    const markdown = formatMetricsMarkdown(metrics);

    expect(markdown).not.toContain("cached");
    expect(markdown).not.toContain("Cost");
  });
});

describe("postReviewComment", () => {
  it("appends model and metrics footer", async () => {
    // postReviewComment uses the mocked exec from module level
    const { postReviewComment } = await import("../src/agent.js");

    const result = await postReviewComment({
      prUrl: "https://github.com/foo/bar/pull/42",
      reviewText: "Review body",
      model: "openai/gpt-5",
      metricsFooter: "**Review Metrics**\n- Total: `123`",
    });

    expect(result.success).toBe(true);
    const bodyIndex = capturedArgs.indexOf("--body");
    const capturedBody = bodyIndex >= 0 ? capturedArgs[bodyIndex + 1] : "";
    expect(capturedBody).toContain(
      "Review generated by Hodor (model: `openai/gpt-5`)",
    );
    expect(capturedBody).toContain("**Review Metrics**");
  });
});

const SHA_OLD = "1111111111111111111111111111111111111111";
const SHA_MID = "2222222222222222222222222222222222222222";
const SHA_NEW = "3333333333333333333333333333333333333333";

describe("getHodorReviewShaCandidates", () => {
  it("sorts Hodor review markers newest first by created_at", () => {
    const shas = getHodorReviewShaCandidates([
      {
        body: `<!-- hodor:sha:${SHA_NEW} -->\nlatest`,
        created_at: "2026-05-13T15:44:51.465+05:30",
      },
      {
        body: `<!-- hodor:sha:${SHA_MID} -->\nmid`,
        created_at: "2026-04-29T11:46:20.334+05:30",
      },
      {
        body: `<!-- hodor:sha:${SHA_OLD} -->\noldest`,
        created_at: "2026-04-01T15:08:12.964+05:30",
      },
    ]);

    expect(shas).toEqual([SHA_NEW, SHA_MID, SHA_OLD]);
  });

  it("does not let API order make the oldest review win", () => {
    const shas = getHodorReviewShaCandidates([
      {
        body: `<!-- hodor:sha:${SHA_OLD} -->\noldest`,
        created_at: "2026-04-01T15:08:12.964+05:30",
      },
      {
        body: `<!-- hodor:sha:${SHA_NEW} -->\nlatest`,
        created_at: "2026-05-13T15:44:51.465+05:30",
      },
    ]);

    expect(shas).toEqual([SHA_NEW, SHA_OLD]);
  });

  it("deduplicates repeated review markers", () => {
    const shas = getHodorReviewShaCandidates([
      {
        body: `<!-- hodor:sha:${SHA_NEW} -->\nlatest`,
        created_at: "2026-05-13T15:44:51.465+05:30",
      },
      {
        body: `<!-- hodor:sha:${SHA_NEW} -->\nduplicate`,
        created_at: "2026-05-13T15:44:52.465+05:30",
      },
    ]);

    expect(shas).toEqual([SHA_NEW]);
  });
});

const DIFF_HEADER = (path: string) =>
  `diff --git a/${path} b/${path}\nindex abc..def 100644\n--- a/${path}\n+++ b/${path}\n@@ -1,1 +1,1 @@\n-old\n+new\n`;

describe("filterEmbeddedDiff", () => {
  it("passes through a diff with no skippable files", () => {
    const raw = DIFF_HEADER("src/main.go") + DIFF_HEADER("src/util.go");
    const { filtered, skippedFiles } = filterEmbeddedDiff(raw);
    expect(skippedFiles).toEqual([]);
    expect(filtered).toBe(raw);
  });

  it("strips testdata/ files", () => {
    const raw = DIFF_HEADER("src/main.go") + DIFF_HEADER("testdata/fixtures/input.json") + DIFF_HEADER("pkg/testdata/results.md");
    const { filtered, skippedFiles } = filterEmbeddedDiff(raw);
    expect(skippedFiles).toEqual(["testdata/fixtures/input.json", "pkg/testdata/results.md"]);
    expect(filtered).toBe(DIFF_HEADER("src/main.go"));
  });

  it("strips common lockfiles", () => {
    const raw = DIFF_HEADER("package.json") + DIFF_HEADER("package-lock.json") + DIFF_HEADER("go.sum") + DIFF_HEADER("yarn.lock");
    const { filtered, skippedFiles } = filterEmbeddedDiff(raw);
    expect(skippedFiles).toEqual(["package-lock.json", "go.sum", "yarn.lock"]);
    expect(filtered).toBe(DIFF_HEADER("package.json"));
  });

  it("strips .md and .mdx files", () => {
    const raw = DIFF_HEADER("README.md") + DIFF_HEADER("docs/guide.mdx") + DIFF_HEADER("src/app.ts");
    const { filtered, skippedFiles } = filterEmbeddedDiff(raw);
    expect(skippedFiles).toEqual(["README.md", "docs/guide.mdx"]);
    expect(filtered).toBe(DIFF_HEADER("src/app.ts"));
  });

  it("handles empty diff", () => {
    const { filtered, skippedFiles } = filterEmbeddedDiff("");
    expect(skippedFiles).toEqual([]);
    expect(filtered).toBe("");
  });

  it("handles diff where all files are skipped", () => {
    const raw = DIFF_HEADER("go.sum") + DIFF_HEADER("testdata/case1.json");
    const { filtered, skippedFiles } = filterEmbeddedDiff(raw);
    expect(skippedFiles).toHaveLength(2);
    expect(filtered).toBe("");
  });
});
