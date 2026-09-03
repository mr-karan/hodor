import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReviewFinding, ReviewOutput } from "../src/types.js";

const mocks = vi.hoisted(() => ({
  exec: vi.fn(),
  execJson: vi.fn(),
}));

vi.mock("../src/utils/exec.js", () => ({
  exec: mocks.exec,
  execJson: mocks.execJson,
}));

const finding: ReviewFinding = {
  title: "[P1] Preserve authorization",
  body: "The new path skips the ownership check.",
  priority: 1,
  code_location: {
    absolute_file_path: "/workspace/src/app.ts",
    line_range: { start: 12, end: 12 },
  },
};

function review(findings: ReviewFinding[]): ReviewOutput {
  return {
    findings,
    overall_correctness: findings.length > 0 ? "patch is incorrect" : "patch is correct",
    overall_explanation: findings.length > 0 ? "A blocking issue remains." : "No issues remain.",
  };
}

describe("GitLab review publication", () => {
  beforeEach(() => {
    mocks.exec.mockReset();
    mocks.execJson.mockReset();
    mocks.exec.mockResolvedValue({ stdout: "", stderr: "" });
    mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("user")) return { username: "hodor-bot" };
      if (args.some((arg) => arg.includes("merge_requests/42")) && !args.includes("--method")) {
        return {
          diff_refs: {
            base_sha: "a".repeat(40),
            head_sha: "b".repeat(40),
            start_sha: "c".repeat(40),
          },
        };
      }
      return {};
    });
  });

  it("uses old discussions for deduplication but never resolves them incrementally", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
      reconcileDiscussions: false,
    });

    expect(result.success).toBe(true);
    expect(
      mocks.exec.mock.calls.some((call) => {
        const args = call[1] as string[];
        return args.some((arg) => arg.includes("/discussions?"));
      }),
    ).toBe(true);
    expect(
      mocks.exec.mock.calls.some((call) =>
        (call[1] as string[]).some((arg) => arg.includes("/discussions/")),
      ),
    ).toBe(false);
  });

  it("embeds the cache marker in summaries and skips duplicate summaries on reuse", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");
    const cacheMarker = "<!-- hodor:cache:v1:abc123 -->";

    await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
      cacheMarker,
    });
    const summaryCall = mocks.exec.mock.calls.find((call) => {
      const args = call[1] as string[];
      return args.some((arg) => arg.endsWith("/notes")) && args.includes("POST");
    });
    expect(summaryCall?.[2]).toEqual(
      expect.objectContaining({ input: expect.stringContaining(cacheMarker) }),
    );

    mocks.exec.mockClear();
    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
      skipSummary: true,
    });

    expect(result.success).toBe(true);
    expect(mocks.exec.mock.calls.some((call) => {
      const args = call[1] as string[];
      return args.some((arg) => arg.endsWith("/notes")) && args.includes("--method");
    })).toBe(false);
  });

  it("keeps successful inline findings out of the compact rolling summary", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");
    await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      model: "openai/gpt-5",
      metricsFooter: "**Review Metrics** · 3 turns\n- Cost: `$0.1000`",
    });

    const summaryCall = mocks.exec.mock.calls.find((call) => {
      const args = call[1] as string[];
      return args.some((arg) => arg.endsWith("/notes")) && args.includes("POST");
    });
    expect(summaryCall?.[2]).toEqual(
      expect.objectContaining({
        input: expect.stringMatching(/<details>[\s\S]*Review Metrics[\s\S]*<\/details>/),
      }),
    );
    const summaryInput = summaryCall?.[2];
    if (!summaryInput || typeof summaryInput !== "object" || !("input" in summaryInput)) {
      throw new Error("summary input was not captured");
    }
    expect(summaryInput.input).not.toContain(finding.title);
  });

  it("fails the commit status while an earlier blocking discussion remains open", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(finding, "/workspace");
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return {
          stdout: JSON.stringify([
            {
              id: "blocking-discussion",
              notes: [
                {
                  id: 11,
                  body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\n**${finding.title}**\n\n${finding.body}`,
                  resolvable: true,
                  resolved: false,
                  position: { new_path: "src/app.ts", new_line: 12 },
                },
              ],
            },
          ]),
          stderr: "",
        };
      }
      return { stdout: "", stderr: "" };
    });

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      commitStatus: true,
    });

    expect(result.success).toBe(true);
    expect(result.reviewFindings).toHaveLength(1);
    const statusCall = mocks.exec.mock.calls.find((call) =>
      (call[1] as string[]).some((arg) => arg.includes("/statuses/")),
    );
    expect(statusCall?.[2]).toEqual(
      expect.objectContaining({
        input: expect.stringContaining(
          '"state":"failed","name":"hodor","description":"1 blocking issue(s) found"',
        ),
      }),
    );
  });

  it("reconciles stale fingerprinted discussions only after posting a full review", async () => {
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return {
          stdout: JSON.stringify([
            {
              id: "old-discussion",
              notes: [
                {
                  id: 10,
                  body: `<!-- hodor-review -->\n<!-- hodor:finding:${"f".repeat(64)} -->\nold`,
                  resolvable: true,
                  resolved: false,
                },
              ],
            },
          ]),
          stderr: "",
        };
      }
      return { stdout: "", stderr: "" };
    });
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
      reconcileDiscussions: true,
    });

    expect(result.success).toBe(true);
    expect(result.reconciledDiscussions).toBe(1);
    const calls = mocks.exec.mock.calls.map((call) => (call[1] as string[]).join(" "));
    const summaryIndex = calls.findIndex((call) => call.includes("/notes --method POST"));
    const resolveIndex = calls.findIndex((call) =>
      call.includes("/discussions/old-discussion --method PUT"),
    );
    expect(summaryIndex).toBeGreaterThanOrEqual(0);
    expect(resolveIndex).toBeGreaterThan(summaryIndex);
  });

  it("keeps a matching open finding and does not publish a duplicate thread", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(finding, "/workspace");
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return {
          stdout: JSON.stringify([
            {
              id: "matching-discussion",
              notes: [
                {
                  id: 11,
                  body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\nopen`,
                  resolvable: true,
                  resolved: false,
                },
              ],
            },
          ]),
          stderr: "",
        };
      }
      return { stdout: "", stderr: "" };
    });

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      reconcileDiscussions: true,
    });

    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(0);
    expect(result.reconciledDiscussions).toBe(0);
    expect(
      mocks.execJson.mock.calls.some((call) =>
        (call[1] as string[]).some((arg) => arg.includes("/draft_notes")),
      ),
    ).toBe(false);
  });

  it("falls back to the rolling summary when an inline note cannot be created", async () => {
    mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("user")) return { username: "hodor-bot" };
      if (args.some((arg) => arg.includes("merge_requests/42")) && !args.includes("--method")) {
        return {
          diff_refs: {
            base_sha: "a".repeat(40),
            head_sha: "b".repeat(40),
            start_sha: "c".repeat(40),
          },
        };
      }
      if (args.some((arg) => arg.endsWith("/draft_notes"))) {
        throw new Error("position is not on the current diff");
      }
      return {};
    });
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
    });

    expect(result.success).toBe(true);
    expect(result.inlineFailed).toBe(1);
    expect(result.summaryPosted).toBe(true);
    const summaryCall = mocks.exec.mock.calls.find((call) => {
      const args = call[1] as string[];
      return args.some((arg) => arg.endsWith("/notes")) && args.includes("POST");
    });
    expect(summaryCall?.[2]).toEqual(
      expect.objectContaining({ input: expect.stringContaining(finding.title) }),
    );
  });

  it("publishes drafts individually when GitLab bulk publishing fails", async () => {
    mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("user")) return { username: "hodor-bot" };
      if (args.some((arg) => arg.includes("merge_requests/42")) && !args.includes("--method")) {
        return {
          diff_refs: {
            base_sha: "a".repeat(40),
            head_sha: "b".repeat(40),
            start_sha: "c".repeat(40),
          },
        };
      }
      if (args.some((arg) => arg.endsWith("/draft_notes"))) return { id: 123 };
      return {};
    });
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.endsWith("/draft_notes/bulk_publish"))) {
        throw new Error("500 Internal Server Error");
      }
      return { stdout: "", stderr: "" };
    });
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
    });

    expect(result.success).toBe(true);
    expect(result.draftsPublished).toBe(true);
    expect(
      mocks.exec.mock.calls.some((call) =>
        (call[1] as string[]).some((arg) => arg.endsWith("/draft_notes/123/publish")),
      ),
    ).toBe(true);
  });
});
