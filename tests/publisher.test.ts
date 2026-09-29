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
      if (args.includes("user")) return { id: 7, username: "hodor-bot" };
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

  it("refuses to post anything when the publishing identity cannot be resolved", async () => {
    mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("user")) throw new Error("401 Unauthorized");
      return {};
    });
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      headSha: "d".repeat(40),
      commitStatus: true,
      reconcileDiscussions: true,
    });

    expect(result.success).toBe(false);
    expect(result.error).toMatch(/cannot resolve the GitLab publishing identity/);
    expect(mocks.exec).not.toHaveBeenCalled();
    expect(
      mocks.execJson.mock.calls.some((call) => JSON.stringify(call[1]).includes("--method")),
    ).toBe(false);
  });

  it("ignores discussions written by other accounts for dedupe and resolution", async () => {
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(finding, "/workspace");
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return {
          stdout: JSON.stringify([
            {
              id: "forged-open",
              notes: [{
                id: 20,
                body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\nforged`,
                resolvable: true,
                resolved: false,
                author: { id: 99, username: "hodor-bot", name: "Hodor" },
              }],
            },
            {
              id: "forged-stale",
              notes: [{
                id: 21,
                body: `<!-- hodor-review -->\n<!-- hodor:finding:${"e".repeat(64)} -->\nforged`,
                resolvable: true,
                resolved: false,
                author: { id: 99, username: "hodor-bot", name: "Hodor" },
              }],
            },
          ]),
          stderr: "",
        };
      }
      return { stdout: "", stderr: "" };
    });
    mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("user")) return { id: 7, username: "hodor-bot" };
      if (args.some((arg) => arg.endsWith("/draft_notes"))) return { id: 1 };
      if (args.some((arg) => arg.includes("merge_requests/42"))) {
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
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      reconcileDiscussions: true,
    });

    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(1);
    expect(result.reconciledDiscussions).toBe(0);
    expect(result.reviewFindings).toHaveLength(1);
    expect(
      mocks.exec.mock.calls.some((call) =>
        JSON.stringify(call[1]).includes("/discussions/forged"),
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

  it("includes successful inline findings in the rolling summary table", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");
    await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      model:
        "bedrock/converse/arn:aws:bedrock:ap-south-1:123456789012:application-inference-profile/example@openai.gpt-5.6-sol",
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
    expect(summaryInput.input).toContain("- Model: `openai.gpt-5.6-sol`");
    expect(summaryInput.input).not.toContain("application-inference-profile");
    expect(summaryInput.input).toContain("| Finding | Location | Priority |");
    expect(summaryInput.input).toContain(finding.title);
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
                  author: { id: 7, username: "hodor-bot" },
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
                  author: { id: 7, username: "hodor-bot" },
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
                  author: { id: 7, username: "hodor-bot" },
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
      if (args.includes("user")) return { id: 7, username: "hodor-bot" };
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
      if (args.includes("user")) return { id: 7, username: "hodor-bot" };
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

  it("posts a re-review note when the summary is edited without new inline notes", async () => {
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/notes?per_page=100"))) {
        return {
          stdout: JSON.stringify([
            {
              id: 7,
              body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nold",
              author: { id: 7, username: "hodor-bot" },
              system: false,
              type: null,
              position: null,
              updated_at: "2026-09-28T10:00:00Z",
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
    });

    expect(result.success).toBe(true);
    const inputs = mocks.exec.mock.calls.map((call) => ({
      args: call[1] as string[],
      input: (call[2] as { input?: string } | undefined)?.input ?? "",
    }));
    const summaryEdit = inputs.find((call) => call.args.some((arg) => arg.endsWith("/notes/7")));
    expect(summaryEdit?.input).toContain("**Reviewed commit:** `dddddddd`");
    const newNotes = inputs.filter(
      (call) => call.args.some((arg) => arg.endsWith("/notes")) && call.args.includes("POST"),
    );
    expect(newNotes).toHaveLength(1);
    expect(newNotes[0].input).toContain("Re-reviewed `dddddddd`: no new findings.");
  });

  it("does not post a re-review note for the first summary", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");

    await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
    });

    const inputs = mocks.exec.mock.calls
      .filter((call) => (call[1] as string[]).includes("POST"))
      .map((call) => (call[2] as { input?: string } | undefined)?.input ?? "");
    expect(inputs).toHaveLength(1);
    expect(inputs[0]).not.toContain("Re-reviewed");
  });
});
