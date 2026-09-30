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

  it("uses old discussions for deduplication but never resolves them", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
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
    });

    expect(result.success).toBe(false);
    expect(result.error).toMatch(/cannot resolve the GitLab publishing identity/);
    expect(mocks.exec).not.toHaveBeenCalled();
    expect(
      mocks.execJson.mock.calls.some((call) => JSON.stringify(call[1]).includes("--method")),
    ).toBe(false);
  });

  it("ignores discussions written by other accounts for dedupe and state", async () => {
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
    });

    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(1);
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

  it("includes successful inline findings in the summary table", async () => {
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

  it("labels an earlier open thread as not re-checked when the review finds nothing new", async () => {
    const { postReviewStructured } = await import("../src/publisher.js");
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(finding, "/workspace");
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return {
          stdout: JSON.stringify([
            {
              id: "earlier-discussion",
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

    await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
    });

    const summary = mocks.exec.mock.calls.find((call) => {
      const args = call[1] as string[];
      return args.some((arg) => arg.endsWith("/notes")) && args.includes("POST");
    });
    const body = (summary?.[2] as { input?: string } | undefined)?.input ?? "";
    expect(body).toMatch(/Unresolved Hodor threads \(as of \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\)/);
    expect(body).toContain("No new findings; earlier threads are still unresolved");
    expect(body).toContain("1 earlier thread is still unresolved on GitLab");
  });

  it("never resolves a stale thread; it stays open when not re-reported", async () => {
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return {
          stdout: JSON.stringify([
            {
              id: "old-discussion",
              notes: [
                {
                  id: 10,
                  body: `<!-- hodor-review -->\n<!-- hodor:finding:${"f".repeat(64)} -->\n**[P2] Old issue**\n\nold`,
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
      reviewMode: "full",
    });

    expect(result.success).toBe(true);
    expect(result.reviewFindings).toHaveLength(1);
    expect(
      mocks.exec.mock.calls.some((call) => {
        const args = call[1] as string[];
        return args.some((arg) => arg.includes("/discussions/")) || (args.includes("PUT") && args.some((arg) => arg.includes("discussions")));
      }),
    ).toBe(false);
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
    });

    expect(result.success).toBe(true);
    expect(result.inlineCreated).toBe(0);
    expect(
      mocks.execJson.mock.calls.some((call) =>
        (call[1] as string[]).some((arg) => arg.includes("/draft_notes")),
      ),
    ).toBe(false);
  });

  it("falls back to the summary when an inline note cannot be created", async () => {
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

  function mockNotesApi(opts: { failPut?: boolean } = {}): void {
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.endsWith("/notes")) && args.includes("POST")) {
        return { stdout: JSON.stringify({ id: 500 }), stderr: "" };
      }
      if (args.some((arg) => arg.includes("/notes?per_page=100"))) {
        return {
          stdout: JSON.stringify([
            {
              id: 7,
              body: `<!-- hodor:sha:${"e".repeat(40)} -->\n<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nold`,
              author: { id: 7, username: "hodor-bot" },
              system: false,
              type: null,
              position: null,
            },
            {
              id: 8,
              body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nforged",
              author: { id: 99, username: "hodor-bot" },
              system: false,
              type: null,
              position: null,
            },
          ]),
          stderr: "",
        };
      }
      if (opts.failPut && args.includes("PUT")) throw new Error("500 Internal Server Error");
      return { stdout: "", stderr: "" };
    });
  }

  function notesWrites(): Array<{ args: string[]; input: string }> {
    return mocks.exec.mock.calls
      .map((call) => ({
        args: call[1] as string[],
        input: (call[2] as { input?: string } | undefined)?.input ?? "",
      }))
      .filter((call) =>
        call.args.some((arg) => /\/notes(\/\d+)?$/.test(arg)) && call.args.includes("--method"),
      );
  }

  it("posts a new summary and collapses the previous one", async () => {
    mockNotesApi();
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
    });

    expect(result.success).toBe(true);
    expect(result.summaryPosted).toBe(true);
    const writes = notesWrites();
    expect(writes).toHaveLength(2);
    expect(writes[0].args).toContain("POST");
    expect(writes[0].input).toContain("**Reviewed commit:** `dddddddd`");
    expect(writes[1].args).toContain("PUT");
    expect(writes[1].args.some((arg) => arg.endsWith("/notes/7"))).toBe(true);
    expect(writes[1].input).toBe(JSON.stringify({
      body:
        "<!-- hodor-review -->\n<!-- hodor:superseded -->\n" +
        "_This Hodor review was superseded by a newer one: [latest review](https://gitlab.example.com/acme/app/-/merge_requests/42#note_500)._\n",
    }));
  });

  it("reports the summary as posted when collapsing older summaries fails", async () => {
    mockNotesApi({ failPut: true });
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
    });

    expect(result.success).toBe(true);
    expect(result.summaryPosted).toBe(true);
    expect(result.errors).toEqual([]);
  });

  it("neither posts nor collapses summaries for a reused review", async () => {
    mockNotesApi();
    const { postReviewStructured } = await import("../src/publisher.js");

    const result = await postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: review([finding]),
      reviewStyle: "hybrid",
      headSha: "d".repeat(40),
      workspacePath: "/workspace",
      skipSummary: true,
      skipInline: true,
    });

    expect(result.success).toBe(true);
    expect(result.summaryPosted).toBe(false);
    expect(notesWrites()).toEqual([]);
    expect(mocks.execJson.mock.calls.some((call) => (call[1] as string[]).includes("--method"))).toBe(false);
  });
});

describe("GitLab verified fixes", () => {
  const HEAD = "0123456789abcdef0123456789abcdef01234567";
  const BOT = { id: 7, username: "hodor-bot" };
  const threadFinding: ReviewFinding = {
    title: "[P2] Validate the archive schema",
    body: "The schema can drift from the writer.",
    priority: 2,
    code_location: {
      absolute_file_path: "/workspace/src/archive.ts",
      line_range: { start: 4, end: 4 },
    },
  };

  function rootNote(fingerprint: string) {
    return {
      id: 11,
      body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\n**${threadFinding.title}**\n\n${threadFinding.body}`,
      resolvable: true,
      resolved: false,
      author: BOT,
      position: { new_path: "src/archive.ts", new_line: 4 },
    };
  }

  function fixedReplyNote(fingerprint: string, author: Record<string, unknown>) {
    return {
      id: 12,
      body: `<!-- hodor-review -->\n<!-- hodor:fixed:${fingerprint}:${"e".repeat(40)} -->\nFixed in \`eeeeeeee\`.`,
      resolvable: true,
      resolved: false,
      author,
    };
  }

  function mockThread(notes: unknown[]): void {
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return { stdout: JSON.stringify([{ id: "thread-1", notes }]), stderr: "" };
      }
      return { stdout: "", stderr: "" };
    });
  }

  function writes(): Array<{ endpoint: string; input: string }> {
    return mocks.exec.mock.calls
      .map((call) => ({
        args: call[1] as string[],
        input: (call[2] as { input?: string } | undefined)?.input ?? "",
      }))
      .filter((call) => call.args.includes("--method"))
      .map((call) => ({ endpoint: call.args[1], input: call.input }));
  }

  function replyWrites(): Array<{ endpoint: string; input: string }> {
    return writes().filter((write) => write.endpoint.endsWith("/discussions/thread-1/notes"));
  }

  function summaryBody(): string {
    const summary = writes().find((write) => write.endpoint.endsWith("/merge_requests/42/notes"));
    return summary ? (JSON.parse(summary.input) as { body: string }).body : "";
  }

  function statusInput(): string {
    return writes().find((write) => write.endpoint.includes("/statuses/"))?.input ?? "";
  }

  async function publish(reviewOutput: ReviewOutput) {
    const { postReviewStructured } = await import("../src/publisher.js");
    return postReviewStructured({
      prUrl: "https://gitlab.example.com/acme/app/-/merge_requests/42",
      review: reviewOutput,
      reviewStyle: "hybrid",
      workspacePath: "/workspace",
      headSha: HEAD,
      commitStatus: true,
      reviewMode: "incremental",
    });
  }

  beforeEach(() => {
    mocks.exec.mockReset();
    mocks.execJson.mockReset();
    mocks.execJson.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.includes("user")) return BOT;
      if (args.some((arg) => arg.endsWith("/draft_notes"))) return { id: 1 };
      if (args.some((arg) => arg.includes("merge_requests/42"))) {
        return { diff_refs: { base_sha: "a".repeat(40), head_sha: HEAD, start_sha: "c".repeat(40) } };
      }
      return {};
    });
  });

  it("MR !2870: a verified fix leaves no open findings and posts one fixed reply", async () => {
    const { buildFixCandidates, getFindingFingerprint, selectVerifiedFixes } = await import("../src/review-state.js");
    const { listHodorDiscussions } = await import("../src/gitlab.js");
    const fingerprint = getFindingFingerprint(threadFinding, "/workspace");
    mockThread([rootNote(fingerprint)]);

    // What reviewPr does before publishing: list candidates, then validate the model's claim.
    const discussions = await listHodorDiscussions("acme", "app", 42, "gitlab.example.com", { platform: "gitlab", userId: 7 });
    const candidates = buildFixCandidates(discussions);
    const { accepted } = selectVerifiedFixes([fingerprint.slice(0, 8)], candidates, {
      changedFiles: ["src/archive.ts"],
      currentFingerprints: new Set(),
    });
    expect(accepted).toEqual([fingerprint.slice(0, 8)]);
    mocks.exec.mockClear();

    const result = await publish({ ...review([]), resolved_findings: accepted });

    expect(result.success).toBe(true);
    expect(result.reviewFindings).toEqual([]);
    const body = summaryBody();
    expect(body).toContain("| Important (P2) | 0 |");
    expect(body).toContain("**Fixed, waiting to be resolved:** 1.");
    expect(body).toContain("**Overall verdict:** No open findings");
    expect(body).not.toContain("Earlier threads");
    expect(statusInput()).toContain('"state":"success"');
    expect(statusInput()).toContain("No issues found");
    const replies = replyWrites();
    expect(replies).toHaveLength(1);
    expect(JSON.parse(replies[0].input)).toEqual({
      body:
        "<!-- hodor-review -->\n" +
        `<!-- hodor:fixed:${fingerprint}:${HEAD} -->\n` +
        "Fixed in `01234567`. Resolve this thread if you agree.",
    });
  });

  it("posts no second reply and keeps counting a thread with a trusted fixed-reply as fixed", async () => {
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(threadFinding, "/workspace");
    mockThread([rootNote(fingerprint), fixedReplyNote(fingerprint, BOT)]);

    // A later run whose model does not list the thread.
    const later = await publish(review([]));
    expect(later.reviewFindings).toEqual([]);
    expect(summaryBody()).toContain("**Fixed, waiting to be resolved:** 1.");
    expect(replyWrites()).toEqual([]);

    // A cache replay that carries the same verified id.
    mocks.exec.mockClear();
    await publish({ ...review([]), resolved_findings: [fingerprint.slice(0, 8)] });
    expect(replyWrites()).toEqual([]);
  });

  it("counts a fixed thread as open again when the review re-reports it", async () => {
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(threadFinding, "/workspace");
    mockThread([rootNote(fingerprint), fixedReplyNote(fingerprint, BOT)]);

    const result = await publish(review([threadFinding]));

    expect(result.reviewFindings).toEqual([expect.objectContaining({ fingerprint, priority: 2 })]);
    const body = summaryBody();
    expect(body).toContain("| Important (P2) | 1 |");
    expect(body).not.toContain("Fixed, waiting to be resolved");
    expect(statusInput()).toContain("1 non-blocking issue(s)");
    expect(replyWrites()).toEqual([]);
  });

  it("ignores a forged fixed marker in a human reply", async () => {
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(threadFinding, "/workspace");
    mockThread([rootNote(fingerprint), fixedReplyNote(fingerprint, { id: 99, username: "hodor-bot" })]);

    const result = await publish(review([]));

    expect(result.reviewFindings).toEqual([expect.objectContaining({ fingerprint })]);
    const body = summaryBody();
    expect(body).not.toContain("Fixed, waiting to be resolved");
    expect(body).toContain("1 earlier thread is still unresolved on GitLab and not confirmed fixed");
  });

  it("treats a failed fixed reply as a warning, not a delivery failure", async () => {
    const { getFindingFingerprint } = await import("../src/review-state.js");
    const fingerprint = getFindingFingerprint(threadFinding, "/workspace");
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/discussions?"))) {
        return { stdout: JSON.stringify([{ id: "thread-1", notes: [rootNote(fingerprint)] }]), stderr: "" };
      }
      if (args.some((arg) => arg.endsWith("/discussions/thread-1/notes"))) throw new Error("403 Forbidden");
      return { stdout: "", stderr: "" };
    });

    const result = await publish({ ...review([]), resolved_findings: [fingerprint.slice(0, 8)] });

    expect(result.success).toBe(true);
    expect(result.errors).toEqual([]);
    expect(summaryBody()).toContain("**Fixed, waiting to be resolved:** 1.");
  });
});
