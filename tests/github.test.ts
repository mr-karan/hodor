import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchGithubPrMetadata, fetchGithubPrNotes, normalizeGithubMetadata } from "../src/github.js";

const mocks = vi.hoisted(() => ({
  exec: vi.fn(),
  execJson: vi.fn(),
}));

vi.mock("../src/utils/exec.js", () => ({
  exec: mocks.exec,
  execJson: mocks.execJson,
}));

describe("normalizeGithubMetadata", () => {
  it("maps PR fields from gh pr view", () => {
    const metadata = normalizeGithubMetadata({
      title: "Update handler",
      headRefName: "feature",
      baseRefName: "main",
      author: { login: "alice" },
    });

    expect(metadata).toEqual(expect.objectContaining({
      title: "Update handler",
      source_branch: "feature",
      target_branch: "main",
      author: { username: "alice", name: undefined },
    }));
    expect(metadata.Notes).toBeUndefined();
  });
});

describe("fetchGithubPrNotes", () => {
  beforeEach(() => {
    mocks.exec.mockReset();
    mocks.execJson.mockReset();
  });

  it("reads comments and reviews from paginated REST output with numeric ids", async () => {
    mocks.exec.mockImplementation(async (_cmd: string, args: string[]) => {
      if (args.some((arg) => arg.includes("/issues/5/comments"))) {
        return {
          stdout:
            JSON.stringify([{
              body: "Human reviewer feedback with enough context",
              user: { id: 1, login: "alice", type: "User" },
              created_at: "2026-07-15T00:00:00Z",
              updated_at: "2026-07-15T01:00:00Z",
            }]) +
            JSON.stringify([{
              body: "Second page comment ][ with brackets",
              user: { id: 2, login: "bob", type: "User" },
              created_at: "2026-07-15T02:00:00Z",
            }]),
          stderr: "",
        };
      }
      if (args.some((arg) => arg.includes("/pulls/5/reviews"))) {
        return {
          stdout: JSON.stringify([{
            body: `<!-- hodor:sha:${"1".repeat(40)} -->\n<!-- hodor-review -->\nNo issues found.`,
            user: { id: 41898282, login: "github-actions[bot]", type: "Bot" },
            submitted_at: "2026-07-16T00:00:00Z",
          }]),
          stderr: "",
        };
      }
      throw new Error(`unexpected call ${args.join(" ")}`);
    });

    const notes = await fetchGithubPrNotes("octo", "widget", 5, "github.example.com");

    expect(notes.map((note) => note.author)).toEqual([
      { id: 1, username: "alice" },
      { id: 2, username: "bob" },
      { id: 41898282, username: "github-actions[bot]" },
    ]);
    expect(notes[1].body).toContain("][");
    expect(notes[2].created_at).toBe("2026-07-16T00:00:00Z");
    expect(mocks.exec.mock.calls[0]?.[1]).toEqual([
      "api",
      "--hostname",
      "github.example.com",
      "repos/octo/widget/issues/5/comments?per_page=100",
      "--paginate",
    ]);
  });

  it("keeps PR metadata when the notes cannot be fetched", async () => {
    mocks.execJson.mockResolvedValueOnce({ title: "Update handler", baseRefName: "main" });
    mocks.exec.mockRejectedValue(new Error("HTTP 502"));

    const metadata = await fetchGithubPrMetadata("octo", "widget", 5, "github.com");

    expect(metadata.title).toBe("Update handler");
    expect(metadata.Notes).toBeUndefined();
  });
});
