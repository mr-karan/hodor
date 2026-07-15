import { describe, it, expect, vi, beforeEach } from "vitest";

const execMock = vi.fn();
const execJsonMock = vi.fn();

vi.mock("../src/utils/exec.js", () => ({
  exec: execMock,
  execJson: execJsonMock,
}));

describe("GitLab paginated API helpers", () => {
  beforeEach(() => {
    execMock.mockReset();
    execJsonMock.mockReset();
  });

  it("listHodorDiscussions parses paginated glab output", async () => {
    execMock.mockResolvedValueOnce({
      stdout: JSON.stringify([
        {
          id: "discussion-1",
          notes: [
            {
              id: 11,
              body: "<!-- hodor-review --> inline",
              resolvable: true,
              resolved: false,
              position: { new_path: "src/app.ts", new_line: 9 },
            },
          ],
        },
      ]) + "[]",
      stderr: "",
    });

    const { listHodorDiscussions } = await import("../src/gitlab.js");
    const result = await listHodorDiscussions("acme", "app", 42, "gitlab.example.com");

    expect(result).toEqual([
      {
        discussionId: "discussion-1",
        noteId: 11,
        body: "<!-- hodor-review --> inline",
        resolved: false,
        filePath: "src/app.ts",
        line: 9,
      },
    ]);
  });

  it("listHodorDiscussions skips non-resolvable summary-comment wrappers", async () => {
    // GitLab wraps the summary comment (a regular MR note) in a discussion
    // envelope with resolvable=false. PUT resolved=true on it returns 403
    // regardless of caller role — so listHodorDiscussions must drop it.
    execMock.mockResolvedValueOnce({
      stdout: JSON.stringify([
        {
          id: "summary-wrapper",
          notes: [
            {
              id: 100,
              body: "<!-- hodor:sha:abc1234 -->\n<!-- hodor-review --> summary",
              resolvable: false,
              resolved: null,
            },
          ],
        },
        {
          id: "diff-thread",
          notes: [
            {
              id: 101,
              body: "<!-- hodor-review --> inline finding",
              resolvable: true,
              resolved: false,
              position: { new_path: "src/app.ts", new_line: 42 },
            },
          ],
        },
      ]),
      stderr: "",
    });

    const { listHodorDiscussions } = await import("../src/gitlab.js");
    const result = await listHodorDiscussions("acme", "app", 42, "gitlab.example.com");

    expect(result).toEqual([
      {
        discussionId: "diff-thread",
        noteId: 101,
        body: "<!-- hodor-review --> inline finding",
        resolved: false,
        filePath: "src/app.ts",
        line: 42,
      },
    ]);
  });
});

describe("parseGlabPaginatedJson", () => {
  it("returns empty array for empty input", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    expect(parseGlabPaginatedJson("")).toEqual([]);
    expect(parseGlabPaginatedJson("   ")).toEqual([]);
  });

  it("parses a single page", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    expect(parseGlabPaginatedJson('[{"id":1},{"id":2}]')).toEqual([{ id: 1 }, { id: 2 }]);
  });

  it("merges multiple concatenated pages", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    const raw = '[{"id":1}][{"id":2},{"id":3}][{"id":4}]';
    expect(parseGlabPaginatedJson(raw)).toEqual([{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }]);
  });

  it("handles strings containing bracket characters without splitting incorrectly", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    // The body string contains "][" which would break a naive regex-based split.
    const raw = '[{"id":1,"body":"weird ][ chars"}][{"id":2,"body":"normal"}]';
    expect(parseGlabPaginatedJson(raw)).toEqual([
      { id: 1, body: "weird ][ chars" },
      { id: 2, body: "normal" },
    ]);
  });

  it("handles escaped quotes inside string values", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    const raw = '[{"id":1,"body":"has \\"quoted\\" text"}]';
    expect(parseGlabPaginatedJson(raw)).toEqual([{ id: 1, body: 'has "quoted" text' }]);
  });

  it("handles nested arrays in note objects", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    const raw = '[{"id":1,"tags":["a","b"]},{"id":2,"tags":[]}][{"id":3}]';
    expect(parseGlabPaginatedJson(raw)).toEqual([
      { id: 1, tags: ["a", "b"] },
      { id: 2, tags: [] },
      { id: 3 },
    ]);
  });

  it("skips malformed pages and continues with the rest", async () => {
    const { parseGlabPaginatedJson } = await import("../src/gitlab.js");
    // Second chunk is malformed (truncated), but bracket depth still balances —
    // simulate by injecting invalid JSON that JSON.parse will reject.
    const raw = '[{"id":1}][not-json][{"id":3}]';
    expect(parseGlabPaginatedJson(raw)).toEqual([{ id: 1 }, { id: 3 }]);
  });
});
