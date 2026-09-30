import { describe, it, expect, vi, beforeEach } from "vitest";
import type { GitlabPublisherIdentity } from "../src/types.js";

const execMock = vi.fn();
const execJsonMock = vi.fn();

vi.mock("../src/utils/exec.js", () => ({
  exec: execMock,
  execJson: execJsonMock,
}));

const BOT: GitlabPublisherIdentity = { platform: "gitlab", userId: 7 };
const BOT_AUTHOR = { id: 7, username: "hodor-bot", name: "Hodor" };

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
              author: BOT_AUTHOR,
              resolved: false,
              position: { new_path: "src/app.ts", new_line: 9 },
            },
          ],
        },
      ]) + "[]",
      stderr: "",
    });

    const { listHodorDiscussions } = await import("../src/gitlab.js");
    const result = await listHodorDiscussions("acme", "app", 42, "gitlab.example.com", BOT);

    expect(result).toEqual([
      {
        discussionId: "discussion-1",
        noteId: 11,
        body: "<!-- hodor-review --> inline",
        resolved: false,
        filePath: "src/app.ts",
        line: 9,
        humanReplies: [],
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
              author: BOT_AUTHOR,
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
              author: BOT_AUTHOR,
              resolved: false,
              position: { new_path: "src/app.ts", new_line: 42 },
            },
          ],
        },
      ]),
      stderr: "",
    });

    const { listHodorDiscussions } = await import("../src/gitlab.js");
    const result = await listHodorDiscussions("acme", "app", 42, "gitlab.example.com", BOT);

    expect(result).toEqual([
      {
        discussionId: "diff-thread",
        noteId: 101,
        body: "<!-- hodor-review --> inline finding",
        resolved: false,
        filePath: "src/app.ts",
        line: 42,
        humanReplies: [],
      },
    ]);
  });
});

describe("listHodorDiscussions provenance", () => {
  beforeEach(() => {
    execMock.mockReset();
    execJsonMock.mockReset();
  });

  it("drops forged, anonymous, and system discussion notes before fingerprinting", async () => {
    const fingerprint = "f".repeat(64);
    const body = `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\nfinding`;
    const note = (id: number, extra: Record<string, unknown>) => ({
      id,
      body,
      resolvable: true,
      resolved: false,
      ...extra,
    });
    execMock.mockResolvedValueOnce({
      stdout: JSON.stringify([
        { id: "bot", notes: [note(1, { author: BOT_AUTHOR })] },
        { id: "same-name", notes: [note(2, { author: { id: 99, username: "hodor-bot", name: "Hodor" } })] },
        { id: "string-id", notes: [note(3, { author: { id: "7", username: "hodor-bot" } })] },
        { id: "no-author", notes: [note(4, {})] },
        { id: "no-id", notes: [note(5, { author: { username: "hodor-bot" } })] },
        { id: "system", notes: [note(6, { author: BOT_AUTHOR, system: true })] },
        { id: "human", notes: [note(7, { author: { id: 8, username: "alice" } })] },
      ]),
      stderr: "",
    });

    const { listHodorDiscussions } = await import("../src/gitlab.js");
    const result = await listHodorDiscussions("acme", "app", 42, "gitlab.example.com", BOT);

    expect(result.map((discussion) => discussion.discussionId)).toEqual(["bot"]);
  });
});

describe("listHodorDiscussions thread replies", () => {
  const fingerprint = "a".repeat(64);
  const sha = "b".repeat(40);
  const root = {
    id: 1,
    body: `<!-- hodor-review -->\n<!-- hodor:finding:${fingerprint} -->\n**[P2] Keep the schema in sync**\n\nBody.`,
    resolvable: true,
    resolved: false,
    author: BOT_AUTHOR,
    created_at: "2026-09-01T10:00:00Z",
    position: { new_path: "src/app.ts", new_line: 3 },
  };
  const fixedReply = (id: number, author: Record<string, unknown>, replySha = sha) => ({
    id,
    body: `<!-- hodor-review -->\n<!-- hodor:fixed:${fingerprint}:${replySha} -->\nFixed in \`${replySha.slice(0, 8)}\`.`,
    resolvable: true,
    resolved: false,
    author,
    created_at: "2026-09-02T10:00:00Z",
  });

  beforeEach(() => {
    execMock.mockReset();
    execJsonMock.mockReset();
  });

  async function list(discussions: unknown[]) {
    execMock.mockResolvedValueOnce({ stdout: JSON.stringify(discussions), stderr: "" });
    const { listHodorDiscussions } = await import("../src/gitlab.js");
    return listHodorDiscussions("acme", "app", 42, "gitlab.example.com", BOT);
  }

  it("reads fixedAtSha only from the publisher's own reply in the same thread", async () => {
    const result = await list([
      { id: "fixed", notes: [root, fixedReply(2, BOT_AUTHOR)] },
    ]);

    expect(result).toHaveLength(1);
    expect(result[0]).toMatchObject({ discussionId: "fixed", noteId: 1, fixedAtSha: sha, humanReplies: [] });
  });

  it("keeps a forged fixed marker in a human reply as plain text", async () => {
    const forged = fixedReply(2, { id: 99, username: "hodor-bot", name: "Hodor" });
    const result = await list([{ id: "forged", notes: [root, forged] }]);

    expect(result).toHaveLength(1);
    expect(result[0].fixedAtSha).toBeUndefined();
    expect(result[0].humanReplies).toEqual([{ noteId: 2, author: "hodor-bot", body: forged.body }]);
  });

  it("does not let a fixed-reply in another thread mark this one", async () => {
    const result = await list([
      { id: "target", notes: [root] },
      { id: "other", notes: [{ ...root, id: 3, body: "<!-- hodor-review --> other" }, fixedReply(4, BOT_AUTHOR)] },
    ]);

    expect(result.find((discussion) => discussion.discussionId === "target")?.fixedAtSha).toBeUndefined();
  });

  it("attaches human replies, resolver, and latest activity to their finding thread", async () => {
    const result = await list([
      {
        id: "resolved",
        notes: [
          { ...root, resolved: true, resolved_by: { id: 8, username: "alice" }, resolved_at: "2026-09-05T10:00:00Z" },
          { id: 2, body: "false positive", author: { id: 8, username: "alice" }, created_at: "2026-09-04T10:00:00Z" },
          { id: 3, body: "changed the label", author: BOT_AUTHOR, system: true },
        ],
      },
      {
        id: "other",
        notes: [{ ...root, id: 4, body: `<!-- hodor-review -->\n<!-- hodor:finding:${"c".repeat(64)} -->\n**[P3] Other**` }],
      },
    ]);

    expect(result).toEqual([
      expect.objectContaining({
        discussionId: "resolved",
        resolved: true,
        resolvedBy: "alice",
        updatedAt: "2026-09-05T10:00:00Z",
        humanReplies: [{ noteId: 2, author: "alice", body: "false positive" }],
      }),
      expect.objectContaining({ discussionId: "other", humanReplies: [] }),
    ]);
  });
});

describe("replyToGitlabDiscussion", () => {
  beforeEach(() => {
    execMock.mockReset();
    execMock.mockResolvedValue({ stdout: "{}", stderr: "" });
  });

  it("posts a note to the discussion notes endpoint", async () => {
    const { replyToGitlabDiscussion } = await import("../src/gitlab.js");
    await replyToGitlabDiscussion("acme", "app", 42, "abc123", "hello", "gitlab.example.com");

    expect(execMock).toHaveBeenCalledWith(
      "glab",
      [
        "api",
        "projects/acme%2Fapp/merge_requests/42/discussions/abc123/notes",
        "--method",
        "POST",
        "-H",
        "Content-Type: application/json",
        "--input",
        "-",
      ],
      expect.objectContaining({ input: JSON.stringify({ body: "hello" }) }),
    );
  });
});

describe("fetchGitlabPublisherIdentity", () => {
  beforeEach(() => {
    execJsonMock.mockReset();
  });

  it("returns the numeric id of the authenticated user", async () => {
    execJsonMock.mockResolvedValueOnce({ id: 7, username: "hodor-bot" });
    const { fetchGitlabPublisherIdentity } = await import("../src/gitlab.js");

    await expect(fetchGitlabPublisherIdentity("gitlab.example.com")).resolves.toEqual(BOT);
    expect(execJsonMock.mock.calls[0]?.[1]).toEqual(["api", "user"]);
    expect(execJsonMock.mock.calls[0]?.[2]).toEqual(
      expect.objectContaining({ env: expect.objectContaining({ GITLAB_HOST: "gitlab.example.com" }) }),
    );
  });

  it("rejects a user without a numeric id", async () => {
    execJsonMock.mockResolvedValueOnce({ username: "hodor-bot" });
    const { fetchGitlabPublisherIdentity } = await import("../src/gitlab.js");

    await expect(fetchGitlabPublisherIdentity("gitlab.example.com")).rejects.toThrow(/numeric id/);
  });
});

describe("publishGitlabMrSummary", () => {
  const NEW_NOTE_ID = 500;
  const COLLAPSED_BODY =
    "<!-- hodor-review -->\n<!-- hodor:superseded -->\n" +
    `_This Hodor review was superseded by a newer one: [latest review](https://gitlab.example.com/acme/app/-/merge_requests/42#note_${NEW_NOTE_ID})._\n`;
  const SUPERSEDED_BODY =
    "<!-- hodor-review -->\n<!-- hodor:superseded -->\n_This Hodor review was superseded by a newer one: [latest review](https://gitlab.example.com/acme/app/-/merge_requests/42#note_400)._\n";

  const existingNotes = [
    { id: 10, type: null, body: "<!-- hodor:summary:v1 -->\nhuman note", author: { id: 8, username: "alice" } },
    {
      id: 11,
      type: null,
      body: `<!-- hodor:sha:${"a".repeat(40)} -->\n<!-- hodor-review -->\nlegacy`,
      author: BOT_AUTHOR,
    },
    { id: 12, type: null, body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nprevious", author: BOT_AUTHOR },
    {
      id: 13,
      type: null,
      body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nsame username, different account",
      author: { id: 99, username: "hodor-bot", name: "Hodor" },
    },
    { id: 14, type: null, body: SUPERSEDED_BODY, author: BOT_AUTHOR },
    {
      id: 15,
      type: "DiffNote",
      body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\ninline",
      author: BOT_AUTHOR,
      position: { new_path: "src/app.ts", new_line: 3 },
    },
    { id: 16, type: null, system: true, body: "<!-- hodor:summary:v1 -->\nsystem", author: BOT_AUTHOR },
    { id: 17, type: null, body: "Plain bot comment", author: BOT_AUTHOR },
    { id: NEW_NOTE_ID, type: null, body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nnew", author: BOT_AUTHOR },
  ];

  function mockGitlab(opts: { failPutFor?: number; failList?: boolean; failPost?: boolean } = {}): void {
    execMock.mockImplementation(async (_command: string, args: string[]) => {
      const endpoint = args[1] ?? "";
      if (args.includes("POST")) {
        if (opts.failPost) throw new Error("403 Forbidden");
        return { stdout: JSON.stringify({ id: NEW_NOTE_ID, body: "new" }), stderr: "" };
      }
      if (endpoint.includes("/notes?")) {
        if (opts.failList) throw new Error("502 Bad Gateway");
        return { stdout: JSON.stringify(existingNotes), stderr: "" };
      }
      if (args.includes("PUT") && opts.failPutFor != null && endpoint.endsWith(`/notes/${opts.failPutFor}`)) {
        throw new Error("500 Internal Server Error");
      }
      return { stdout: "{}", stderr: "" };
    });
  }

  function editedNoteIds(): string[] {
    return execMock.mock.calls
      .map((call) => call[1] as string[])
      .filter((args) => args.includes("PUT"))
      .map((args) => (args[1] ?? "").replace(/^.*\/notes\//, ""));
  }

  function inputOf(call: unknown[]): string {
    const options = call[2];
    return options && typeof options === "object" && "input" in options && typeof options.input === "string"
      ? options.input
      : "";
  }

  beforeEach(() => {
    execMock.mockReset();
    execJsonMock.mockReset();
  });

  async function publish(): Promise<{ noteId: number | null; collapsed: number }> {
    const { publishGitlabMrSummary } = await import("../src/gitlab.js");
    return publishGitlabMrSummary(
      "acme",
      "app",
      42,
      "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nnew",
      "gitlab.example.com",
      BOT,
    );
  }

  it("never collapses a newer summary posted by an overlapping run", async () => {
    existingNotes.push({
      id: NEW_NOTE_ID + 1,
      type: null,
      body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nnewer run",
      author: BOT_AUTHOR,
    });
    try {
      mockGitlab();
      await publish();
      expect(editedNoteIds()).not.toContain(String(NEW_NOTE_ID + 1));
      expect(editedNoteIds()).toEqual(["11", "12"]);
    } finally {
      existingNotes.pop();
    }
  });

  it("posts a new summary, then collapses only the publisher's older summaries", async () => {
    mockGitlab();

    const result = await publish();

    expect(result).toEqual({ noteId: NEW_NOTE_ID, collapsed: 2 });
    const calls = execMock.mock.calls;
    expect(calls[0][1]).toContain("POST");
    expect(inputOf(calls[0])).toBe(JSON.stringify({ body: "<!-- hodor-review -->\n<!-- hodor:summary:v1 -->\nnew" }));
    expect(editedNoteIds()).toEqual(["11", "12"]);
    const puts = calls.filter((call) => (call[1] as string[]).includes("PUT"));
    for (const put of puts) {
      expect(inputOf(put)).toBe(JSON.stringify({ body: COLLAPSED_BODY }));
    }
  });

  it("keeps the new summary as posted when collapsing a note fails", async () => {
    mockGitlab({ failPutFor: 11 });

    await expect(publish()).resolves.toEqual({ noteId: NEW_NOTE_ID, collapsed: 1 });
    expect(editedNoteIds()).toEqual(["11", "12"]);
  });

  it("keeps the new summary as posted when notes cannot be listed", async () => {
    mockGitlab({ failList: true });

    await expect(publish()).resolves.toEqual({ noteId: NEW_NOTE_ID, collapsed: 0 });
    expect(editedNoteIds()).toEqual([]);
  });

  it("edits nothing when GitLab returns no id for the new note", async () => {
    execMock.mockResolvedValue({ stdout: "", stderr: "" });

    await expect(publish()).resolves.toEqual({ noteId: null, collapsed: 0 });
    expect(execMock).toHaveBeenCalledTimes(1);
  });

  it("throws and edits nothing when the new summary cannot be posted", async () => {
    mockGitlab({ failPost: true });

    await expect(publish()).rejects.toThrow(/Failed to post summary to MR !42/);
    expect(editedNoteIds()).toEqual([]);
  });
});

describe("parsePaginatedJsonArrays", () => {
  it("returns empty array for empty input", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    expect(parsePaginatedJsonArrays("")).toEqual([]);
    expect(parsePaginatedJsonArrays("   ")).toEqual([]);
  });

  it("parses a single page", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    expect(parsePaginatedJsonArrays('[{"id":1},{"id":2}]')).toEqual([{ id: 1 }, { id: 2 }]);
  });

  it("merges multiple concatenated pages", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    const raw = '[{"id":1}][{"id":2},{"id":3}][{"id":4}]';
    expect(parsePaginatedJsonArrays(raw)).toEqual([{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }]);
  });

  it("handles strings containing bracket characters without splitting incorrectly", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    // The body string contains "][" which would break a naive regex-based split.
    const raw = '[{"id":1,"body":"weird ][ chars"}][{"id":2,"body":"normal"}]';
    expect(parsePaginatedJsonArrays(raw)).toEqual([
      { id: 1, body: "weird ][ chars" },
      { id: 2, body: "normal" },
    ]);
  });

  it("handles escaped quotes inside string values", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    const raw = '[{"id":1,"body":"has \\"quoted\\" text"}]';
    expect(parsePaginatedJsonArrays(raw)).toEqual([{ id: 1, body: 'has "quoted" text' }]);
  });

  it("handles nested arrays in note objects", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    const raw = '[{"id":1,"tags":["a","b"]},{"id":2,"tags":[]}][{"id":3}]';
    expect(parsePaginatedJsonArrays(raw)).toEqual([
      { id: 1, tags: ["a", "b"] },
      { id: 2, tags: [] },
      { id: 3 },
    ]);
  });

  it("skips malformed pages and continues with the rest", async () => {
    const { parsePaginatedJsonArrays } = await import("../src/utils/json.js");
    // Second chunk is malformed (truncated), but bracket depth still balances —
    // simulate by injecting invalid JSON that JSON.parse will reject.
    const raw = '[{"id":1}][not-json][{"id":3}]';
    expect(parsePaginatedJsonArrays(raw)).toEqual([{ id: 1 }, { id: 3 }]);
  });
});
