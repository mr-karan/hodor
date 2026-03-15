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

  it("cleanupHodorComments parses paginated glab output", async () => {
    execMock
      .mockResolvedValueOnce({
        stdout:
          '[{"id":1,"body":"<!-- hodor-review --> old"}]' +
          '[{"id":2,"body":"human comment"}]',
        stderr: "",
      })
      .mockResolvedValueOnce({ stdout: "", stderr: "" });

    const { cleanupHodorComments } = await import("../src/gitlab.js");
    await expect(cleanupHodorComments("acme", "app", 42, "gitlab.example.com")).resolves.toBe(1);

    expect(execMock).toHaveBeenCalledTimes(2);
    expect(execMock.mock.calls[1][1]).toContain("DELETE");
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
});
