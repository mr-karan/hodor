import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const execMock = vi.fn();
const execJsonMock = vi.fn();

vi.mock("../src/utils/exec.js", () => ({
  exec: execMock,
  execJson: execJsonMock,
}));

const ORIGINAL_ENV = { ...process.env };

function resetEnv(): void {
  process.env = { ...ORIGINAL_ENV };
  delete process.env.GITHUB_ACTIONS;
  delete process.env.GITHUB_WORKSPACE;
  delete process.env.GITHUB_REPOSITORY;
  delete process.env.GITHUB_BASE_REF;
  delete process.env.GITLAB_CI;
  delete process.env.CI_PROJECT_DIR;
  delete process.env.CI_PROJECT_PATH;
  delete process.env.CI_MERGE_REQUEST_TARGET_BRANCH_NAME;
  delete process.env.CI_MERGE_REQUEST_DIFF_BASE_SHA;
  delete process.env.GITEA_ACTIONS;
  delete process.env.FORGEJO_ACTIONS;
}

describe("setupWorkspace", () => {
  beforeEach(() => {
    resetEnv();
    execMock.mockReset();
    execJsonMock.mockReset();
    execJsonMock.mockResolvedValue({ baseRefName: "main" });
  });

  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  it("falls back to cloning in GitHub Actions when GITHUB_WORKSPACE is not a git checkout", async () => {
    process.env.GITHUB_ACTIONS = "true";
    process.env.GITHUB_WORKSPACE = "/__w/repo/repo";
    process.env.GITHUB_REPOSITORY = "octo/repo";
    process.env.GITHUB_BASE_REF = "";

    execMock.mockImplementation(async (cmd: string, args: string[]) => {
      if (cmd === "git" && args.join(" ") === "remote get-url origin") {
        throw new Error("not a git repository");
      }
      return { stdout: "", stderr: "" };
    });

    const { setupWorkspace } = await import("../src/workspace.js");
    const result = await setupWorkspace({
      platform: "github",
      owner: "octo",
      repo: "repo",
      prNumber: "123",
    });

    expect(result.workspace).not.toBe("/__w/repo/repo");
    expect(result.isTemporary).toBe(true);
    expect(result.targetBranch).toBe("main");
    expect(execMock).toHaveBeenCalledWith("gh", ["version"]);
    expect(execMock).toHaveBeenCalledWith(
      "gh",
      ["repo", "clone", "octo/repo", result.workspace],
    );
    expect(execMock).toHaveBeenCalledWith("gh", ["pr", "checkout", "123"], { cwd: result.workspace });
  });

  it("uses a valid GitHub Actions checkout and resolves missing base ref from PR metadata", async () => {
    process.env.GITHUB_ACTIONS = "true";
    process.env.GITHUB_WORKSPACE = "/__w/repo/repo";
    process.env.GITHUB_REPOSITORY = "octo/repo";
    process.env.GITHUB_BASE_REF = "";

    execMock.mockResolvedValue({ stdout: "https://github.com/octo/repo.git\n", stderr: "" });
    execJsonMock.mockResolvedValue({ baseRefName: "develop" });

    const { setupWorkspace } = await import("../src/workspace.js");
    const result = await setupWorkspace({
      platform: "github",
      owner: "octo",
      repo: "repo",
      prNumber: "123",
    });

    expect(result).toEqual({
      workspace: "/__w/repo/repo",
      targetBranch: "develop",
      diffBaseSha: null,
      isTemporary: false,
    });
    expect(execMock).not.toHaveBeenCalledWith("gh", ["version"]);
    expect(execJsonMock).toHaveBeenCalledWith(
      "gh",
      ["pr", "view", "123", "--json", "headRefName,baseRefName"],
      { cwd: "/__w/repo/repo" },
    );
  });

  it("normalizes empty GitLab CI branch variables to null and still validates checkout", async () => {
    process.env.GITLAB_CI = "true";
    process.env.CI_PROJECT_DIR = "/builds/group/repo";
    process.env.CI_PROJECT_PATH = "group/repo";
    process.env.CI_MERGE_REQUEST_TARGET_BRANCH_NAME = "";
    process.env.CI_MERGE_REQUEST_DIFF_BASE_SHA = "";

    execMock.mockResolvedValue({ stdout: "git@gitlab.example.com:group/repo.git\n", stderr: "" });

    const { setupWorkspace } = await import("../src/workspace.js");
    const result = await setupWorkspace({
      platform: "gitlab",
      owner: "group",
      repo: "repo",
      prNumber: "42",
      host: "gitlab.example.com",
    });

    expect(result).toEqual({
      workspace: "/builds/group/repo",
      targetBranch: "main",
      diffBaseSha: null,
      isTemporary: false,
    });
  });

  it("recalculates the GitLab MR base after a force-pushed rebase", async () => {
    process.env.GITLAB_CI = "true";
    process.env.CI_PROJECT_DIR = "/builds/group/repo";
    process.env.CI_PROJECT_PATH = "group/repo";
    process.env.CI_MERGE_REQUEST_TARGET_BRANCH_NAME = "main";
    process.env.CI_MERGE_REQUEST_DIFF_BASE_SHA = "stale-base-sha";

    execMock.mockImplementation(async (cmd: string, args: string[]) => {
      if (cmd === "git" && args.join(" ") === "remote get-url origin") {
        return { stdout: "git@gitlab.example.com:group/repo.git\n", stderr: "" };
      }
      if (cmd === "git" && args[0] === "merge-base") {
        return { stdout: "current-base-sha\n", stderr: "" };
      }
      return { stdout: "", stderr: "" };
    });

    const { setupWorkspace } = await import("../src/workspace.js");
    const result = await setupWorkspace({
      platform: "gitlab",
      owner: "group",
      repo: "repo",
      prNumber: "42",
      host: "gitlab.example.com",
    });

    expect(result.diffBaseSha).toBe("current-base-sha");
    expect(execMock).toHaveBeenCalledWith(
      "git",
      ["fetch", "--no-tags", "origin", "main"],
      { cwd: "/builds/group/repo" },
    );
    expect(execMock).toHaveBeenCalledWith(
      "git",
      ["merge-base", "HEAD", "FETCH_HEAD"],
      { cwd: "/builds/group/repo" },
    );
  });
});
