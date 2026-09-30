import * as childProcess from "node:child_process";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import {
  createAgentSession,
  DefaultResourceLoader,
  SessionManager,
  SettingsManager,
  type AgentSession,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream, type AssistantMessage, type ToolCall } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { createReviewResourceLoader, getReviewSessionTools } from "../src/agent.js";
import { createModelRuntime } from "../src/models-json.js";
import { createReviewToolset, globToRegExp, REVIEW_TOOL_NAMES, type ReviewToolset } from "../src/review-tools.js";

// Spy on spawn while keeping the real implementation, so the sentinel test
// can inspect the exact environment each tool subprocess received.
vi.mock("node:child_process", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:child_process")>();
  return { ...actual, spawn: vi.fn(actual.spawn) };
});

const SENTINEL = "hodor-sentinel-credential-7f3a";
const SECRET = "needle-secret";

let base = "";
let repo = "";
let reviewDiff = "";
let toolset: ReviewToolset;
let session: AgentSession;
let savedToken: string | undefined;

function git(cwd: string, args: string[]): string {
  return execFileSync("git", args, {
    cwd,
    encoding: "utf-8",
    env: {
      ...process.env,
      GIT_CONFIG_GLOBAL: "/dev/null",
      GIT_CONFIG_NOSYSTEM: "1",
      GIT_AUTHOR_NAME: "Test",
      GIT_AUTHOR_EMAIL: "test@example.com",
      GIT_COMMITTER_NAME: "Test",
      GIT_COMMITTER_EMAIL: "test@example.com",
    },
  });
}

function write(path: string, content: string): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, content);
}

const submitReviewTool: ToolDefinition = {
  name: "submit_review",
  exposure: "model-only",
  label: "Submit Review",
  description: "Submit the review.",
  parameters: Type.Object({}),
  execute: async () => ({ content: [{ type: "text", text: "ok" }], details: {} }),
};

async function createSession(
  singleTurn: boolean,
  codemode = false,
  codemodeLimits?: { timeoutMs: number; maxOutputTokens: number },
): Promise<AgentSession> {
  const agentDir = join(base, "pi-agent");
  mkdirSync(agentDir, { recursive: true });
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, cacheWarming: "off" });
  const resourceLoader = await createReviewResourceLoader({ cwd: repo, agentDir, settingsManager, codemode, codemodeLimits });
  const modelRuntime = await createModelRuntime(null);
  const { session: created } = await createAgentSession({
    cwd: repo,
    agentDir,
    model: modelRuntime.getModel("anthropic", "claude-opus-5-5"),
    modelRuntime,
    ...getReviewSessionTools({ singleTurn, reviewTools: toolset.definitions, submitReviewTool, codemode }),
    sessionManager: SessionManager.inMemory(),
    settingsManager,
    resourceLoader,
  });
  return created;
}

/** Execute a tool the way the agent loop does, through the session's registered AgentTool. */
async function call(name: string, params: Record<string, unknown>): Promise<string> {
  const tool = session.agent.state.tools.find((candidate) => candidate.name === name);
  if (!tool) throw new Error(`tool ${name} is not active`);
  const result = await tool.execute(`call-${name}`, params);
  return result.content
    .map((block) => (block.type === "text" ? block.text : `[${block.type}]`))
    .join("\n");
}

beforeAll(async () => {
  savedToken = process.env.GITLAB_TOKEN;
  process.env.GITLAB_TOKEN = SENTINEL;

  base = mkdtempSync(join(tmpdir(), "hodor-tools-"));
  repo = join(base, "repo");
  mkdirSync(repo);
  write(join(base, "outside", "secret.txt"), `${SECRET} outside\n`);
  write(join(base, "repo2", "secret.txt"), `${SECRET} sibling\n`);

  git(repo, ["init", "-q"]);
  write(join(repo, "src", "app.ts"), "export function greet(name: string) {\n  return `hi ${name}`; // needle\n}\n");
  write(join(repo, "src", "util.ts"), "export const needle = 1;\n");
  write(join(repo, "deleted.ts"), "export const gone = true; // needle\n");
  write(join(repo, ".github", "workflows", "ci.yml"), "name: ci # needle-dot\n");
  write(join(repo, ".gitignore"), ".env\nignored/\n");
  write(join(repo, "many.txt"), Array.from({ length: 300 }, (_, i) => `many needle ${i}`).join("\n") + "\n");
  symlinkSync("../outside/secret.txt", join(repo, "link-out"));
  symlinkSync("src/app.ts", join(repo, "link-in"));
  symlinkSync("src", join(repo, "linkdir"));
  git(repo, ["add", "-A"]);
  git(repo, ["commit", "-q", "-m", "base"]);

  write(join(repo, "src", "app.ts"), "export function greet(name: string) {\n  return `hello ${name}`; // needle\n}\n");
  write(join(repo, "src", "new.ts"), "export const fresh = 'needle';\n");
  write(join(repo, "big.txt"), Array.from({ length: 3000 }, (_, i) => `line ${i}`).join("\n") + "\n");
  rmSync(join(repo, "deleted.ts"));
  git(repo, ["add", "-A"]);
  git(repo, ["commit", "-q", "-m", "change"]);
  reviewDiff = git(repo, ["--no-pager", "diff", "HEAD~1", "HEAD"]);

  // Files a prompt-injected diff would want: none of these are tracked.
  write(join(repo, ".env"), `GITLAB_TOKEN=${SECRET}\n`);
  write(join(repo, "ignored", "cache.ts"), `${SECRET}\n`);
  write(join(repo, "untracked.txt"), `${SECRET}\n`);
  write(join(repo, "src", "untracked.ts"), `${SECRET}\n`);
  write(join(repo, ".glab-ci", "config.yml"), `token: ${SECRET}\n`);
  symlinkSync("../outside", join(repo, "vendor"));
  git(repo, ["config", "hodor.test", SECRET]);

  toolset = await createReviewToolset({ workspacePath: repo, reviewDiff });
  session = await createSession(false);
});

afterAll(() => {
  session?.dispose();
  toolset?.dispose();
  if (savedToken === undefined) delete process.env.GITLAB_TOKEN;
  else process.env.GITLAB_TOKEN = savedToken;
  rmSync(base, { recursive: true, force: true });
});

describe("review session tools", () => {
  it("exposes exactly the confined tools and no shell", () => {
    expect([...session.getActiveToolNames()].sort()).toEqual(
      ["find", "git_diff", "grep", "ls", "read", "submit_review"],
    );
    expect(session.agent.state.tools.map((tool) => tool.name).sort()).toEqual(
      ["find", "git_diff", "grep", "ls", "read", "submit_review"],
    );
    expect(session.getActiveToolNames()).not.toContain("bash");
    expect(session.getToolDefinition("bash")).toBeUndefined();
  });

  it("replaces Pi's built-in read, grep, find, and ls with the confined definitions", () => {
    for (const name of REVIEW_TOOL_NAMES) {
      const definition = toolset.definitions.find((candidate) => candidate.name === name);
      expect(definition).toBeDefined();
      expect(session.getToolDefinition(name)).toBe(definition);
    }
    const sources = session.getAllTools()
      .filter((tool) => session.getActiveToolNames().includes(tool.name))
      .map((tool) => tool.sourceInfo.source);
    expect(sources).not.toContain("builtin");
  });

  it("exposes only submit_review on the single-turn fast path", async () => {
    const fastSession = await createSession(true);
    try {
      expect(fastSession.getActiveToolNames()).toEqual(["submit_review"]);
    } finally {
      fastSession.dispose();
    }
  });
});

describe("codemode (experimental)", () => {
  /**
   * Run one codemode script through a real agent turn: a scripted model issues
   * the codemode call, then submit_review to end the run. Nested tool calls
   * need a real assistant turn, so the tool cannot be executed directly.
   */
  async function runScript(
    code: string,
    limits?: { timeoutMs: number; maxOutputTokens: number },
  ): Promise<string> {
    // The scripted stream never sends a request, but prompt() requires a key.
    vi.stubEnv("ANTHROPIC_API_KEY", "offline-test-key");
    const cm = await createSession(false, true, limits);
    try {
      const model = cm.model;
      if (!model) throw new Error("session has no model");
      const calls: ToolCall[] = [
        { type: "toolCall", id: "script", name: "codemode", arguments: { code } },
        { type: "toolCall", id: "submit", name: "submit_review", arguments: {} },
      ];
      let turn = 0;
      cm.agent.streamFunction = () => {
        const call = calls[turn++];
        if (!call) throw new Error("unexpected extra turn");
        const message: AssistantMessage = {
          role: "assistant",
          content: [call],
          api: model.api,
          provider: model.provider,
          model: model.id,
          usage: {
            input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
          },
          stopReason: "toolUse",
          timestamp: Date.now(),
        };
        const stream = createAssistantMessageEventStream();
        stream.push({ type: "start", partial: message });
        stream.push({ type: "done", reason: "toolUse", message });
        return stream;
      };
      await cm.prompt("offline codemode test");
      const result = cm.messages.find(
        (message) => message.role === "toolResult" && message.toolName === "codemode",
      );
      if (!result || result.role !== "toolResult") throw new Error("no codemode result");
      return result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
    } finally {
      cm.dispose();
      vi.unstubAllEnvs();
    }
  }

  it("adds only codemode to the confined tools", async () => {
    const cm = await createSession(false, true);
    try {
      expect([...cm.getActiveToolNames()].sort()).toEqual(
        ["codemode", "find", "git_diff", "grep", "ls", "read", "submit_review"],
      );
      for (const forbidden of ["bash", "tool_search", "write", "edit"]) {
        expect(cm.getToolDefinition(forbidden)).toBeUndefined();
      }
    } finally {
      cm.dispose();
    }
  });

  it("gives scripts the confined tools and no host globals, model API, or submit_review", async () => {
    const out = await runScript(
      "text(JSON.stringify({ globals: [typeof process, typeof require, typeof fetch, typeof setTimeout, typeof models], " +
        "tools: ALL_TOOLS.map((t) => t.name).sort(), submit: typeof tools.submit_review }));",
    );
    const parsed: unknown = JSON.parse(out.slice(out.indexOf("{"), out.lastIndexOf("}") + 1));
    expect(parsed).toEqual({
      globals: ["undefined", "undefined", "undefined", "undefined", "undefined"],
      tools: ["find", "git_diff", "grep", "ls", "read"],
      submit: "undefined",
    });
  });

  it("stops a script that runs past the enforced timeout", async () => {
    const started = Date.now();
    const out = await runScript("while (true) {}", { timeoutMs: 500, maxOutputTokens: 10_000 });
    expect(out).toMatch(/Script failed/);
    expect(Date.now() - started).toBeLessThan(10_000);
  });

  it("caps a timeout the script asks for", async () => {
    const out = await runScript('// @options: {"timeout_ms": 999999999}\nwhile (true) {}', {
      timeoutMs: 500,
      maxOutputTokens: 10_000,
    });
    expect(out).toMatch(/Script failed/);
  });

  it("keeps confinement for reads made from a script", async () => {
    const script = [
      "const r = await Promise.allSettled([",
      "  tools.read({ path: '/etc/passwd' }),",
      "  tools.read({ path: '.git/config' }),",
      "  tools.read({ path: 'src/app.ts' }),",
      "]);",
      "text(JSON.stringify(r.map((x) => x.status)));",
    ].join("\n");
    expect(await runScript(script)).toContain('["rejected","rejected","fulfilled"]');
  });
});

describe("read", () => {
  it("reads tracked files, tracked dotfiles, and tracked symlinks to tracked files", async () => {
    expect(await call("read", { path: "src/app.ts" })).toContain("hello");
    expect(await call("read", { path: join(repo, "src", "util.ts") })).toContain("needle = 1");
    expect(await call("read", { path: ".github/workflows/ci.yml" })).toContain("needle-dot");
    expect(await call("read", { path: ".gitignore" })).toContain(".env");
    expect(await call("read", { path: "link-in" })).toContain("hello");
  });

  it.each([
    ["an absolute path outside the repository", "/etc/passwd", /outside the repository/],
    ["`..` traversal", "../outside/secret.txt", /outside the repository/],
    ["a sibling directory sharing the repository prefix", "../repo2/secret.txt", /outside the repository/],
    ["an absolute sibling-prefix path", "SIBLING", /outside the repository/],
    ["`~` expansion", "~/.gitconfig", /outside the repository/],
    ["a tracked symlink escaping the repository", "link-out", /symlink whose target is outside/],
    ["an untracked symlinked directory", "vendor/secret.txt", /not a tracked file/],
    ["a path through a tracked directory symlink", "linkdir/app.ts", /not a tracked file/],
    ["an ignored .env file", ".env", /not a tracked file/],
    ["an ignored directory", "ignored/cache.ts", /not a tracked file/],
    ["an untracked file", "untracked.txt", /not a tracked file/],
    ["the CI glab config", ".glab-ci/config.yml", /not a tracked file/],
    ["git metadata", ".git/config", /inside \.git/],
    ["a file deleted by the change", "deleted.ts", /not a tracked file/],
  ])("rejects %s", async (_label, path, reason) => {
    const target = path === "SIBLING" ? join(base, "repo2", "secret.txt") : path;
    await expect(call("read", { path: target })).rejects.toThrow(reason);
  });
});

describe("ls", () => {
  it("lists only tracked entries", async () => {
    const entries = (await call("ls", {})).split("\n");
    expect(entries).toEqual(expect.arrayContaining([".github/", ".gitignore", "src/", "link-in", "many.txt"]));
    for (const hidden of [".env", ".git/", ".git", ".glab-ci/", "ignored/", "untracked.txt", "vendor/", "link-out"]) {
      expect(entries).not.toContain(hidden);
    }
    expect((await call("ls", { path: "src" })).split("\n")).toEqual(["app.ts", "new.ts", "util.ts"]);
  });

  it("rejects .git, untracked directories, and paths outside the repository", async () => {
    await expect(call("ls", { path: ".git" })).rejects.toThrow(/inside \.git/);
    await expect(call("ls", { path: ".glab-ci" })).rejects.toThrow(/not a tracked file/);
    await expect(call("ls", { path: ".." })).rejects.toThrow(/outside the repository/);
    await expect(call("ls", { path: "vendor" })).rejects.toThrow(/not a tracked file/);
  });
});

describe("find", () => {
  it("finds tracked files by name and by path glob", async () => {
    expect((await call("find", { pattern: "*.ts" })).split("\n").sort()).toEqual(
      ["src/app.ts", "src/new.ts", "src/util.ts"],
    );
    expect((await call("find", { pattern: "src/**/*.ts", path: "." })).split("\n")).toHaveLength(3);
    expect(await call("find", { pattern: "*.yml" })).toBe(".github/workflows/ci.yml");
    expect(await call("find", { pattern: "*.ts", path: "src" })).toContain("app.ts");
    expect(await call("find", { pattern: "config.yml" })).toBe("No files found matching pattern");
  });

  it("rejects search roots outside the tracked tree", async () => {
    await expect(call("find", { pattern: "*", path: "/etc" })).rejects.toThrow(/outside the repository/);
    await expect(call("find", { pattern: "*", path: ".git" })).rejects.toThrow(/inside \.git/);
    await expect(call("find", { pattern: "*", path: "src/app.ts" })).rejects.toThrow(/not a directory/);
  });

  it("translates globs like fd", () => {
    expect(globToRegExp("**/node_modules/**").test("node_modules/a.js")).toBe(true);
    expect(globToRegExp("*.{ts,tsx}").test("a.tsx")).toBe(true);
    expect(globToRegExp("*.ts").test("src/a.ts")).toBe(false);
    expect(globToRegExp("src/[!x]?.ts").test("src/ab.ts")).toBe(true);
  });
});

describe("grep", () => {
  it("returns tracked matches only and never untracked secrets", async () => {
    const output = await call("grep", { pattern: "needle", limit: 1000 });
    expect(output).toContain("src/app.ts:2:");
    expect(output).toContain("src/util.ts:1: export const needle = 1;");
    expect(output).toContain(".github/workflows/ci.yml:1:");
    expect(output).not.toContain(SECRET);
    expect(output).not.toContain("deleted.ts");
    expect(await call("grep", { pattern: SECRET })).toBe("No matches found");
  });

  it("supports path, glob, literal, ignoreCase, and context like Pi's grep", async () => {
    expect(await call("grep", { pattern: "needle", glob: "*.yml" })).toBe(".github/workflows/ci.yml:1: name: ci # needle-dot");
    expect(await call("grep", { pattern: "needle", path: "src/util.ts" })).toBe("util.ts:1: export const needle = 1;");
    expect(await call("grep", { pattern: "needle", path: "src", glob: "new.ts" })).toBe("new.ts:1: export const fresh = 'needle';");
    expect(await call("grep", { pattern: "${name}", literal: true, path: "src" })).toContain("app.ts:2:");
    expect(await call("grep", { pattern: "HELLO", ignoreCase: true })).toContain("src/app.ts:2:");
    expect(await call("grep", { pattern: "\\bhello\\b", path: "src" })).toContain("app.ts:2:");
    expect(await call("grep", { pattern: "hello", path: "src/app.ts", context: 1 })).toBe(
      "app.ts-1- export function greet(name: string) {\napp.ts:2:   return `hello ${name}`; // needle\napp.ts-3- }",
    );
  });

  it("treats option-like patterns as patterns", async () => {
    expect(await call("grep", { pattern: "--no-index" })).toBe("No matches found");
    expect(await call("grep", { pattern: "--untracked" })).toBe("No matches found");
  });

  it("caps matches with an explicit notice", async () => {
    const output = await call("grep", { pattern: "many needle", limit: 5 });
    expect(output.split("\n").filter((line) => line.startsWith("many.txt:"))).toHaveLength(5);
    expect(output).toContain("[5 matches limit reached. Use limit=10 for more, or refine pattern]");
  });

  it("rejects paths and globs that leave the tracked tree", async () => {
    await expect(call("grep", { pattern: "x", path: "/etc" })).rejects.toThrow(/outside the repository/);
    await expect(call("grep", { pattern: "x", path: "../repo2" })).rejects.toThrow(/outside the repository/);
    await expect(call("grep", { pattern: "x", path: "~" })).rejects.toThrow(/outside the repository/);
    await expect(call("grep", { pattern: "x", path: ".git" })).rejects.toThrow(/inside \.git/);
    await expect(call("grep", { pattern: "x", path: ".env" })).rejects.toThrow(/not a tracked file/);
    await expect(call("grep", { pattern: "x", path: "link-out" })).rejects.toThrow(/symlink/);
    await expect(call("grep", { pattern: "x", glob: ":(top)*" })).rejects.toThrow(/Invalid glob/);
    await expect(call("grep", { pattern: "x", glob: "../*" })).rejects.toThrow(/Invalid glob/);
  });
});

describe("git_diff", () => {
  it("lists every changed file, including deleted ones", async () => {
    const output = await call("git_diff", {});
    expect(output).toContain("Changed files (4):");
    expect(output).toContain("deleted.ts (+0 -1)");
    expect(output).toContain("src/app.ts (+1 -1)");
    expect(output).toContain("src/new.ts (+1 -0)");
  });

  it("serves one changed file's diff, including a deleted file", async () => {
    const app = await call("git_diff", { path: "src/app.ts" });
    expect(app.startsWith("diff --git a/src/app.ts b/src/app.ts")).toBe(true);
    expect(app).toContain("+  return `hello ${name}`; // needle");
    expect(app).not.toContain("src/new.ts");
    expect(await call("git_diff", { path: join(repo, "src", "app.ts") })).toBe(app);
    expect(await call("git_diff", { path: "./src/app.ts" })).toBe(app);
    expect(await call("git_diff", { path: "deleted.ts" })).toContain("-export const gone = true;");
  });

  it("rejects unchanged files and option-like or escaping input", async () => {
    await expect(call("git_diff", { path: "src/util.ts" })).rejects.toThrow(/not a changed file/);
    await expect(call("git_diff", { path: "--output=/tmp/pwned" })).rejects.toThrow(/not a changed file/);
    await expect(call("git_diff", { path: "HEAD~5" })).rejects.toThrow(/not a changed file/);
    await expect(call("git_diff", { path: "../outside/secret.txt" })).rejects.toThrow(/outside the repository/);
    await expect(call("git_diff", { path: "/etc/passwd" })).rejects.toThrow(/outside the repository/);
  });

  it("caps large file diffs with a continuation notice", async () => {
    const first = await call("git_diff", { path: "big.txt" });
    expect(first).toMatch(/\[Showing diff lines 1-2000 of \d+\. Use offset=2001 to continue\.\]$/);
    const rest = await call("git_diff", { path: "big.txt", offset: 2001 });
    expect(rest).toContain("+line 2999");
    expect(rest).not.toContain("Use offset=");
  });
});

describe("tool subprocess environment", () => {
  it("never passes Hodor's credentials or a shell to git", async () => {
    const spawnMock = vi.mocked(childProcess.spawn);
    spawnMock.mockClear();
    await call("grep", { pattern: "needle" });
    await call("grep", { pattern: "needle", path: "src", context: 1 });
    await createReviewToolset({ workspacePath: repo, reviewDiff }).then((extra) => extra.dispose());

    expect(spawnMock.mock.calls.length).toBeGreaterThanOrEqual(3);
    for (const [command, args, options] of spawnMock.mock.calls) {
      expect(command).toMatch(/^\/.*\/git$/);
      expect(args).toEqual(expect.arrayContaining(["--no-pager", "core.fsmonitor=false"]));
      expect(options).toMatchObject({ shell: false });
      const env = options?.env ?? {};
      expect(Object.values(env)).not.toContain(SENTINEL);
      expect(env.GITLAB_TOKEN).toBeUndefined();
      expect(env).toMatchObject({
        PATH: "/usr/local/bin:/usr/bin:/bin",
        LANG: "C",
        GIT_CONFIG_NOSYSTEM: "1",
        GIT_CONFIG_GLOBAL: "/dev/null",
        GIT_TERMINAL_PROMPT: "0",
      });
      expect(env.HOME).not.toBe(homedir());
    }
  });
});
