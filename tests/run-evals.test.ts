import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { MAX_REVIEW_INSTRUCTIONS_BYTES } from "../src/review-instructions.js";

const tempDirs: string[] = [];
const runnerPath = fileURLToPath(new URL("../scripts/run-evals.ts", import.meta.url));

function fixture(): { dir: string; out: string } {
  const dir = mkdtempSync(join(tmpdir(), "hodor-eval-profile-"));
  tempDirs.push(dir);
  const repo = join(dir, "repo");
  mkdirSync(repo);
  const git = (...args: string[]) => execFileSync("git", args, { cwd: repo, stdio: "pipe" });
  git("init", "--quiet", "--initial-branch", "main");
  writeFileSync(join(repo, "file.ts"), "export const value = 1;\n");
  git("add", "file.ts");
  git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "base");
  mkdirSync(join(dir, "cases"));
  writeFileSync(join(dir, "cases", "case.json"), JSON.stringify({
    id: "synthetic",
    description: "No-spend runner integration",
    repo,
    base: "HEAD",
    head: "HEAD",
    expectedFindings: [],
  }));
  mkdirSync(join(dir, "templates"));
  writeFileSync(join(dir, "templates", "security-review-instructions.md"), "");
  return { dir, out: join(dir, "runs.jsonl") };
}

function run(dir: string, out: string, args: string[]): { status: number | null; output: string } {
  // A provider rejected by preflight ensures this test cannot make a paid call.
  const result = spawnSync("bun", [runnerPath, "--cases", join(dir, "cases"), "--model", "invalid/test", "--out", out, ...args], {
    cwd: dir,
    env: { PATH: process.env.PATH, PI_OFFLINE: "1" },
    encoding: "utf8",
    timeout: 30_000,
  });
  return { status: result.status, output: `${result.stdout}${result.stderr}` };
}

function readRecord(out: string): Record<string, unknown> {
  const value: unknown = JSON.parse(readFileSync(out, "utf8"));
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Expected a JSONL record");
  return value;
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("evaluation runner profiles", () => {
  it("validates profile names even when listing cases", () => {
    const { dir, out } = fixture();
    const result = run(dir, out, ["--list", "--profile", "unknown"]);
    expect(result.status).toBe(1);
    expect(result.output).toContain("--profile must be one of: security");
  });

  it("resolves the installed profile, not the cwd look-alike, when listing", () => {
    const { dir, out } = fixture();
    expect(run(dir, out, ["--list", "--profile", "security"])).toEqual({
      status: 0,
      output: "synthetic: No-spend runner integration\n",
    });
  });

  it("preserves the unprofiled JSONL fields and console header", () => {
    const { dir, out } = fixture();
    const result = run(dir, out, []);
    expect(result.status).toBe(1);
    expect(result.output).toContain("run: 1 case(s) x 1 trial(s) = 1 review(s) with invalid/test\n");
    const record = readRecord(out);
    expect(Object.keys(record)).toEqual([
      "label", "model", "focus", "instructions", "reasoningEffort", "codemode",
      "caseId", "trial", "score", "error", "cost", "durationSeconds",
    ]);
    expect(record).toMatchObject({ instructions: [], focus: null, error: expect.stringContaining('Unsupported provider "invalid"') });
  });

  it("records the selected profile without changing explicit file metadata", () => {
    const { dir, out } = fixture();
    const overlay = join(dir, "overlay.md");
    writeFileSync(overlay, "OVERLAY_RULE");
    const result = run(dir, out, ["--instructions", overlay, "--profile", "security", "--focus", "FOCUS_RULE"]);
    expect(result.status).toBe(1);
    expect(result.output).toContain("with invalid/test, profile security, instructions");
    expect(readRecord(out)).toMatchObject({
      profile: "security",
      instructions: [overlay],
      focus: "FOCUS_RULE",
      error: expect.stringContaining('Unsupported provider "invalid"'),
    });
  });

  it("actually passes the resolved profile into the review instruction budget", () => {
    const { dir, out } = fixture();
    const first = join(dir, "first.md");
    const second = join(dir, "second.md");
    writeFileSync(first, "x".repeat(MAX_REVIEW_INSTRUCTIONS_BYTES));
    writeFileSync(second, "y".repeat(MAX_REVIEW_INSTRUCTIONS_BYTES));
    const files = ["--instructions", first, "--instructions", second];
    expect(run(dir, out, files).status).toBe(1);
    expect(readRecord(out).error).toEqual(expect.stringContaining('Unsupported provider "invalid"'));
    const profileOut = join(dir, "profile.jsonl");
    expect(run(dir, profileOut, [...files, "--profile", "security"]).status).toBe(1);
    expect(readRecord(profileOut)).toMatchObject({ profile: "security", error: expect.stringContaining("Combined instructions exceed") });
  });
});
