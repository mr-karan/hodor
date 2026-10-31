import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const tempDirs: string[] = [];
const cliPath = fileURLToPath(new URL("../src/cli.ts", import.meta.url));

afterEach(() => {
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function runPreflight(args: string[]): string {
  const dir = mkdtempSync(join(tmpdir(), "hodor-cli-profile-"));
  tempDirs.push(dir);
  // A checkout-supplied look-alike must not replace the installed profile.
  mkdirSync(join(dir, "templates"));
  writeFileSync(join(dir, "templates", "security-review-instructions.md"), "");
  writeFileSync(join(dir, "first.md"), "FIRST_RULE\n");
  writeFileSync(join(dir, "second.md"), "SECOND_RULE\n");
  // An invalid provider stops the real review at preflight, before any model call.
  const result = spawnSync("bun", [cliPath, "--local", "--verbose", "--model", "invalid/test", ...args], {
    cwd: dir,
    env: { PATH: process.env.PATH, PI_OFFLINE: "1" },
    encoding: "utf8",
    timeout: 30_000,
  });
  expect(result.status).toBe(1);
  const output = `${result.stdout}${result.stderr}`;
  expect(output).toContain('Unsupported provider "invalid"');
  return output;
}

describe("CLI profile preflight", () => {
  it("does not select a profile or review key by default", () => {
    const output = runPreflight([]);
    expect(output).not.toContain("Review profile:");
    expect(output).not.toContain("Review key:");
    expect(output).not.toContain("Explicit instructions:");
    expect(output).not.toContain("Review focus:");
  });

  it("loads the installed security profile before explicit files, regardless of flag order", () => {
    const output = runPreflight([
      "--instructions", "first.md", "--focus", "FOCUS_RULE",
      "--profile", "security", "--instructions", "second.md",
    ]);
    const positions = ["Review profile: security", "Explicit instructions: first.md", "Explicit instructions: second.md", "Review focus: supplied"].map((text) => output.indexOf(text));
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(output).not.toContain("Review key:");
  });

  it("keeps an explicitly selected review key", () => {
    expect(runPreflight(["--profile", "security", "--review-key", "security"])).toContain("Review key: security");
  });
});
