import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const cwd = process.cwd();

function runCli(args: string[]): { status: number; output: string } {
  try {
    const output = execFileSync("bun", ["run", "src/cli.ts", ...args], {
      cwd,
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { status: 0, output };
  } catch (error) {
    const failure = error as { status?: number; stdout?: string; stderr?: string };
    return {
      status: failure.status ?? 1,
      output: `${failure.stdout ?? ""}${failure.stderr ?? ""}`,
    };
  }
}

describe("CLI policy validation", () => {
  it("reports the release version", () => {
    expect(runCli(["--version"])).toEqual({ status: 0, output: "0.7.0-rc.2\n" });
  });

  it("rejects invalid priority thresholds before starting a review", () => {
    const result = runCli(["--local", "--fail-on-priority", "critical"]);
    expect(result.status).toBe(1);
    expect(result.output).toContain("P0, P1, P2, P3");
  });

  it("requires a delivery target for strict delivery mode", () => {
    const result = runCli(["--local", "--require-delivery"]);
    expect(result.status).toBe(1);
    expect(result.output).toContain("requires --post or --code-quality");
  });

  it("documents review profiles and additional instructions in help", () => {
    const result = runCli(["--help"]);

    expect(result.status).toBe(0);
    expect(result.output).toContain("--review-instructions <path>");
    expect(result.output).toContain("custom review instruction profile");
    expect(result.output).toContain("--additional-instructions <text>");
    expect(result.output.replace(/\s+/g, " ")).toContain("Additional review instructions appended after the selected review profile");
    expect(result.output).not.toContain("--prompt-file");
    expect(result.output).not.toContain("--prompt <");
  });

  it("rejects an unreadable review profile before starting workspace setup", () => {
    const missingProfile = join(cwd, ".hodor-missing-review-profile-test.md");
    const result = runCli(["--local", "--review-instructions", missingProfile]);

    expect(result.status).toBe(1);
    expect(result.output).toContain("Unable to read review instructions");
    expect(result.output).not.toContain("Setting up workspace");
  });

  it("rejects removed prompt-template override flags", () => {
    for (const legacyFlag of ["--prompt-file", "--prompt"]) {
      const result = runCli(["--local", legacyFlag, "legacy-value"]);

      expect(result.status).toBe(1);
      expect(result.output).toContain(`unknown option '${legacyFlag}'`);
    }
  });
});
