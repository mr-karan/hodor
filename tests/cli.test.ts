import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const cwd = process.cwd();
const packageVersion = (
  JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf-8")) as {
    version: string;
  }
).version;

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
    expect(runCli(["--version"])).toEqual({ status: 0, output: `${packageVersion}\n` });
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

  it("documents additive instruction files and focus in help", () => {
    const result = runCli(["--help"]);

    expect(result.status).toBe(0);
    expect(result.output).toContain("--instructions <path>");
    expect(result.output.replace(/\s+/g, " ")).toContain("Path to additive review instructions");
    expect(result.output).toContain("--focus <text>");
    expect(result.output).not.toContain("--review-instructions");
    expect(result.output).not.toContain("--additional-instructions");
    expect(result.output).not.toContain("--prompt-file");
    expect(result.output).not.toContain("--prompt <");
  });

  it("rejects an unreadable review profile before starting workspace setup", () => {
    const missingProfile = join(cwd, ".hodor-missing-review-profile-test.md");
    const result = runCli(["--local", "--instructions", missingProfile]);

    expect(result.status).toBe(1);
    expect(result.output).toContain("Unable to read review instructions");
    expect(result.output).not.toContain("Setting up workspace");
  });

  it("accumulates repeated instruction paths in their supplied order", () => {
    const first = join(cwd, ".hodor-missing-first-instructions.md");
    const second = join(cwd, ".hodor-missing-second-instructions.md");
    const result = runCli(["--local", "--instructions", first, "--instructions", second]);
    expect(result.status).toBe(1);
    expect(result.output).toContain(first);
    expect(result.output).not.toContain(second);
  });

  it("rejects replacement and additional-instruction flags with migration advice", () => {
    const profile = runCli(["--local", "--review-instructions=legacy.md"]);
    expect(profile.status).toBe(1);
    expect(profile.output).toContain("Use --instructions <path>");
    expect(profile.output).toContain("instead of replacing");
    const additional = runCli(["--local", "--additional-instructions", "legacy"]);
    expect(additional.status).toBe(1);
    expect(additional.output).toContain("Use --focus <text>");
  });

  it("rejects removed prompt-template override flags", () => {
    for (const legacyFlag of ["--prompt-file", "--prompt"]) {
      const result = runCli(["--local", legacyFlag, "legacy-value"]);

      expect(result.status).toBe(1);
      expect(result.output).toContain(`unknown option '${legacyFlag}'`);
    }
  });
});
