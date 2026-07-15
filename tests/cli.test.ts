import { execFileSync } from "node:child_process";
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
    expect(runCli(["--version"])).toEqual({ status: 0, output: "0.6.3-rc.1\n" });
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
});
