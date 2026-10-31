import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  MAX_REVIEW_INSTRUCTIONS_BYTES,
  loadDefaultReviewInstructions,
  loadReviewInstructionsFile,
  loadReviewProfileInstructions,
  validateReviewInstructions,
} from "../src/review-instructions.js";

const tempDirs: string[] = [];

function makeTempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "hodor-review-instructions-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("review instruction profiles", () => {
  it("adds no instructions by default", () => {
    expect(loadReviewProfileInstructions()).toEqual([]);
  });

  it("loads the exact bundled security profile, not a working-directory file", () => {
    const expected = readFileSync(new URL("../templates/security-review-instructions.md", import.meta.url), "utf8");
    expect(loadReviewProfileInstructions("security")).toEqual([expected]);
    expect(expected).toContain("Report only security vulnerabilities");
  });

  it.each(["", "default", "Security", "../security", "unknown"])("rejects invalid profile %j", (profile) => {
    expect(() => loadReviewProfileInstructions(profile)).toThrow("--profile must be one of: security");
  });

  it("loads the bundled default profile", () => {
    const instructions = loadDefaultReviewInstructions();

    expect(instructions).toContain("Identify production bugs introduced by the proposed change");
  });

  it("loads a custom profile verbatim relative to the supplied working directory", () => {
    const dir = makeTempDir();
    const profile = "\n# Security review\nKeep this whitespace.  \n";
    writeFileSync(join(dir, "security.md"), profile);

    expect(loadReviewInstructionsFile("security.md", dir)).toBe(profile);
  });

  it("rejects whitespace-only profile files", () => {
    const dir = makeTempDir();
    writeFileSync(join(dir, "empty.md"), " \n\t ");

    expect(() => loadReviewInstructionsFile("empty.md", dir)).toThrow(/must not be empty or whitespace-only/);
  });

  it("rejects profiles larger than the byte limit", () => {
    const dir = makeTempDir();
    writeFileSync(join(dir, "large.md"), Buffer.alloc(MAX_REVIEW_INSTRUCTIONS_BYTES + 1, "a"));

    expect(() => loadReviewInstructionsFile("large.md", dir)).toThrow(/size limit/);
  });

  it("rejects profile files that are not valid UTF-8", () => {
    const dir = makeTempDir();
    writeFileSync(join(dir, "invalid.md"), Buffer.from([0xc3, 0x28]));

    expect(() => loadReviewInstructionsFile("invalid.md", dir)).toThrow(/valid UTF-8/);
  });

  it("rejects a directory where a profile file is required", () => {
    const dir = makeTempDir();
    mkdirSync(join(dir, "profile-directory"));

    expect(() => loadReviewInstructionsFile("profile-directory", dir)).toThrow(/not a file/);
  });

  it("preserves non-empty instruction content during validation", () => {
    const content = "  retain leading and trailing whitespace  \n";

    expect(validateReviewInstructions(content)).toBe(content);
  });
});
