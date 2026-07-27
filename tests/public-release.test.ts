import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  decodePublicReleaseDenylist,
  findPublicReleaseDisclosures,
} from "../scripts/check-public-release.js";

const tempDirs: string[] = [];

function makeTempDir(): string {
  const directory = mkdtempSync(join(tmpdir(), "hodor-public-release-"));
  tempDirs.push(directory);
  return directory;
}

afterEach(() => {
  for (const directory of tempDirs.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("public release disclosure scan", () => {
  it("decodes a validated base64 configuration", () => {
    const encoded = Buffer.from(JSON.stringify({
      terms: ["private.example"],
      allow: [{ path: "README.md", line: 2 }],
    })).toString("base64");

    expect(decodePublicReleaseDenylist(encoded)).toEqual({
      terms: ["private.example"],
      allow: [{ path: "README.md", line: 2 }],
    });
  });

  it("finds case-insensitive matches without printing matched content", async () => {
    const directory = makeTempDir();
    mkdirSync(join(directory, "src"));
    writeFileSync(join(directory, "src", "config.ts"), "const host = 'PRIVATE.EXAMPLE';\n");
    writeFileSync(join(directory, "safe.ts"), "export const safe = true;\n");

    await expect(findPublicReleaseDisclosures(directory, {
      terms: ["private.example"],
    })).resolves.toEqual([{ path: "src/config.ts", line: 1 }]);
  });

  it("honors exact allowed locations and ignores dependency metadata", async () => {
    const directory = makeTempDir();
    mkdirSync(join(directory, "node_modules"));
    writeFileSync(join(directory, "README.md"), "public attribution: private.example\n");
    writeFileSync(join(directory, "node_modules", "metadata.txt"), "private.example\n");

    await expect(findPublicReleaseDisclosures(directory, {
      terms: ["private.example"],
      allow: [{ path: "README.md", line: 1 }],
    })).resolves.toEqual([]);
  });
});
