import { readdir, readFile } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export interface PublicReleaseDenylist {
  terms: string[];
  allow?: Array<{
    path: string;
    line: number;
  }>;
}

export interface DisclosureMatch {
  path: string;
  line: number;
}

const EXCLUDED_DIRECTORIES = new Set([".git", "node_modules"]);

export function decodePublicReleaseDenylist(encoded: string): PublicReleaseDenylist {
  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.from(encoded, "base64").toString("utf8"));
  } catch {
    throw new Error("HODOR_PUBLIC_RELEASE_DENYLIST_B64 must contain base64-encoded JSON");
  }

  if (!isRecord(parsed) || !Array.isArray(parsed.terms)) {
    throw new Error("Public release denylist must contain a terms array");
  }

  const terms = parsed.terms
    .filter((term): term is string => typeof term === "string")
    .map((term) => term.trim())
    .filter((term) => term.length > 0);
  if (terms.length !== parsed.terms.length || terms.length === 0) {
    throw new Error("Public release denylist terms must be non-empty strings");
  }

  if (parsed.allow != null && !Array.isArray(parsed.allow)) {
    throw new Error("Public release denylist allow must be an array");
  }
  const allow = (parsed.allow ?? []).map((entry: unknown) => {
    if (
      !isRecord(entry) ||
      typeof entry.path !== "string" ||
      entry.path.length === 0 ||
      !Number.isInteger(entry.line) ||
      (entry.line as number) < 1
    ) {
      throw new Error("Public release denylist allow entries require a path and positive line number");
    }
    return { path: entry.path, line: entry.line as number };
  });

  return { terms, allow };
}

export async function findPublicReleaseDisclosures(
  root: string,
  denylist: PublicReleaseDenylist,
): Promise<DisclosureMatch[]> {
  const terms = denylist.terms.map((term) => term.toLowerCase());
  const allowedLocations = new Set(
    (denylist.allow ?? []).map(({ path, line }) => `${path.split(sep).join("/")}:${line}`),
  );
  const matches = new Map<string, DisclosureMatch>();

  for (const filePath of await collectFiles(resolve(root))) {
    const contents = await readFile(filePath);
    if (contents.includes(0)) continue;

    const displayPath = relative(root, filePath).split(sep).join("/");
    for (const [index, line] of contents.toString("utf8").split(/\r?\n/).entries()) {
      const lineNumber = index + 1;
      const key = `${displayPath}:${lineNumber}`;
      if (allowedLocations.has(key)) continue;

      const normalizedLine = line.toLowerCase();
      if (terms.some((term) => normalizedLine.includes(term))) {
        matches.set(key, { path: displayPath, line: lineNumber });
      }
    }
  }

  return [...matches.values()].sort((a, b) =>
    a.path.localeCompare(b.path) || a.line - b.line,
  );
}

async function collectFiles(directory: string): Promise<string[]> {
  const files: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && EXCLUDED_DIRECTORIES.has(entry.name)) continue;

    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...await collectFiles(path));
    } else if (entry.isFile()) {
      files.push(path);
    }
  }
  return files;
}


function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function main(): Promise<void> {
  const encoded = process.env.HODOR_PUBLIC_RELEASE_DENYLIST_B64?.trim();
  if (!encoded) {
    throw new Error("HODOR_PUBLIC_RELEASE_DENYLIST_B64 is required for public releases");
  }

  const matches = await findPublicReleaseDisclosures(
    process.cwd(),
    decodePublicReleaseDenylist(encoded),
  );
  if (matches.length > 0) {
    const locations = matches.map(({ path, line }) => `  ${path}:${line}`).join("\n");
    throw new Error(`Public release blocked by private-content denylist:\n${locations}`);
  }

  console.log("Public release disclosure scan passed");
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
