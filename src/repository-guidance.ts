import { execFile } from "node:child_process";
import { posix } from "node:path";
import { promisify, TextDecoder } from "node:util";
import { exec } from "./utils/exec.js";
import { MAX_REVIEW_INSTRUCTIONS_BYTES, MAX_TOTAL_INSTRUCTIONS_BYTES, validateInstructionSize } from "./review-instructions.js";

const execFileAsync = promisify(execFile);

export interface RepositoryGuidanceFile {
  path: string;
  directory: string;
  content: string;
}

interface TreeEntry {
  kind: "file" | "symlink";
  object: string;
  bytes: number;
}

export async function resolveGuidanceSnapshot(opts: {
  workspacePath: string;
  targetBranch: string;
  localMode: boolean;
}): Promise<string> {
  const { workspacePath, targetBranch, localMode } = opts;
  const ref = localMode ? targetBranch : `refs/remotes/origin/${targetBranch}`;
  const resolveCommit = async (commitRef: string): Promise<string> => {
    const { stdout } = await exec("git", ["rev-parse", "--verify", "--end-of-options", `${commitRef}^{commit}`], { cwd: workspacePath });
    const sha = stdout.trim();
    if (!/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(sha)) throw new Error("Invalid guidance snapshot commit");
    return sha;
  };
  try {
    return await resolveCommit(ref);
  } catch {
    if (!localMode) {
      try {
        await exec("git", ["fetch", "--quiet", "--no-tags", "--", "origin", targetBranch], { cwd: workspacePath });
        return await resolveCommit("FETCH_HEAD");
      } catch {
        // Never substitute the MR's HEAD or its previous reviewed commit.
      }
    }
    throw new Error(`Cannot resolve repository guidance snapshot from ${ref}`);
  }
}

async function readBlob(workspacePath: string, object: string, path: string): Promise<string> {
  let bytes: Buffer;
  try {
    ({ stdout: bytes } = await execFileAsync("git", ["cat-file", "blob", object], {
      cwd: workspacePath,
      encoding: "buffer",
      maxBuffer: MAX_REVIEW_INSTRUCTIONS_BYTES,
      timeout: 30_000,
    }));
  } catch {
    throw new Error(`Cannot read repository guidance: ${path}`);
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(`Repository guidance must be valid UTF-8: ${path}`);
  }
}

async function readGuidanceFile(
  workspacePath: string,
  tree: ReadonlyMap<string, TreeEntry>,
  path: string,
): Promise<string> {
  const visited = new Set<string>();
  let currentPath = path;
  for (let hop = 0; hop < 16; hop++) {
    if (visited.has(currentPath)) throw new Error(`Repository guidance symlink cycle: ${path}`);
    visited.add(currentPath);
    const entry = tree.get(currentPath);
    if (!entry) {
      throw new Error(`Repository guidance is not a tracked regular file: ${path}`);
    }
    validateInstructionSize(entry.bytes, `Repository guidance from ${path}`);
    const content = await readBlob(workspacePath, entry.object, currentPath);
    if (entry.kind === "file") return content;
    const target = posix.normalize(posix.join(posix.dirname(currentPath), content));
    if (posix.isAbsolute(content) || target === ".." || target.startsWith("../")) {
      throw new Error(`Repository guidance symlink leaves the repository: ${path}`);
    }
    currentPath = target;
  }
  throw new Error(`Repository guidance symlink chain is too long: ${path}`);
}

async function loadGuidanceTree(workspacePath: string, snapshotSha: string): Promise<Map<string, TreeEntry>> {
  const { stdout } = await exec("git", ["ls-tree", "-r", "-z", "-l", snapshotSha], { cwd: workspacePath });
  const tree = new Map<string, TreeEntry>();
  for (const record of stdout.split("\0")) {
    if (!record) continue;
    const match = record.match(/^(\d+) (\w+) ([a-f0-9]+) +([\d-]+)\t([\s\S]+)$/);
    if (!match) throw new Error("Invalid repository guidance tree output");
    if (match[2] === "blob") {
      tree.set(match[5], { kind: match[1] === "120000" ? "symlink" : "file", object: match[3], bytes: Number(match[4]) });
    }
  }
  return tree;
}

function getGuidanceDirectories(changedPaths: readonly string[]): string[] {
  const directories = new Set<string>([""]);
  for (const path of changedPaths) {
    if (posix.isAbsolute(path) || path.split("/").includes("..")) {
      throw new Error("Changed guidance path leaves the repository");
    }
    let directory = posix.dirname(path);
    while (directory !== ".") {
      directories.add(directory);
      directory = posix.dirname(directory);
    }
  }
  return [...directories].sort((a, b) => a.split("/").length - b.split("/").length || a.localeCompare(b));
}

export async function loadRepositoryGuidance(opts: {
  workspacePath: string;
  snapshotSha: string;
  changedPaths: readonly string[];
}): Promise<RepositoryGuidanceFile[]> {
  const { workspacePath, snapshotSha, changedPaths } = opts;
  if (!/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(snapshotSha)) throw new Error("Invalid guidance snapshot commit");
  const directories = getGuidanceDirectories(changedPaths);
  const tree = await loadGuidanceTree(workspacePath, snapshotSha);
  const files: RepositoryGuidanceFile[] = [];
  let totalBytes = 0;
  for (const directory of directories) {
    const agentsPath = posix.join(directory, "AGENTS.md");
    const claudePath = posix.join(directory, "CLAUDE.md");
    const path = tree.has(agentsPath) ? agentsPath : tree.has(claudePath) ? claudePath : null;
    if (!path) continue;
    const content = await readGuidanceFile(workspacePath, tree, path);
    totalBytes += Buffer.byteLength(content, "utf8");
    if (totalBytes > MAX_TOTAL_INSTRUCTIONS_BYTES) {
      throw new Error(`Combined repository guidance exceeds the ${MAX_TOTAL_INSTRUCTIONS_BYTES}-byte size limit`);
    }
    if (content.trim()) files.push({ path, directory, content });
  }
  return files;
}
