import { spawn } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, statSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, posix, relative, resolve, sep } from "node:path";
import { Type } from "typebox";
import {
  createFindToolDefinition,
  createLsToolDefinition,
  createReadToolDefinition,
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  defineTool,
  detectSupportedImageMimeTypeFromFile,
  formatSize,
  truncateHead,
  truncateLine,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { findExecutable } from "./utils/exec.js";

/**
 * Confined review tools.
 *
 * The reviewed change is untrusted input, so the model gets no shell. It sees
 * the change through `git_diff`, which serves the diff Hodor computed in
 * trusted code, and the repository through read/grep/find/ls, which accept
 * only paths that `git ls-files` lists. Untracked and ignored files, `.git/`,
 * symlinks that leave the tracked tree, and paths outside the repository are
 * rejected with an error that names the reason.
 */

/** Tool names the model sees in a normal review, in prompt order. */
export const REVIEW_TOOL_NAMES = ["git_diff", "read", "grep", "find", "ls"] as const;

const MAX_READ_FILE_BYTES = 10 * 1024 * 1024;
const MANIFEST_MAX_BYTES = 64 * 1024 * 1024;
const MANIFEST_TIMEOUT_MS = 60_000;
const GREP_TIMEOUT_MS = 30_000;
const GREP_MAX_OUTPUT_BYTES = 4 * 1024 * 1024;
const GREP_DEFAULT_LIMIT = 100;
const GREP_MAX_LINE_LENGTH = 500;
const GIT_STDERR_MAX_CHARS = 4096;

const TRACKED_SCOPE_NOTE =
  " Any file tracked by git in this repository is available, not only the changed files." +
  " Untracked or ignored files, .git, and paths outside the repository are rejected.";

// ---------------------------------------------------------------------------
// Git subprocess sandbox
// ---------------------------------------------------------------------------

const CHILD_PATH = "/usr/local/bin:/usr/bin:/bin";

// Options that keep git from running helpers or reaching the network. They go
// before the subcommand so they apply to every invocation.
const GIT_GLOBAL_ARGS = [
  "--no-pager",
  "-c", "core.pager=cat",
  "-c", "core.fsmonitor=false",
  "-c", "core.quotePath=false",
  "-c", "protocol.allow=never",
  // The global config is replaced by /dev/null, which also drops any
  // safe.directory entry the CI image relies on. Hodor already runs git in
  // this checkout with full trust, so accept it here too.
  "-c", "safe.directory=*",
];

export interface GitRunLimits {
  timeoutMs: number;
  maxBytes: number;
  /** Stop reading after this many newline-terminated records. */
  maxLines?: number;
  signal?: AbortSignal;
}

export interface GitRunResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  /** True when Hodor stopped git because a byte or line limit was reached. */
  stoppedEarly: boolean;
}

/**
 * Runs git with an absolute binary path, no shell, and a minimal environment.
 * The child sees none of Hodor's credentials: no tokens, no cloud or provider
 * keys, no user or system git config.
 */
export class GitSandbox {
  readonly env: NodeJS.ProcessEnv;
  readonly #gitPath: string;
  readonly #home: string;

  private constructor(gitPath: string, home: string) {
    this.#gitPath = gitPath;
    this.#home = home;
    this.env = {
      PATH: CHILD_PATH,
      LANG: "C",
      LC_ALL: "C",
      HOME: home,
      XDG_CONFIG_HOME: home,
      GIT_CONFIG_NOSYSTEM: "1",
      GIT_CONFIG_GLOBAL: "/dev/null",
      GIT_TERMINAL_PROMPT: "0",
      GIT_OPTIONAL_LOCKS: "0",
      GIT_NO_LAZY_FETCH: "1",
    };
  }

  static create(): GitSandbox {
    const gitPath = findExecutable("git");
    if (!gitPath) throw new Error("git was not found on PATH; review tools need it");
    return new GitSandbox(gitPath, mkdtempSync(join(tmpdir(), "hodor-tools-home-")));
  }

  run(cwd: string, args: string[], limits: GitRunLimits): Promise<GitRunResult> {
    return new Promise((resolvePromise, reject) => {
      if (limits.signal?.aborted) {
        reject(new Error("Operation aborted"));
        return;
      }
      const child = spawn(this.#gitPath, [...GIT_GLOBAL_ARGS, ...args], {
        cwd,
        env: this.env,
        shell: false,
        stdio: ["ignore", "pipe", "pipe"],
      });
      const chunks: Buffer[] = [];
      let size = 0;
      let lines = 0;
      let stderr = "";
      let stoppedEarly = false;
      let timedOut = false;
      let aborted = false;

      const stop = (): void => {
        if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
      };
      const timer = setTimeout(() => {
        timedOut = true;
        stop();
      }, limits.timeoutMs);
      const onAbort = (): void => {
        aborted = true;
        stop();
      };
      limits.signal?.addEventListener("abort", onAbort, { once: true });

      child.stdout.on("data", (chunk: Buffer) => {
        if (stoppedEarly) return;
        const room = limits.maxBytes - size;
        const part = chunk.length > room ? chunk.subarray(0, room) : chunk;
        chunks.push(part);
        size += part.length;
        if (limits.maxLines !== undefined) {
          for (const byte of part) if (byte === 0x0a) lines++;
        }
        if (chunk.length > room || (limits.maxLines !== undefined && lines >= limits.maxLines)) {
          stoppedEarly = true;
          stop();
        }
      });
      child.stderr.on("data", (chunk: Buffer) => {
        if (stderr.length < GIT_STDERR_MAX_CHARS) stderr += chunk.toString("utf-8");
      });
      child.on("error", (error) => {
        clearTimeout(timer);
        limits.signal?.removeEventListener("abort", onAbort);
        reject(error);
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        limits.signal?.removeEventListener("abort", onAbort);
        if (aborted) {
          reject(new Error("Operation aborted"));
        } else if (timedOut) {
          reject(new Error(`git ${args[0] ?? ""} timed out after ${limits.timeoutMs / 1000}s`));
        } else {
          resolvePromise({
            stdout: Buffer.concat(chunks).toString("utf-8"),
            stderr: stderr.slice(0, GIT_STDERR_MAX_CHARS).trim(),
            exitCode: code,
            stoppedEarly,
          });
        }
      });
    });
  }

  dispose(): void {
    rmSync(this.#home, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// Tracked-file manifest and path confinement
// ---------------------------------------------------------------------------

export type TrackedEntryKind = "file" | "directory";

export interface TrackedEntry {
  /** Real path on disk, after symlink resolution. */
  absolutePath: string;
  /** Repository-relative POSIX path of the real target ("" for the root). */
  relativePath: string;
  kind: TrackedEntryKind;
  size: number;
}

/** Resolve a model-supplied path the way Pi's tools do (`~`, `@` prefix, relative to root). */
export function resolveToolPath(root: string, input: string): string {
  let path = input.startsWith("@") ? input.slice(1) : input;
  if (path === "~") path = homedir();
  else if (path.startsWith("~/")) path = join(homedir(), path.slice(2));
  return resolve(root, path);
}

function hasGitSegment(relativePath: string): boolean {
  return relativePath.split("/").includes(".git");
}

/**
 * The set of paths `git ls-files` reports for the workspace, plus their
 * ancestor directories. Every tool access goes through `resolve()`.
 */
export class TrackedTree {
  readonly root: string;
  readonly realRoot: string;
  /** Tracked files, sorted. */
  readonly files: readonly string[];
  readonly #files: ReadonlySet<string>;
  readonly #dirs: ReadonlySet<string>;

  constructor(workspacePath: string, trackedFiles: Iterable<string>) {
    this.root = resolve(workspacePath);
    this.realRoot = realpathSync(this.root);
    const files = new Set<string>();
    const dirs = new Set<string>();
    for (const file of trackedFiles) {
      if (!file || hasGitSegment(file)) continue;
      files.add(file);
      for (let dir = posix.dirname(file); dir !== "."; dir = posix.dirname(dir)) dirs.add(dir);
    }
    this.#files = files;
    this.#dirs = dirs;
    this.files = [...files].sort();
  }

  static async load(workspacePath: string, git: GitSandbox): Promise<TrackedTree> {
    const root = resolve(workspacePath);
    const result = await git.run(root, ["ls-files", "-z", "--cached"], {
      timeoutMs: MANIFEST_TIMEOUT_MS,
      maxBytes: MANIFEST_MAX_BYTES,
    });
    if (result.stoppedEarly) {
      throw new Error(`git ls-files output exceeds ${formatSize(MANIFEST_MAX_BYTES)}`);
    }
    if (result.exitCode !== 0) {
      throw new Error(`git ls-files failed in ${root}: ${result.stderr || `exit code ${result.exitCode}`}`);
    }
    return new TrackedTree(root, result.stdout.split("\0"));
  }

  /** True when a repository-relative POSIX path is a tracked file or a directory holding one. */
  isTracked(relativePath: string): boolean {
    return relativePath === "" || this.#files.has(relativePath) || this.#dirs.has(relativePath);
  }

  /** Repository-relative POSIX path for an absolute path, or null when it is outside. */
  toRelative(absolutePath: string): string | null {
    for (const base of [this.root, this.realRoot]) {
      const rel = relative(base, absolutePath);
      if (rel === "") return "";
      if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) continue;
      return rel.split(sep).join("/");
    }
    return null;
  }

  /**
   * Resolve a path and require it to be tracked, inside the repository, and
   * of the expected kind. Both the path as given and its symlink target must
   * be tracked. Throws an Error that tells the model why access was refused.
   */
  resolve(inputPath: string, expected?: TrackedEntryKind): TrackedEntry {
    const absolutePath = resolve(this.root, inputPath);
    const rel = this.toRelative(absolutePath);
    if (rel === null) {
      throw new Error(
        `${inputPath} is outside the repository. Only files tracked by git in this repository are accessible.`,
      );
    }
    const display = rel === "" ? "." : rel;
    if (hasGitSegment(rel)) {
      throw new Error(`${display} is inside .git, which is not accessible. Use git_diff to see the change.`);
    }
    if (!this.isTracked(rel)) {
      throw new Error(
        `${display} is not a tracked file or directory in this repository (it is untracked, ignored, or does not exist). ` +
          "Use find or ls to locate tracked files.",
      );
    }

    let realPath: string;
    try {
      realPath = realpathSync(absolutePath);
    } catch {
      throw new Error(`${display} is tracked but missing from the checkout.`);
    }
    const realRel = this.toRelative(realPath);
    if (realRel === null || hasGitSegment(realRel) || !this.isTracked(realRel)) {
      throw new Error(`${display} is a symlink whose target is outside the repository or not tracked.`);
    }

    const stats = statSync(realPath);
    const kind: TrackedEntryKind | null = stats.isFile() ? "file" : stats.isDirectory() ? "directory" : null;
    const consistent = kind === "file"
      ? this.#files.has(realRel)
      : kind === "directory" && (realRel === "" || this.#dirs.has(realRel));
    if (kind === null || !consistent) {
      throw new Error(`${display} is not a tracked regular file or directory.`);
    }
    if (expected && kind !== expected) {
      throw new Error(`${display} is a ${kind}, not a ${expected}.`);
    }
    return { absolutePath: realPath, relativePath: realRel, kind, size: stats.size };
  }

  /** Read a tracked file, refusing files larger than the read limit. */
  readFile(inputPath: string): Buffer {
    const entry = this.resolve(inputPath, "file");
    if (entry.size > MAX_READ_FILE_BYTES) {
      throw new Error(
        `${entry.relativePath} is ${formatSize(entry.size)}, above the ${formatSize(MAX_READ_FILE_BYTES)} read limit.`,
      );
    }
    return readFileSync(entry.absolutePath);
  }
}

// ---------------------------------------------------------------------------
// Glob matching for `find`
// ---------------------------------------------------------------------------

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
}

/** Translate a glob (`*`, `**`, `?`, `[...]`, `{a,b}`) to a regular expression source. */
function globToRegExpSource(glob: string): string {
  let source = "";
  for (let i = 0; i < glob.length; i++) {
    const char = glob[i];
    if (char === "*") {
      if (glob[i + 1] === "*") {
        i++;
        if (glob[i + 1] === "/") {
          i++;
          source += "(?:.*/)?";
        } else {
          source += ".*";
        }
      } else {
        source += "[^/]*";
      }
    } else if (char === "?") {
      source += "[^/]";
    } else if (char === "[") {
      const end = glob.indexOf("]", i + 2);
      if (end === -1) {
        source += "\\[";
        continue;
      }
      let body = glob.slice(i + 1, end).replace(/\\/g, "\\\\");
      if (body.startsWith("!")) body = `^${body.slice(1)}`;
      source += `[${body}]`;
      i = end;
    } else if (char === "{") {
      const end = glob.indexOf("}", i + 1);
      if (end === -1) {
        source += "\\{";
        continue;
      }
      const alternatives = glob.slice(i + 1, end).split(",").map(globToRegExpSource);
      source += `(?:${alternatives.join("|")})`;
      i = end;
    } else {
      source += escapeRegExp(char);
    }
  }
  return source;
}

export function globToRegExp(glob: string): RegExp {
  return new RegExp(`^${globToRegExpSource(glob)}$`);
}

/**
 * Match tracked files under `searchPath` like Pi's fd-backed find: a pattern
 * without `/` matches the file name; a pattern with `/` matches the path
 * relative to the search directory at any depth.
 */
export function findTrackedFiles(
  tree: TrackedTree,
  pattern: string,
  searchPath: string,
  options: { ignore: string[]; limit: number },
): string[] {
  const dir = tree.resolve(searchPath, "directory");
  const prefix = dir.relativePath === "" ? "" : `${dir.relativePath}/`;
  const matchFullPath = pattern.includes("/");
  const fullPattern = pattern.startsWith("/")
    ? pattern.slice(1)
    : pattern.startsWith("**/") ? pattern : `**/${pattern}`;
  const matcher = globToRegExp(matchFullPath ? fullPattern : pattern);
  const ignores = options.ignore.map(globToRegExp);

  const results: string[] = [];
  for (const file of tree.files) {
    if (!file.startsWith(prefix)) continue;
    const rel = file.slice(prefix.length);
    if (ignores.some((ignore) => ignore.test(rel))) continue;
    if (!matcher.test(matchFullPath ? rel : posix.basename(rel))) continue;
    results.push(rel);
    if (results.length >= options.limit) break;
  }
  return results;
}

// ---------------------------------------------------------------------------
// Review diff for `git_diff`
// ---------------------------------------------------------------------------

export interface ChangedFileDiff {
  oldPath: string;
  newPath: string;
  text: string;
  additions: number;
  deletions: number;
}

/** Split a unified `git diff` into per-file sections, indexed by old and new path. */
export function splitReviewDiff(diff: string): { files: ChangedFileDiff[]; byPath: Map<string, ChangedFileDiff> } {
  const files: ChangedFileDiff[] = [];
  const byPath = new Map<string, ChangedFileDiff>();
  for (const section of diff.split(/(?=^diff --git )/m)) {
    const header = section.match(/^diff --git a\/(.*?) b\/(.*?)$/m);
    if (!header || header.index !== 0) continue;
    let additions = 0;
    let deletions = 0;
    for (const line of section.split("\n")) {
      if (line.startsWith("+") && !line.startsWith("+++")) additions++;
      else if (line.startsWith("-") && !line.startsWith("---")) deletions++;
    }
    const file: ChangedFileDiff = {
      oldPath: header[1],
      newPath: header[2],
      text: section.replace(/\n$/, ""),
      additions,
      deletions,
    };
    files.push(file);
    byPath.set(file.newPath, file);
    if (!byPath.has(file.oldPath)) byPath.set(file.oldPath, file);
  }
  return { files, byPath };
}

function formatChangedFileList(files: ChangedFileDiff[]): string {
  if (files.length === 0) return "No files changed in this review.";
  const lines = files.map((file) => {
    const name = file.oldPath === file.newPath ? file.newPath : `${file.oldPath} -> ${file.newPath}`;
    return `${name} (+${file.additions} -${file.deletions})`;
  });
  return `Changed files (${files.length}):\n${lines.join("\n")}\n\n` +
    "Call git_diff with path set to one of these files to see its diff.";
}

function toChangedFileKey(tree: TrackedTree, input: string): string {
  const trimmed = input.trim();
  if (isAbsolute(trimmed) || trimmed === "~" || trimmed.startsWith("~/")) {
    const rel = tree.toRelative(resolveToolPath(tree.root, trimmed));
    if (rel === null) throw new Error(`${input} is outside the repository.`);
    return rel;
  }
  const normalized = posix.normalize(trimmed.replace(/^@/, ""));
  if (normalized === ".." || normalized.startsWith("../")) {
    throw new Error(`${input} is outside the repository.`);
  }
  return normalized.replace(/^(\.\/)+/, "");
}

const gitDiffSchema = Type.Object({
  path: Type.Optional(Type.String({
    description: "Repository-relative path of one changed file. Omit to list the changed files.",
  })),
  offset: Type.Optional(Type.Number({
    description: "Diff line to start from (1-indexed), to continue a truncated file diff",
  })),
});

function createGitDiffTool(tree: TrackedTree, reviewDiff: string, inspectedFiles: Set<string>): ToolDefinition {
  const { files, byPath } = splitReviewDiff(reviewDiff);
  return defineTool({
    name: "git_diff",
    label: "git_diff",
    description:
      "Show the change under review. Without arguments, list the changed files with added and removed line counts. " +
      "With path, return the unified diff of that one changed file, including deleted and renamed files. " +
      "Hodor fixes the compared revisions; you cannot pass revisions or git options. " +
      `Output is truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB; use offset to continue.`,
    promptSnippet: "Show the changed files or one changed file's diff",
    parameters: gitDiffSchema,
    async execute(_toolCallId, { path, offset }) {
      if (path === undefined || path.trim() === "") {
        const truncation = truncateHead(formatChangedFileList(files));
        const notice = truncation.truncated ? "\n\n[File list truncated.]" : "";
        return { content: [{ type: "text", text: truncation.content + notice }], details: undefined };
      }

      const key = toChangedFileKey(tree, path);
      const file = byPath.get(key);
      if (!file) {
        throw new Error(
          `${path} is not a changed file in this review. Call git_diff without a path to list the changed files; ` +
            "use read for unchanged files.",
        );
      }

      const lines = file.text.split("\n");
      const start = offset !== undefined && offset > 1 ? Math.floor(offset) - 1 : 0;
      if (start >= lines.length) {
        throw new Error(`Offset ${offset} is beyond the end of this diff (${lines.length} lines).`);
      }
      const truncation = truncateHead(lines.slice(start).join("\n"));
      let text = truncation.content;
      if (truncation.firstLineExceedsLimit) {
        text = `[Diff line ${start + 1} exceeds ${formatSize(DEFAULT_MAX_BYTES)}. Use read on ${file.newPath} instead.]`;
      } else if (truncation.truncated) {
        const end = start + truncation.outputLines;
        text += `\n\n[Showing diff lines ${start + 1}-${end} of ${lines.length}. Use offset=${end + 1} to continue.]`;
      }
      if (!truncation.firstLineExceedsLimit) inspectedFiles.add(key);
      return { content: [{ type: "text", text }], details: undefined };
    },
  });
}

// ---------------------------------------------------------------------------
// grep over tracked files
// ---------------------------------------------------------------------------

const grepSchema = Type.Object({
  pattern: Type.String({ description: "Search pattern (regex or literal string)" }),
  path: Type.Optional(Type.String({ description: "Directory or file to search (default: repository root)" })),
  glob: Type.Optional(Type.String({ description: "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'" })),
  ignoreCase: Type.Optional(Type.Boolean({ description: "Case-insensitive search (default: false)" })),
  literal: Type.Optional(Type.Boolean({ description: "Treat pattern as literal string instead of regex (default: false)" })),
  context: Type.Optional(Type.Number({ description: "Number of lines to show before and after each match (default: 0)" })),
  limit: Type.Optional(Type.Number({ description: "Maximum number of matches to return (default: 100)" })),
});

interface GrepMatch {
  path: string;
  lineNumber: number;
  text: string;
}

/** Parse `git grep -z -n` records: `path NUL line NUL text LF`. */
function parseGrepRecords(stdout: string, limit: number): GrepMatch[] {
  const matches: GrepMatch[] = [];
  let pos = 0;
  while (pos < stdout.length && matches.length < limit) {
    const pathEnd = stdout.indexOf("\0", pos);
    const lineEnd = pathEnd === -1 ? -1 : stdout.indexOf("\0", pathEnd + 1);
    const textEnd = lineEnd === -1 ? -1 : stdout.indexOf("\n", lineEnd + 1);
    if (textEnd === -1) break;
    const lineNumber = Number(stdout.slice(pathEnd + 1, lineEnd));
    if (Number.isInteger(lineNumber)) {
      matches.push({ path: stdout.slice(pos, pathEnd), lineNumber, text: stdout.slice(lineEnd + 1, textEnd) });
    }
    pos = textEnd + 1;
  }
  return matches;
}

function toGlobPathspec(glob: string): string {
  const segments = glob.split("/");
  if (glob.startsWith(":") || glob.startsWith("/") || glob.includes("\0") || segments.includes("..")) {
    throw new Error(`Invalid glob ${glob}: use a pattern relative to the search path, such as '*.ts' or 'src/**/*.ts'.`);
  }
  return `:(glob)${glob.includes("/") ? glob : `**/${glob}`}`;
}

function createGrepTool(tree: TrackedTree, git: GitSandbox): ToolDefinition {
  return defineTool({
    name: "grep",
    label: "grep",
    description:
      "Search the contents of tracked files for a pattern (Perl-compatible regex, or literal). " +
      "Returns matching lines with file paths and line numbers. " +
      `Output is truncated to ${GREP_DEFAULT_LIMIT} matches or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). ` +
      `Long lines are truncated to ${GREP_MAX_LINE_LENGTH} chars.` + TRACKED_SCOPE_NOTE,
    promptSnippet: "Search tracked file contents for patterns",
    parameters: grepSchema,
    async execute(_toolCallId, { pattern, path: searchDir, glob, ignoreCase, literal, context, limit }, signal) {
      const target = tree.resolve(resolveToolPath(tree.root, searchDir || "."));
      const effectiveLimit = Math.max(1, Math.floor(limit ?? GREP_DEFAULT_LIMIT));
      const contextLines = context && context > 0 ? Math.floor(context) : 0;
      // Run from the searched directory so git limits the search to it and
      // prints paths relative to it, as Pi's grep does.
      const cwd = target.kind === "directory" ? target.absolutePath : dirname(target.absolutePath);
      const pathspecs = target.kind === "file"
        ? [`:(literal)${basename(target.absolutePath)}`]
        : glob ? [toGlobPathspec(glob)] : [];

      const search = (syntax: "-F" | "-P" | "-E"): Promise<GitRunResult> => git.run(cwd, [
        "grep", "--no-color", "-n", "-z", "-I", "--no-textconv", syntax,
        ...(ignoreCase ? ["-i"] : []),
        "-e", pattern, "--", ...pathspecs,
      ], {
        timeoutMs: GREP_TIMEOUT_MS,
        maxBytes: GREP_MAX_OUTPUT_BYTES,
        maxLines: effectiveLimit,
        signal,
      });

      let result = await search(literal ? "-F" : "-P");
      if (!literal && result.exitCode === 128 && /perl|pcre/i.test(result.stderr)) {
        // This git build lacks PCRE support; extended regex is the closest fallback.
        result = await search("-E");
      }
      if (!result.stoppedEarly && result.exitCode === 1) {
        return { content: [{ type: "text", text: "No matches found" }], details: undefined };
      }
      if (!result.stoppedEarly && result.exitCode !== 0) {
        throw new Error(result.stderr || `git grep exited with code ${result.exitCode}`);
      }

      const matches = parseGrepRecords(result.stdout, effectiveLimit);
      if (matches.length === 0) {
        return { content: [{ type: "text", text: "No matches found" }], details: undefined };
      }

      let linesTruncated = false;
      const clip = (text: string): string => {
        const { text: clipped, wasTruncated } = truncateLine(text.replace(/\r/g, ""), GREP_MAX_LINE_LENGTH);
        if (wasTruncated) linesTruncated = true;
        return clipped;
      };
      const fileLines = new Map<string, string[]>();
      const getFileLines = (path: string): string[] => {
        let lines = fileLines.get(path);
        if (!lines) {
          try {
            lines = tree.readFile(join(cwd, path)).toString("utf-8").replace(/\r\n?/g, "\n").split("\n");
          } catch {
            lines = [];
          }
          fileLines.set(path, lines);
        }
        return lines;
      };

      const outputLines: string[] = [];
      for (const match of matches) {
        if (contextLines === 0) {
          outputLines.push(`${match.path}:${match.lineNumber}: ${clip(match.text)}`);
          continue;
        }
        const lines = getFileLines(match.path);
        if (lines.length === 0) {
          outputLines.push(`${match.path}:${match.lineNumber}: (unable to read file)`);
          continue;
        }
        const first = Math.max(1, match.lineNumber - contextLines);
        const last = Math.min(lines.length, match.lineNumber + contextLines);
        for (let current = first; current <= last; current++) {
          const separator = current === match.lineNumber ? ":" : "-";
          outputLines.push(`${match.path}${separator}${current}${separator} ${clip(lines[current - 1] ?? "")}`);
        }
      }

      const truncation = truncateHead(outputLines.join("\n"), { maxLines: Number.MAX_SAFE_INTEGER });
      const notices: string[] = [];
      if (matches.length >= effectiveLimit) {
        notices.push(`${effectiveLimit} matches limit reached. Use limit=${effectiveLimit * 2} for more, or refine pattern`);
      } else if (result.stoppedEarly) {
        notices.push(`Search output exceeded ${formatSize(GREP_MAX_OUTPUT_BYTES)}. Refine the pattern or path`);
      }
      if (truncation.truncated) notices.push(`${formatSize(DEFAULT_MAX_BYTES)} limit reached`);
      if (linesTruncated) {
        notices.push(`Some lines truncated to ${GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines`);
      }
      const text = notices.length > 0 ? `${truncation.content}\n\n[${notices.join(". ")}]` : truncation.content;
      return { content: [{ type: "text", text }], details: undefined };
    },
  });
}

// ---------------------------------------------------------------------------
// read / ls / find on Pi's factories with confined operations
// ---------------------------------------------------------------------------

function createConfinedReadTool(tree: TrackedTree, inspectedFiles: Set<string>): ToolDefinition {
  const definition = createReadToolDefinition(tree.root, {
    operations: {
      access: async (absolutePath) => {
        tree.resolve(absolutePath, "file");
      },
      readFile: async (absolutePath) => tree.readFile(absolutePath),
      detectImageMimeType: async (absolutePath) =>
        detectSupportedImageMimeTypeFromFile(tree.resolve(absolutePath, "file").absolutePath),
    },
  });
  const execute: typeof definition.execute = async (...args) => {
    const result = await definition.execute(...args);
    inspectedFiles.add(tree.resolve(args[1].path, "file").relativePath);
    return result;
  };
  // Pi's TUI renderers are typed for the concrete schema and do not fit
  // ToolDefinition[]; Hodor has no TUI, so drop them.
  return {
    ...definition,
    execute,
    description: definition.description + TRACKED_SCOPE_NOTE,
    renderCall: undefined,
    renderResult: undefined,
  };
}

function createConfinedLsTool(tree: TrackedTree): ToolDefinition {
  const definition = createLsToolDefinition(tree.root, {
    operations: {
      exists: (absolutePath) => {
        tree.resolve(absolutePath);
        return true;
      },
      stat: (absolutePath) => {
        const entry = tree.resolve(absolutePath);
        return { isDirectory: () => entry.kind === "directory" };
      },
      readdir: (absolutePath) => {
        const entry = tree.resolve(absolutePath, "directory");
        const prefix = entry.relativePath === "" ? "" : `${entry.relativePath}/`;
        return readdirSync(entry.absolutePath).filter((name) => tree.isTracked(prefix + name));
      },
    },
  });
  return {
    ...definition,
    description: `${definition.description} Lists only tracked entries.${TRACKED_SCOPE_NOTE}`,
    renderCall: undefined,
    renderResult: undefined,
  };
}

function createConfinedFindTool(tree: TrackedTree): ToolDefinition {
  const definition = createFindToolDefinition(tree.root, {
    operations: {
      exists: (absolutePath) => {
        tree.resolve(absolutePath, "directory");
        return true;
      },
      glob: (pattern, searchPath, options) => findTrackedFiles(tree, pattern, searchPath, options),
    },
  });
  return {
    ...definition,
    description:
      "Search for tracked files by glob pattern. Returns matching file paths relative to the search directory. " +
      `Output is truncated to 1000 results or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first).` +
      TRACKED_SCOPE_NOTE,
    promptSnippet: "Find tracked files by glob pattern",
    renderCall: undefined,
    renderResult: undefined,
  };
}

// ---------------------------------------------------------------------------
// Toolset
// ---------------------------------------------------------------------------

export interface ReviewToolset {
  tree: TrackedTree;
  /** Paths whose code was served by read or git_diff, excluding internal reads. */
  inspectedFiles: ReadonlySet<string>;
  /** Definitions for REVIEW_TOOL_NAMES; same names replace Pi's built-ins. */
  definitions: ToolDefinition[];
  dispose(): void;
}

/**
 * Build the confined tools for one review session.
 *
 * `reviewDiff` is the full diff Hodor computed for the review; `git_diff`
 * serves it without running git again.
 */
export async function createReviewToolset(opts: {
  workspacePath: string;
  reviewDiff: string;
}): Promise<ReviewToolset> {
  const git = GitSandbox.create();
  try {
    const tree = await TrackedTree.load(opts.workspacePath, git);
    const inspectedFiles = new Set<string>();
    return {
      tree,
      inspectedFiles,
      definitions: [
        createGitDiffTool(tree, opts.reviewDiff, inspectedFiles),
        createConfinedReadTool(tree, inspectedFiles),
        createGrepTool(tree, git),
        createConfinedFindTool(tree),
        createConfinedLsTool(tree),
      ],
      dispose: () => git.dispose(),
    };
  } catch (error) {
    git.dispose();
    throw error;
  }
}
