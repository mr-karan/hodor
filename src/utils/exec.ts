import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export interface ExecResult {
  stdout: string;
  stderr: string;
}

export interface ExecOptions {
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  input?: string;
}

export async function exec(
  cmd: string,
  args: string[],
  opts?: ExecOptions,
): Promise<ExecResult> {
  if (typeof opts?.input === "string") {
    return new Promise<ExecResult>((resolve, reject) => {
      const child = spawn(cmd, args, {
        cwd: opts.cwd,
        env: opts.env ?? process.env,
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk: Buffer | string) => {
        stdout += chunk.toString();
      });
      child.stderr.on("data", (chunk: Buffer | string) => {
        stderr += chunk.toString();
      });
      child.on("error", (error) => {
        reject(error);
      });
      child.on("close", (code, signal) => {
        if (code === 0) {
          resolve({ stdout, stderr });
          return;
        }

        const parts: string[] = [`Command failed: ${cmd} ${args.join(" ")}`];
        if (stderr.trim()) {
          parts.push(`stderr:\n${stderr.trim()}`);
        }
        if (stdout.trim()) {
          parts.push(`stdout:\n${stdout.trim()}`);
        }
        if (signal) {
          parts.push(`signal: ${signal}`);
        }
        const error = new Error(parts.join("\n"));
        reject(error);
      });

      child.stdin.write(opts.input);
      child.stdin.end();
    });
  }

  const { stdout, stderr } = await execFileAsync(cmd, args, {
    cwd: opts?.cwd,
    env: opts?.env ?? process.env,
    maxBuffer: 50 * 1024 * 1024, // 50MB
  });
  return { stdout, stderr };
}

export async function execJson<T = Record<string, unknown>>(
  cmd: string,
  args: string[],
  opts?: ExecOptions,
): Promise<T> {
  const { stdout } = await exec(cmd, args, opts);
  return JSON.parse(stdout.trim()) as T;
}
