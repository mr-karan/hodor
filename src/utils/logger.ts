import chalk from "chalk";

export type LogLevel = "debug" | "info" | "warn" | "error";

let currentLevel: LogLevel = "warn";
let buffering = false;
const bufferedLines: string[] = [];
const warningMessages: string[] = [];

const LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
}

/**
 * While buffering, debug and info lines are kept for drainBufferedLogs()
 * instead of printed. Warnings and errors always print live.
 */
export function setLogBuffering(enabled: boolean): void {
  buffering = enabled;
}

/** Return the buffered debug and info lines, oldest first, and clear them. */
export function drainBufferedLogs(): string[] {
  return bufferedLines.splice(0, bufferedLines.length);
}

/** Warning and error messages printed so far in this process, oldest first. */
export function getWarnings(): readonly string[] {
  return warningMessages;
}

function shouldLog(level: LogLevel): boolean {
  return LEVELS[level] >= LEVELS[currentLevel];
}

function timestamp(): string {
  return new Date().toISOString();
}

function emit(level: LogLevel, label: string, msg: string): void {
  if (!shouldLog(level)) return;
  if (level === "warn" || level === "error") warningMessages.push(`${level.toUpperCase()} ${msg}`);
  const line = `${chalk.gray(timestamp())} ${label} ${msg}`;
  if (buffering && (level === "debug" || level === "info")) {
    bufferedLines.push(line);
  } else {
    process.stderr.write(`${line}\n`);
  }
}

export const logger = {
  debug(msg: string): void {
    emit("debug", chalk.gray("DEBUG"), msg);
  },
  info(msg: string): void {
    emit("info", `${chalk.blue("INFO")} `, msg);
  },
  warn(msg: string): void {
    emit("warn", `${chalk.yellow("WARN")} `, msg);
  },
  error(msg: string): void {
    emit("error", chalk.red("ERROR"), msg);
  },
};
