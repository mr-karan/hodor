import { readFileSync, statSync, type Stats } from "node:fs";
import { resolve } from "node:path";
import { TextDecoder } from "node:util";
import { getTemplatePath } from "./templates.js";

export const MAX_REVIEW_INSTRUCTIONS_BYTES = 128 * 1024;
export const MAX_TOTAL_INSTRUCTIONS_BYTES = 256 * 1024;

export function validateInstructionsBudget(contents: readonly string[]): void {
  const bytes = contents.reduce((total, content) => total + Buffer.byteLength(content, "utf8"), 0);
  if (bytes > MAX_TOTAL_INSTRUCTIONS_BYTES) {
    throw new Error(`Combined instructions exceed the ${MAX_TOTAL_INSTRUCTIONS_BYTES}-byte size limit`);
  }
}

export function validateInstructionSize(bytes: number, source: string): void {
  if (bytes > MAX_REVIEW_INSTRUCTIONS_BYTES) {
    throw new Error(`${source} exceeds the ${MAX_REVIEW_INSTRUCTIONS_BYTES}-byte size limit`);
  }
}

export function validateReviewInstructions(content: string, source = "review instructions"): string {
  validateInstructionSize(Buffer.byteLength(content, "utf8"), source);

  if (content.trim().length === 0) {
    throw new Error(`${source} must not be empty or whitespace-only`);
  }

  return content;
}

export function loadReviewInstructionsFile(filePath: string, cwd = process.cwd()): string {
  const resolvedPath = resolve(cwd, filePath);
  const source = `Review instructions from ${resolvedPath}`;
  let stat: Stats;

  try {
    stat = statSync(resolvedPath);
  } catch (error) {
    throw new Error(`Unable to read review instructions from ${resolvedPath}: ${error}`);
  }

  if (!stat.isFile()) {
    throw new Error(`Review instructions path is not a file: ${resolvedPath}`);
  }
  validateInstructionSize(stat.size, source);

  let bytes: Buffer;
  try {
    bytes = readFileSync(resolvedPath);
  } catch (error) {
    throw new Error(`Unable to read review instructions from ${resolvedPath}: ${error}`);
  }

  validateInstructionSize(bytes.byteLength, source);

  let content: string;
  try {
    content = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${source} must be valid UTF-8: ${error}`);
  }

  return validateReviewInstructions(content, source);
}

export function loadDefaultReviewInstructions(): string {
  return loadReviewInstructionsFile(getTemplatePath("default-review-instructions.md"));
}
