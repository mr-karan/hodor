import { readFileSync, statSync, type Stats } from "node:fs";
import { resolve } from "node:path";
import { TextDecoder } from "node:util";
import { getTemplatePath } from "./templates.js";

export const MAX_REVIEW_INSTRUCTIONS_BYTES = 128 * 1024;

export function validateReviewInstructions(content: string, source = "review instructions"): string {
  if (Buffer.byteLength(content, "utf8") > MAX_REVIEW_INSTRUCTIONS_BYTES) {
    throw new Error(
      `${source} exceeds the ${MAX_REVIEW_INSTRUCTIONS_BYTES}-byte size limit`,
    );
  }

  if (content.trim().length === 0) {
    throw new Error(`${source} must not be empty or whitespace-only`);
  }

  return content;
}

export function loadReviewInstructionsFile(filePath: string, cwd = process.cwd()): string {
  const resolvedPath = resolve(cwd, filePath);
  let stat: Stats;

  try {
    stat = statSync(resolvedPath);
  } catch (error) {
    throw new Error(`Unable to read review instructions from ${resolvedPath}: ${error}`);
  }

  if (!stat.isFile()) {
    throw new Error(`Review instructions path is not a file: ${resolvedPath}`);
  }

  let bytes: Buffer;
  try {
    bytes = readFileSync(resolvedPath);
  } catch (error) {
    throw new Error(`Unable to read review instructions from ${resolvedPath}: ${error}`);
  }

  if (bytes.byteLength > MAX_REVIEW_INSTRUCTIONS_BYTES) {
    throw new Error(
      `Review instructions from ${resolvedPath} exceeds the ${MAX_REVIEW_INSTRUCTIONS_BYTES}-byte size limit`,
    );
  }

  let content: string;
  try {
    content = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`Review instructions from ${resolvedPath} must be valid UTF-8: ${error}`);
  }

  return validateReviewInstructions(content, `Review instructions from ${resolvedPath}`);
}

export function loadDefaultReviewInstructions(): string {
  return loadReviewInstructionsFile(getTemplatePath("default-review-instructions.md"));
}
