import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { logger } from "./utils/logger.js";
import { summarizeGitlabNotes } from "./gitlab.js";
import type { MrMetadata } from "./types.js";

// Resolve templates directory relative to this file (works in both src/ and dist/)
function getTemplatesDir(): string {
  const currentDir = dirname(fileURLToPath(import.meta.url));
  return resolve(currentDir, "..", "templates");
}

export function buildPrReviewPrompt(opts: {
  prUrl?: string;
  targetBranch?: string;
  diffBaseSha?: string | null;
  mrMetadata?: MrMetadata | null;
  customInstructions?: string | null;
  customPromptFile?: string | null;
  localMode?: boolean;
}): string {
  const {
    prUrl,
    targetBranch = "main",
    diffBaseSha,
    mrMetadata,
    customInstructions,
    customPromptFile,
    localMode = false,
  } = opts;

  // Step 1: Determine template (always tool submission; rendered to markdown post-hoc)
  let templateFile: string;
  if (customPromptFile) {
    templateFile = customPromptFile;
    logger.info(`Using custom prompt file: ${templateFile}`);
  } else {
    templateFile = resolve(getTemplatesDir(), "tool-review.md");
    logger.info("Using tool-based review template");
  }

  // Step 2: Load template
  let templateText: string;
  try {
    templateText = readFileSync(templateFile, "utf-8");
  } catch (err) {
    throw new Error(`Failed to load prompt template from ${templateFile}: ${err}`);
  }

  // Validate ref inputs to prevent shell injection via branch/SHA names.
  // Block shell metacharacters while allowing valid git ref chars (@, +, ~, ^, etc.)
  const dangerousChars = /[;\|`$&<>(){}\n\r\0\\!]/;
  if (dangerousChars.test(targetBranch)) {
    throw new Error(`Invalid target branch name: ${targetBranch}`);
  }
  if (diffBaseSha && dangerousChars.test(diffBaseSha)) {
    throw new Error(`Invalid diff base SHA: ${diffBaseSha}`);
  }

  // Prepare git diff commands
  const baseRef = diffBaseSha ?? (localMode ? targetBranch : `origin/${targetBranch}`);
  const prDiffCmd = `git --no-pager diff ${baseRef}${diffBaseSha ? " HEAD" : "...HEAD"} --name-only`;
  const gitDiffCmd = `git --no-pager diff ${baseRef}${diffBaseSha ? " HEAD" : "...HEAD"}`;

  if (diffBaseSha) {
    logger.info(`Using GitLab CI_MERGE_REQUEST_DIFF_BASE_SHA: ${diffBaseSha.slice(0, 8)}`);
  }

  // Diff explanation
  let diffExplanation: string;
  if (diffBaseSha) {
    diffExplanation =
      `**GitLab CI Advantage**: This uses GitLab's pre-calculated merge base SHA ` +
      `(\`CI_MERGE_REQUEST_DIFF_BASE_SHA\`), which matches exactly what the GitLab UI shows. ` +
      `This is more reliable than three-dot syntax because it handles force pushes, rebases, ` +
      `and messy histories correctly.`;
  } else {
    diffExplanation =
      `**Three-dot syntax** shows ONLY changes introduced on the source branch, ` +
      `excluding changes already on \`${targetBranch}\`.`;
  }

  // Step 3: Build MR sections
  const { contextSection, notesSection, reminderSection } = buildMrSections(mrMetadata);

  // Determine the display URL
  const displayUrl = localMode
    ? `local diff (against ${targetBranch})`
    : prUrl ?? "unknown";

  // Step 4: Interpolate
  let prompt = templateText
    .replace(/\{pr_url\}/g, displayUrl)
    .replace(/\{pr_diff_cmd\}/g, prDiffCmd)
    .replace(/\{git_diff_cmd\}/g, gitDiffCmd)
    .replace(/\{target_branch\}/g, targetBranch)
    .replace(/\{diff_explanation\}/g, diffExplanation)
    .replace(/\{mr_context_section\}/g, contextSection)
    .replace(/\{mr_notes_section\}/g, notesSection)
    .replace(/\{mr_reminder_section\}/g, reminderSection);

  // Step 5: Append custom instructions
  if (customInstructions) {
    prompt += `\n\n## Additional Instructions\n\n${customInstructions}\n`;
    logger.info("Appended custom instructions to prompt");
  }

  return prompt;
}

export function buildMrSections(mrMetadata?: MrMetadata | null): {
  contextSection: string;
  notesSection: string;
  reminderSection: string;
} {
  if (!mrMetadata) {
    return { contextSection: "", notesSection: "", reminderSection: "" };
  }

  const contextLines: string[] = [];

  if (mrMetadata.title) {
    contextLines.push(`- Title: ${mrMetadata.title}`);
  }

  const author =
    mrMetadata.author?.username ?? mrMetadata.author?.name;
  if (author) {
    contextLines.push(`- Author: @${author}`);
  }

  if (mrMetadata.source_branch && mrMetadata.target_branch) {
    contextLines.push(
      `- Branches: ${mrMetadata.source_branch} → ${mrMetadata.target_branch}`,
    );
  }

  if (mrMetadata.changes_count) {
    contextLines.push(`- Files changed: ${mrMetadata.changes_count}`);
  }

  const pipelineStatus = mrMetadata.pipeline?.status;
  const pipelineUrl = mrMetadata.pipeline?.web_url;
  if (pipelineStatus) {
    const statusText = pipelineStatus.replace(/_/g, " ");
    contextLines.push(
      pipelineUrl
        ? `- Pipeline: ${statusText} (${pipelineUrl})`
        : `- Pipeline: ${statusText}`,
    );
  }

  let labelNames = normalizeLabelNames(mrMetadata.label_details);
  if (labelNames.length === 0) {
    labelNames = normalizeLabelNames(mrMetadata.labels);
  }
  if (labelNames.length > 0) {
    contextLines.push(`- Labels: ${labelNames.join(", ")}`);
  }

  const description = (mrMetadata.description ?? "").trim();
  let descriptionSection = "";
  if (description) {
    descriptionSection =
      "**Author Description:**\n" + truncateBlock(description, 800);
  }

  let contextSection = "";
  if (contextLines.length > 0 || descriptionSection) {
    contextSection = "## MR Context\n" + contextLines.join("\n");
    if (descriptionSection) {
      contextSection += "\n\n" + descriptionSection;
    }
    contextSection += "\n";
  }

  let notesSection = "";
  const notesSummary = summarizeGitlabNotes(mrMetadata.Notes);
  if (notesSummary) {
    notesSection = `## Existing MR Notes\n${notesSummary}\n`;
  }

  let reminderSection = "";
  if (notesSummary) {
    reminderSection =
      "## Review Note Deduplication\n\n" +
      "The discussions above may already cover some issues. Before reporting a finding:\n" +
      "1. Check if it's already mentioned in existing notes\n" +
      "2. Only report if your finding is materially different or more specific\n" +
      "3. If an existing note is incorrect/outdated, explain why in your finding\n\n" +
      "Focus on discovering NEW issues not yet discussed.\n";
  }

  return { contextSection, notesSection, reminderSection };
}

function truncateBlock(text: string, limit: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= limit) return trimmed;
  return trimmed.slice(0, limit - 1).trimEnd() + "…";
}

export function normalizeLabelNames(
  rawLabels: unknown,
): string[] {
  if (!rawLabels) return [];

  const names: string[] = [];

  function addLabel(value: unknown): void {
    let name = "";
    if (typeof value === "string") {
      name = value.trim();
    } else if (typeof value === "object" && value !== null) {
      const labelValue = (value as Record<string, unknown>).name;
      if (typeof labelValue === "string") {
        name = labelValue.trim();
      }
    } else if (value != null) {
      name = String(value).trim();
    }
    if (name) names.push(name);
  }

  if (Array.isArray(rawLabels)) {
    for (const label of rawLabels) addLabel(label);
  } else {
    addLabel(rawLabels);
  }

  return names;
}
