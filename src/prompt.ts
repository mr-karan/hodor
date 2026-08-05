import { readFileSync } from "node:fs";
import { getTemplatePath } from "./templates.js";
import { logger } from "./utils/logger.js";
import { summarizeGitlabNotes, summarizeHodorNotes } from "./gitlab.js";
import type { ReviewDiffMode } from "./review-diff.js";
import type { MrMetadata, Platform } from "./types.js";


export function buildPrReviewPrompt(opts: {
  prUrl: string;
  platform: Platform;
  targetBranch?: string;
  diffBaseSha?: string | null;
  mrMetadata?: MrMetadata | null;
  embeddedDiff?: string | null;
  previousReviewSha?: string | null;
  reviewDiffMode?: ReviewDiffMode;
  changedFiles?: string[];
  localMode?: boolean;
  singleTurn?: boolean;
}): string {
  const {
    prUrl,
    platform,
    targetBranch = "main",
    diffBaseSha,
    mrMetadata,
    embeddedDiff,
    previousReviewSha,
    reviewDiffMode,
    changedFiles = [],
    localMode = false,
    singleTurn = false,
  } = opts;

  let templateText: string;
  try {
    templateText = readFileSync(getTemplatePath("review-task.md"), "utf-8");
  } catch (error) {
    throw new Error(`Failed to load the review task template: ${error}`);
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
  if (previousReviewSha && !/^[a-f0-9]{40}$/.test(previousReviewSha)) {
    throw new Error(`Invalid previous review SHA: ${previousReviewSha}`);
  }

  // Prepare platform-specific commands
  let prDiffCmd: string;
  let gitDiffCmd: string;

  // Normal incremental mode uses a merge-base-aware three-dot diff. If history
  // was rewritten, compare the previously reviewed and current snapshots
  // directly; ancestry no longer describes the change the reviewer needs.
  if (previousReviewSha) {
    const separator = reviewDiffMode === "snapshot" ? " " : "...";
    prDiffCmd = `git --no-pager diff ${previousReviewSha}${separator}HEAD --name-only`;
    gitDiffCmd = `git --no-pager diff ${previousReviewSha}${separator}HEAD`;
    logger.info(`${reviewDiffMode === "snapshot" ? "Snapshot" : "Incremental"} review: diffing from ${previousReviewSha.slice(0, 8)} to HEAD`);
  } else if (localMode) {
    // Plain two-arg diff includes uncommitted (staged + unstaged) changes
    prDiffCmd = `git --no-pager diff ${targetBranch} --name-only`;
    gitDiffCmd = `git --no-pager diff ${targetBranch}`;
  } else if (platform === "github" || platform === "gitea") {
    prDiffCmd = `git --no-pager diff origin/${targetBranch}...HEAD --name-only`;
    gitDiffCmd = `git --no-pager diff origin/${targetBranch}...HEAD`;
  } else {
    // gitlab
    if (diffBaseSha) {
      prDiffCmd = `git --no-pager diff ${diffBaseSha} HEAD --name-only`;
      gitDiffCmd = `git --no-pager diff ${diffBaseSha} HEAD`;
      logger.info(`Using GitLab CI_MERGE_REQUEST_DIFF_BASE_SHA: ${diffBaseSha.slice(0, 8)}`);
    } else {
      prDiffCmd = `git --no-pager diff origin/${targetBranch}...HEAD --name-only`;
      gitDiffCmd = `git --no-pager diff origin/${targetBranch}...HEAD`;
    }
  }

  // Diff explanation
  let diffExplanation: string;
  if (previousReviewSha) {
    diffExplanation = reviewDiffMode === "snapshot"
      ? `**Snapshot delta mode**: The MR history was rewritten. This directly compares the last reviewed snapshot ` +
        `(commit \`${previousReviewSha.slice(0, 8)}\`) with the current HEAD; it does not imply ancestry.`
      : `**Incremental mode**: Showing only changes since the last hodor review ` +
        `(commit \`${previousReviewSha.slice(0, 8)}\`).`;
  } else if (diffBaseSha) {
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

  // The fast path only makes sense when the diff is already in context; without
  // it the reviewer has no way to see the change at all.
  const oneTurn = singleTurn && Boolean(embeddedDiff);

  // Step 3b: Build incremental review section
  let incrementalSection = "";
  if (previousReviewSha) {
    incrementalSection =
      `## ${reviewDiffMode === "snapshot" ? "Snapshot Delta" : "Incremental Review"} Mode\n\n` +
      `This is a follow-up review. A previous hodor review was done at commit \`${previousReviewSha.slice(0, 8)}\`. ` +
      (reviewDiffMode === "snapshot"
        ? "The branch history was rewritten, so the diff below compares that reviewed snapshot directly with the current HEAD. "
        : "The diff below shows ONLY changes since that review. ") +
      "Your job is to review that delta, not the whole MR again.\n\n" +
      "Rules for incremental reviews:\n" +
      "1. Only report findings introduced or still affected by the new delta.\n" +
      "2. Do not re-report issues that are already mentioned in existing notes unless the new delta changes the same code and the issue remains newly relevant.\n" +
      "3. If the delta is small and self-contained, decide from the embedded diff and submit the review without broad repository exploration.\n" +
      (oneTurn
        ? "4. No file-inspection tools are available; if a mechanical change like a route/path/string rename leaves a compatibility question you cannot settle from the diff, do not report it.\n"
        : "4. For mechanical changes like route/path/string renames, verify the direct call sites or tests only when the diff itself leaves a concrete compatibility question.\n") +
      "5. If the delta does not produce a qualifying finding under the selected review instructions, submit no findings.\n\n";
  }

  // Step 3c: Build conditional sections based on whether diff is embedded
  let embeddedDiffSection: string;
  let diffFetchInstructions: string;
  let reviewProcessSection: string;
  let startInstruction: string;
  let runtimeToolsSection: string;

  if (embeddedDiff) {
    const changedFileManifest = changedFiles.length > 0
      ? `\nChanged files (${changedFiles.length}):\n${changedFiles.map((file) => `- \`${file}\``).join("\n")}\n`
      : "";
    embeddedDiffSection =
      "## Full Diff (Pre-fetched)\n\n" +
      "The complete diff for this PR is provided below. Analyze it directly. " +
      "Do not run another command to list changed files. " +
      (oneTurn
        ? "This diff is small and self-contained; it is the complete basis for your review.\n"
        : "Use `read` or `grep` only if you need additional file context beyond what the diff shows.\n") +
      changedFileManifest + "\n" +
      "````diff\n" + embeddedDiff + "\n````\n";

    diffFetchInstructions =
      "## Review the Diff Above\n\n" +
      "### Critical Rules\n" +
      "- ONLY review files that appear in the diff above\n" +
      "- ONLY analyze actual code changes (+ and - lines in the diff)\n" +
      "- NEVER review files not in the diff\n" +
      "- NEVER flag \"files will be deleted when merging\" (outdated branch)\n" +
      "- NEVER flag \"dependency version downgrade\" (branch not rebased)\n" +
      `- NEVER compare entire codebase to ${targetBranch} - DIFF ONLY\n`;

    if (oneTurn) {
      reviewProcessSection =
        "## Review Process\n\n" +
        "1. Analyze the embedded diff above thoroughly\n" +
        "2. Call `submit_review` in this same turn with your findings\n\n" +
        "No file-inspection tools are available for this review. Base every finding on the diff above. " +
        "If a potential issue cannot be established from the diff alone, do not report it.\n";

      startInstruction =
        "Analyze the diff above and call `submit_review` now, in this turn.";
    } else {
      reviewProcessSection =
        "## Review Process\n\n" +
        "1. Analyze the embedded diff above thoroughly\n" +
        "2. Use `grep` to search for patterns when needed\n" +
        "3. Use bounded line-range reads when surrounding context is essential; avoid reading entire large files\n" +
        "4. Do not repeat a diff, grep, or read operation whose result is already in context\n" +
        "5. Submit your review using `submit_review`\n";

      startInstruction = previousReviewSha
        ? "Analyze only the incremental diff provided above. If it is self-contained, submit your review without extra tool calls."
        : "Analyze the diff provided above, then submit your review using `submit_review`.";
    }
  } else {
    embeddedDiffSection = "";

    diffFetchInstructions =
      "## Step 1: List Changed Files (MANDATORY FIRST STEP)\n\n" +
      "**Run this command FIRST to get the list of changed files:**\n" +
      "```bash\n" + prDiffCmd + "\n```\n\n" +
      "This lists ONLY the filenames changed in this PR. **Do NOT dump the entire diff here** - " +
      "you'll inspect each file individually in Step 2. Only review files that appear in this output.\n\n" +
      "## Step 2: Review Changed Files Only\n\n" +
      "### Critical Rules\n" +
      "- ONLY review files that appear in the diff from Step 1\n" +
      "- ONLY analyze actual code changes (+ and - lines in the diff)\n" +
      "- Use the most reliable diff command: `" + gitDiffCmd + "`\n" +
      "- NEVER review files not in the diff\n" +
      "- NEVER flag \"files will be deleted when merging\" (outdated branch)\n" +
      "- NEVER flag \"dependency version downgrade\" (branch not rebased)\n" +
      `- NEVER compare entire codebase to ${targetBranch} - DIFF ONLY\n\n` +
      "### Git Diff Command\n\n" +
      "**Most reliable command to see changes:**\n" +
      "```bash\n" + gitDiffCmd + "\n```\n\n" +
      diffExplanation;

    reviewProcessSection =
      "## Review Process\n\n" +
      "**Efficient Sequential Workflow:**\n\n" +
      `1. **List files first**: Run \`${prDiffCmd}\` to get the list of changed files (NOT full diff)\n` +
      `2. **Per-file analysis**: For each file, run \`${gitDiffCmd} -- path/to/file\` to see its specific changes\n` +
      "3. **Batch pattern search**: Use `grep` across multiple files to find common bug patterns (null, undefined, TODO, FIXME, etc.)\n" +
      "4. **Selective deep dive**: Only use `read` to read full file context when the diff alone is insufficient\n" +
      "5. **Group related files**: Analyze related files together (e.g., implementation + tests, interfaces + implementations)\n" +
      "6. **Avoid redundancy**: Don't re-read files unnecessarily; make decisions based on diff context\n";

    startInstruction =
      `Start by running \`${prDiffCmd}\` to list the changed files, then analyze each file individually using \`${gitDiffCmd} -- path/to/file\`.`;
  }

  // The fast path leaves submit_review as the only tool, so advertising the
  // inspection tools would invite calls that cannot succeed.
  runtimeToolsSection = oneTurn
    ? "## Runtime Tools\n\n" +
      "- `submit_review` submits the completed review. It is the only tool available for this review.\n"
    : "## Runtime Tools\n\n" +
      `- \`${prDiffCmd}\` lists the changed files when a diff is not embedded.\n` +
      `- \`${gitDiffCmd} -- path/to/file\` shows the delta for one changed file.\n` +
      "- `read` provides bounded surrounding context.\n" +
      "- `grep` searches for directly relevant code and contracts.\n" +
      "- `submit_review` submits the completed review.\n";

  return templateText
    .replace(/\{pr_url\}/g, prUrl)
    .replace(/\{pr_diff_cmd\}/g, prDiffCmd)
    .replace(/\{git_diff_cmd\}/g, gitDiffCmd)
    .replace(/\{mr_context_section\}/g, contextSection)
    .replace(/\{mr_notes_section\}/g, notesSection)
    .replace(/\{mr_reminder_section\}/g, reminderSection)
    .replace(/\{incremental_section\}/g, incrementalSection)
    .replace(/\{embedded_diff_section\}/g, embeddedDiffSection)
    .replace(/\{diff_fetch_instructions\}/g, diffFetchInstructions)
    .replace(/\{runtime_tools_section\}/g, runtimeToolsSection)
    .replace(/\{review_process_section\}/g, reviewProcessSection)
    .replace(/\{start_instruction\}/g, startInstruction);
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
  const humanNotesSummary = summarizeGitlabNotes(mrMetadata.Notes);
  const hodorNotesSummary = summarizeHodorNotes(mrMetadata.Notes);
  if (humanNotesSummary) {
    notesSection += `## Existing Human MR Notes\n${humanNotesSummary}\n`;
  }
  if (hodorNotesSummary) {
    notesSection +=
      `## Prior Hodor Reviews (deduplication only)\n${hodorNotesSummary}\n` +
      "Use this history only to avoid repeating the same finding. Re-check the current diff independently.\n";
  }

  let reminderSection = "";
  if (humanNotesSummary || hodorNotesSummary) {
    reminderSection =
      "## Review Note Deduplication\n\n" +
      "The human notes and prior Hodor reviews above may already cover some issues. Before reporting a finding:\n" +
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
