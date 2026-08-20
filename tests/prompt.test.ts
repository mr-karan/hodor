import { describe, it, expect } from "vitest";
import {
  buildMrSections,
  buildPrReviewPrompt,
  normalizeLabelNames,
} from "../src/prompt.js";
import { loadDefaultReviewInstructions } from "../src/review-instructions.js";
import { buildReviewSystemPrompt } from "../src/system-prompt.js";
describe("buildMrSections", () => {
  it("handles string labels", () => {
    const metadata = {
      title: "Add string labels support",
      labels: ["bug", "gitlab"],
    };

    const { contextSection } = buildMrSections(metadata);
    expect(contextSection).toContain("- Labels: bug, gitlab");
  });

  it("prefers label_details when available", () => {
    const metadata = {
      title: "Prefer detailed labels",
      labels: ["fallback"],
      label_details: [{ name: "frontend" }, { name: "regression" }],
    };

    const { contextSection } = buildMrSections(metadata);
    expect(contextSection).toContain("- Labels: frontend, regression");
  });

  it("returns empty strings when no metadata", () => {
    const { contextSection, notesSection, reminderSection } =
      buildMrSections(null);
    expect(contextSection).toBe("");
    expect(notesSection).toBe("");
    expect(reminderSection).toBe("");
  });

  it("includes author and branches", () => {
    const metadata = {
      title: "Test PR",
      author: { username: "testuser" },
      source_branch: "feature",
      target_branch: "main",
    };

    const { contextSection } = buildMrSections(metadata);
    expect(contextSection).toContain("- Author: @testuser");
    expect(contextSection).toContain("- Branches: feature → main");
  });

  it("labels prior Hodor output as deduplication-only context", () => {
    const { notesSection } = buildMrSections({
      Notes: [
        {
          body: "<!-- hodor:sha:1111111111111111111111111111111111111111 -->\n<!-- hodor-review -->\nPrior finding with enough text",
          author: { username: "hodor" },
        },
      ],
    });

    expect(notesSection).toContain("Prior Hodor Reviews (deduplication only)");
    expect(notesSection).toContain("Re-check the current diff independently");
  });
});

describe("normalizeLabelNames", () => {
  it("handles string labels", () => {
    expect(normalizeLabelNames(["bug", "feature"])).toEqual([
      "bug",
      "feature",
    ]);
  });

  it("handles dict labels", () => {
    expect(
      normalizeLabelNames([{ name: "bug" }, { name: "feature" }]),
    ).toEqual(["bug", "feature"]);
  });

  it("returns empty for null/undefined", () => {
    expect(normalizeLabelNames(null)).toEqual([]);
    expect(normalizeLabelNames(undefined)).toEqual([]);
  });
});

describe("buildPrReviewPrompt", () => {
  it("uses a direct snapshot diff after rewritten history", () => {
    const sha = "1".repeat(40);
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
      previousReviewSha: sha,
      reviewDiffMode: "snapshot",
      embeddedDiff: "diff --git a/src/a.ts b/src/a.ts\n+const ok = true;",
      changedFiles: ["src/a.ts"],
    });

    expect(prompt).toContain(`git --no-pager diff ${sha} HEAD`);
    expect(prompt).not.toContain(`${sha}...HEAD`);
    expect(prompt).toContain("Snapshot Delta Mode");
    expect(prompt).toContain("Changed files (1)");
    expect(prompt).toContain("Do not run another command to list changed files");
  });

  it("uses the current GitLab MR base after a rebased follow-up review", () => {
    const previousReviewSha = "1".repeat(40);
    const currentMrBaseSha = "2".repeat(40);
    const prompt = buildPrReviewPrompt({
      prUrl: "https://gitlab.com/acme/hodor/-/merge_requests/42",
      platform: "gitlab",
      targetBranch: "main",
      diffBaseSha: currentMrBaseSha,
      previousReviewSha,
      reviewDiffMode: "snapshot",
    });

    expect(prompt).toContain(`git --no-pager diff ${currentMrBaseSha} HEAD`);
    expect(prompt).not.toContain(`git --no-pager diff ${previousReviewSha} HEAD`);
  });

  it("advertises inspection tools by default", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
      embeddedDiff: "diff --git a/src/a.ts b/src/a.ts\n+const ok = true;",
      changedFiles: ["src/a.ts"],
    });

    expect(prompt).toContain("`grep` searches for directly relevant code");
    expect(prompt).toContain("`read` provides bounded surrounding context");
    expect(prompt).not.toContain("It is the only tool available");
  });

  it("withholds inspection tools on the single-turn fast path", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
      embeddedDiff: "diff --git a/src/a.ts b/src/a.ts\n+const ok = true;",
      changedFiles: ["src/a.ts"],
      singleTurn: true,
    });

    expect(prompt).toContain("It is the only tool available for this review");
    expect(prompt).toContain("call `submit_review` now, in this turn");
    expect(prompt).toContain("No file-inspection tools are available");
    expect(prompt).not.toContain("`grep` searches for directly relevant code");
    expect(prompt).not.toContain("`read` provides bounded surrounding context");
  });

  it("keeps the incremental rules consistent with the single-turn fast path", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
      previousReviewSha: "2".repeat(40),
      reviewDiffMode: "incremental",
      embeddedDiff: "diff --git a/src/a.ts b/src/a.ts\n+const ok = true;",
      changedFiles: ["src/a.ts"],
      singleTurn: true,
    });

    expect(prompt).not.toContain("verify the direct call sites or tests");
    expect(prompt).toContain("No file-inspection tools are available");
  });

  it("ignores the fast path when there is no embedded diff to reason from", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
      singleTurn: true,
    });

    expect(prompt).toContain("`grep` searches for directly relevant code");
    expect(prompt).not.toContain("It is the only tool available");
  });

  it("keeps the submission protocol in the effective system prompt", () => {
    const task = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
    });
    const systemPrompt = buildReviewSystemPrompt({
      reviewInstructions: loadDefaultReviewInstructions(),
    });

    expect(task).toContain("submit_review");
    expect(task).not.toContain("Call `submit_review` exactly once");
    expect(systemPrompt).toContain("Call `submit_review` exactly once");
    expect(systemPrompt).toContain("Do not print the final review as normal assistant text.");
  });

  it("keeps generic review lenses in the effective system prompt, not the dynamic task", () => {
    const task = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
    });
    const systemPrompt = buildReviewSystemPrompt({
      reviewInstructions: loadDefaultReviewInstructions(),
    });

    expect(task).not.toContain("Conditional Lenses");
    expect(task).not.toContain("For error handling, retries, fallbacks");
    expect(systemPrompt).toContain("## Conditional Lenses");
    expect(systemPrompt).toContain("For error handling, retries, fallbacks");
    expect(systemPrompt).toContain("For changed behavior, edge cases");
  });
});
