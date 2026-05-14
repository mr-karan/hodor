import { describe, it, expect } from "vitest";
import {
  buildMrSections,
  buildPrReviewPrompt,
  normalizeLabelNames,
} from "../src/prompt.js";

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
  it("uses the tool submission contract by default", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
    });

    expect(prompt).toContain("submit_review");
    expect(prompt).toContain("Do not print the review as normal assistant text.");
    expect(prompt).not.toContain("Output ONLY the raw JSON object");
  });

  it("includes cross-layer contract tracing guidance", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
    });

    expect(prompt).toContain("Contract Trace Checklist");
    expect(prompt).toContain("public `user_id` string vs internal integer primary key");
  });

  it("includes conditional review lenses for focused specialist checks", () => {
    const prompt = buildPrReviewPrompt({
      prUrl: "https://github.com/acme/hodor/pull/42",
      platform: "github",
      targetBranch: "main",
    });

    expect(prompt).toContain("Conditional Review Lenses");
    expect(prompt).toContain("Silent failure / error handling lens");
    expect(prompt).toContain("Critical test gap lens");
    expect(prompt).toContain("Comment/documentation accuracy lens");
    expect(prompt).toContain("Type/API invariant lens");
    expect(prompt).toContain("Simplification lens");
  });
});
