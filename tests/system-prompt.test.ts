import { describe, expect, it } from "vitest";
import { loadDefaultReviewInstructions } from "../src/review-instructions.js";
import {
  buildReviewSystemPrompt,
  HODOR_REVIEW_PROTOCOL,
} from "../src/system-prompt.js";

describe("review system prompt", () => {
  it("uses the bundled profile when composing the default effective system prompt", () => {
    const profile = loadDefaultReviewInstructions();
    const prompt = buildReviewSystemPrompt({ reviewInstructions: profile });

    expect(prompt).toContain("<REVIEW_INSTRUCTIONS>");
    expect(prompt).toContain(profile);
    expect(prompt).toContain("## Conditional Lenses");
    expect(prompt).toContain("<HODOR_REVIEW_PROTOCOL>");
  });

  it("keeps structured review output constraints outside replaceable profiles", () => {
    const prompt = buildReviewSystemPrompt({ reviewInstructions: "CUSTOM_PROFILE" });

    expect(prompt).toContain("imperative and at most 80 characters");
    expect(prompt).toContain("one concise natural-language paragraph");
    expect(prompt).toContain("exact contiguous current-source text");
    expect(prompt).toContain("same `line_range`");
    expect(prompt).toContain("one to three sentences");
  });

  it("uses a custom profile verbatim instead of adding the bundled profile", () => {
    const customProfile = "# Release profile\nReport only authorization regressions.";
    const prompt = buildReviewSystemPrompt({ reviewInstructions: customProfile });

    expect(prompt).toContain(`<REVIEW_INSTRUCTIONS>\n${customProfile}\n</REVIEW_INSTRUCTIONS>`);
    expect(prompt).not.toContain("Identify production bugs introduced by the proposed change.");
  });

  it("places profile, additional instructions, and Hodor protocol in that exact order", () => {
    const profile = "PROFILE_SENTINEL";
    const additional = "ADDITIONAL_SENTINEL";
    const prompt = buildReviewSystemPrompt({
      reviewInstructions: profile,
      additionalInstructions: additional,
    });

    expect(prompt).toBe(
      `<REVIEW_INSTRUCTIONS>\n${profile}\n</REVIEW_INSTRUCTIONS>\n\n` +
      `<ADDITIONAL_INSTRUCTIONS>\n${additional}\n</ADDITIONAL_INSTRUCTIONS>\n\n` +
      `<HODOR_REVIEW_PROTOCOL>\n${HODOR_REVIEW_PROTOCOL}\n</HODOR_REVIEW_PROTOCOL>`,
    );
  });

  it("does not emit an additional-instructions section when none is supplied", () => {
    const prompt = buildReviewSystemPrompt({ reviewInstructions: "PROFILE_SENTINEL" });

    expect(prompt).not.toContain("<ADDITIONAL_INSTRUCTIONS>");
  });
});
