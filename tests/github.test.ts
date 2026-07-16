import { describe, expect, it } from "vitest";
import { normalizeGithubMetadata } from "../src/github.js";

describe("normalizeGithubMetadata", () => {
  it("includes review bodies as notes so Hodor markers are discoverable", () => {
    const metadata = normalizeGithubMetadata({
      title: "Update handler",
      comments: [{
        body: "Human reviewer feedback with enough context",
        author: { login: "alice" },
        createdAt: "2026-07-15T00:00:00Z",
      }],
      reviews: [{
        body: `<!-- hodor:sha:${"1".repeat(40)} -->\n<!-- hodor-review -->\nNo issues found.`,
        author: { login: "hodor" },
        createdAt: "2026-07-16T00:00:00Z",
      }],
    });

    expect(metadata.Notes).toHaveLength(2);
    expect(metadata.Notes?.[0].author?.username).toBe("alice");
    expect(metadata.Notes?.[1].body).toContain("hodor:sha");
  });
});
