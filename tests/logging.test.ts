import { describe, expect, it } from "vitest";
import { summarizeToolOutput } from "../src/logging.js";

describe("summarizeToolOutput", () => {
  it("returns all lines when the output is short", () => {
    expect(summarizeToolOutput("one\ntwo\nthree", "progress")).toEqual([
      "one",
      "two",
      "three",
    ]);
  });

  it("summarizes long progress output as head and tail", () => {
    const lines = Array.from({ length: 8 }, (_, index) => `line-${index + 1}`).join("\n");

    expect(summarizeToolOutput(lines, "progress")).toEqual([
      "line-1",
      "line-2",
      "... 4 lines omitted ...",
      "line-7",
      "line-8",
    ]);
  });

  it("shows a larger head and tail in verbose mode", () => {
    const lines = Array.from({ length: 12 }, (_, index) => `line-${index + 1}`).join("\n");

    expect(summarizeToolOutput(lines, "verbose")).toEqual([
      "line-1",
      "line-2",
      "line-3",
      "line-4",
      "line-5",
      "line-6",
      "... 2 lines omitted ...",
      "line-9",
      "line-10",
      "line-11",
      "line-12",
    ]);
  });
});
