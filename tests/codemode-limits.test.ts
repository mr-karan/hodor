import { describe, expect, it } from "vitest";
import { capCodemodeOptions } from "../src/codemode-limits.js";

const LIMITS = { timeoutMs: 120_000, maxOutputTokens: 10_000 };

function options(code: string): unknown {
  const line = code.slice(0, code.indexOf("\n"));
  return JSON.parse(line.slice("// @options:".length));
}

describe("capCodemodeOptions", () => {
  it("adds the limits when the script has no options line", () => {
    const out = capCodemodeOptions("text(1);", LIMITS);
    expect(options(out)).toEqual({ timeout_ms: 120_000, max_output_tokens: 10_000 });
    expect(out.endsWith("\ntext(1);")).toBe(true);
  });

  it("keeps lower values the model chose", () => {
    const out = capCodemodeOptions('// @options: {"timeout_ms": 5000, "max_output_tokens": 2000}\ntext(1);', LIMITS);
    expect(options(out)).toEqual({ timeout_ms: 5000, max_output_tokens: 2000 });
  });

  it("caps values above the limits", () => {
    const out = capCodemodeOptions('// @options: {"timeout_ms": 999999999, "max_output_tokens": 500000}\ntext(1);', LIMITS);
    expect(options(out)).toEqual({ timeout_ms: 120_000, max_output_tokens: 10_000 });
  });

  it("fills in a missing field and caps non-positive or non-integer values", () => {
    const out = capCodemodeOptions('// @options: {"timeout_ms": 0}\ntext(1);', LIMITS);
    expect(options(out)).toEqual({ timeout_ms: 120_000, max_output_tokens: 10_000 });
  });

  it("leaves a malformed options line for codemode to reject", () => {
    const code = "// @options: {not json\ntext(1);";
    expect(capCodemodeOptions(code, LIMITS)).toBe(code);
  });

  it("handles an options line indented or with CRLF", () => {
    const out = capCodemodeOptions('  // @options: {"timeout_ms": 1000}\r\ntext(1);', LIMITS);
    expect(options(out)).toEqual({ timeout_ms: 1000, max_output_tokens: 10_000 });
    expect(out).toContain("\ntext(1);");
  });
});
