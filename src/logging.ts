export type ConsoleLogMode = "quiet" | "progress" | "verbose";

function truncateLine(line: string, maxLength: number): string {
  if (line.length <= maxLength) return line;
  return `${line.slice(0, maxLength - 1)}…`;
}

export function summarizeToolOutput(
  text: string,
  mode: Exclude<ConsoleLogMode, "quiet">,
): string[] {
  const normalized = text
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();
  if (!normalized) return [];

  const lines = normalized.split("\n");
  const headCount = mode === "verbose" ? 6 : 2;
  const tailCount = mode === "verbose" ? 4 : 2;
  const maxLineLength = mode === "verbose" ? 220 : 160;
  const truncatedLines = lines.map((line) => truncateLine(line, maxLineLength));

  if (lines.length <= headCount + tailCount) {
    return truncatedLines;
  }

  const head = truncatedLines.slice(0, headCount);
  const tail = truncatedLines.slice(-tailCount);
  const omittedCount = lines.length - head.length - tail.length;

  return [
    ...head,
    `... ${omittedCount} lines omitted ...`,
    ...tail,
  ];
}
