import type { ReviewOutput, ReviewPriority } from "./types.js";

export type FailOnPriority = `P${ReviewPriority}`;

export function parseFailOnPriority(value: string): FailOnPriority {
  if (!/^P[0-3]$/.test(value)) {
    throw new Error("--fail-on-priority must be one of: P0, P1, P2, P3");
  }
  return value as FailOnPriority;
}

export function hasBlockingFinding(
  review: ReviewOutput,
  threshold: FailOnPriority,
): boolean {
  const maximumPriority = Number(threshold.slice(1));
  return review.findings.some((finding) => finding.priority <= maximumPriority);
}
