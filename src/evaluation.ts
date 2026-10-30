import type { ReviewOutput, ReviewPriority } from "./types.js";

export interface ExpectedFinding {
  path: string;
  keywords: string[];
  maximumPriority: ReviewPriority;
}

/** A location where an earlier review raised a finding that people rejected. */
export interface TrapFinding {
  path: string;
  line: number;
}

/** How far a finding may sit from a trap's line and still count as raising it again. */
export const TRAP_LINE_TOLERANCE = 3;

export interface EvaluationScore {
  expected: number;
  matched: number;
  missed: ExpectedFinding[];
  falsePositives: number;
  recall: number;
  traps: number;
  trapsHit: number;
}

/** True when a reviewed file is the case-relative path, compared by whole path segments. */
export function findingPathMatches(absolutePath: string, relativePath: string): boolean {
  return absolutePath === relativePath || absolutePath.endsWith(`/${relativePath}`);
}

export function scoreReview(
  review: ReviewOutput,
  expectedFindings: ExpectedFinding[],
  trapFindings: readonly TrapFinding[] = [],
): EvaluationScore {
  const unmatchedActual = new Set(review.findings.map((_, index) => index));
  const missed: ExpectedFinding[] = [];
  let matched = 0;

  for (const expected of expectedFindings) {
    const actualIndex = review.findings.findIndex((finding, index) => {
      if (!unmatchedActual.has(index)) return false;
      const pathMatches = findingPathMatches(finding.code_location.absolute_file_path, expected.path);
      const text = `${finding.title}\n${finding.body}`.toLowerCase();
      const keywordsMatch = expected.keywords.every((keyword) =>
        text.includes(keyword.toLowerCase()),
      );
      return pathMatches && keywordsMatch && finding.priority <= expected.maximumPriority;
    });

    if (actualIndex < 0) {
      missed.push(expected);
      continue;
    }
    unmatchedActual.delete(actualIndex);
    matched++;
  }

  const trapsHit = trapFindings.filter((trap) =>
    review.findings.some((finding) => {
      const { start, end } = finding.code_location.line_range;
      return (
        findingPathMatches(finding.code_location.absolute_file_path, trap.path) &&
        trap.line >= start - TRAP_LINE_TOLERANCE &&
        trap.line <= end + TRAP_LINE_TOLERANCE
      );
    }),
  ).length;

  return {
    expected: expectedFindings.length,
    matched,
    missed,
    falsePositives: unmatchedActual.size,
    recall: expectedFindings.length === 0 ? 1 : matched / expectedFindings.length,
    traps: trapFindings.length,
    trapsHit,
  };
}

interface EvalCaseBase {
  id: string;
  description: string;
  expectedFindings: ExpectedFinding[];
  trapFindings: TrapFinding[];
}

/** A change written out as file contents before and after. */
export interface InlineEvalCase extends EvalCaseBase {
  kind: "inline";
  baseFiles: Record<string, string>;
  changedFiles: Record<string, string>;
}

/** A change replayed from two commits of an existing git repository. */
export interface RepoEvalCase extends EvalCaseBase {
  kind: "repo";
  repo: string;
  base: string;
  head: string;
}

export type EvalCase = InlineEvalCase | RepoEvalCase;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringMap(value: unknown): value is Record<string, string> {
  return isRecord(value) && Object.values(value).every((entry) => typeof entry === "string");
}

function parseExpectedFinding(value: unknown, where: string): ExpectedFinding {
  if (!isRecord(value)) throw new Error(`${where}: expected finding must be an object`);
  const { path, keywords, maximumPriority } = value;
  if (typeof path !== "string" || path === "") throw new Error(`${where}: path must be a non-empty string`);
  if (!Array.isArray(keywords) || keywords.length === 0 || !keywords.every((k) => typeof k === "string" && k !== "")) {
    throw new Error(`${where}: keywords must be a non-empty list of strings`);
  }
  if (maximumPriority !== 0 && maximumPriority !== 1 && maximumPriority !== 2 && maximumPriority !== 3) {
    throw new Error(`${where}: maximumPriority must be 0, 1, 2, or 3`);
  }
  return { path, keywords, maximumPriority };
}

/** Parses one eval case file. Repo paths resolve against the case file's directory. */
export function parseEvalCase(value: unknown, source: string): EvalCase {
  if (!isRecord(value)) throw new Error(`${source}: eval case must be an object`);
  const { id, description, expectedFindings, trapFindings = [] } = value;
  if (typeof id !== "string" || id === "") throw new Error(`${source}: id must be a non-empty string`);
  if (typeof description !== "string") throw new Error(`${source}: description must be a string`);
  if (!Array.isArray(expectedFindings)) throw new Error(`${source}: expectedFindings must be a list`);
  if (!Array.isArray(trapFindings)) throw new Error(`${source}: trapFindings must be a list`);
  const base = {
    id,
    description,
    expectedFindings: expectedFindings.map((finding, index) =>
      parseExpectedFinding(finding, `${source}: expectedFindings[${index}]`),
    ),
    trapFindings: trapFindings.map((trap, index) => {
      if (!isRecord(trap) || typeof trap.path !== "string" || trap.path === "" || !Number.isInteger(trap.line)) {
        throw new Error(`${source}: trapFindings[${index}] needs a path and an integer line`);
      }
      return { path: trap.path, line: Number(trap.line) };
    }),
  };
  if ("repo" in value) {
    const { repo, base: baseRef, head } = value;
    if (typeof repo !== "string" || typeof baseRef !== "string" || typeof head !== "string") {
      throw new Error(`${source}: repo cases need string repo, base, and head`);
    }
    return { ...base, kind: "repo", repo, base: baseRef, head };
  }
  const { baseFiles, changedFiles } = value;
  if (!isStringMap(baseFiles) || !isStringMap(changedFiles)) {
    throw new Error(`${source}: inline cases need baseFiles and changedFiles maps of path to contents`);
  }
  return { ...base, kind: "inline", baseFiles, changedFiles };
}

/** The outcome of one review of one case. A failed review has no score. */
export interface EvalRun {
  caseId: string;
  trial: number;
  score: EvaluationScore | null;
  error: string | null;
  cost: number;
  durationSeconds: number;
}

export interface EvalSummary {
  reviews: number;
  failedReviews: number;
  trials: number;
  expected: number;
  matched: number;
  falsePositives: number;
  recall: number;
  traps: number;
  trapsHit: number;
  /** Recall of each trial, to show run-to-run spread. */
  recallByTrial: number[];
  falsePositivesByTrial: number[];
  trapsHitByTrial: number[];
  cost: number;
  medianDurationSeconds: number;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? (sorted[middle] ?? 0) : ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}

/**
 * Aggregates review runs. A failed review counts its expected findings as
 * missed, so errors lower recall instead of disappearing from the total.
 */
export function summarizeRuns(runs: readonly EvalRun[], expectedByCase: ReadonlyMap<string, number>): EvalSummary {
  const trials = [...new Set(runs.map((run) => run.trial))].sort((a, b) => a - b);
  const totals = (subset: readonly EvalRun[]) => {
    let expected = 0;
    let matched = 0;
    let falsePositives = 0;
    let traps = 0;
    let trapsHit = 0;
    for (const run of subset) {
      expected += run.score?.expected ?? expectedByCase.get(run.caseId) ?? 0;
      matched += run.score?.matched ?? 0;
      falsePositives += run.score?.falsePositives ?? 0;
      traps += run.score?.traps ?? 0;
      trapsHit += run.score?.trapsHit ?? 0;
    }
    return { expected, matched, falsePositives, traps, trapsHit, recall: expected === 0 ? 1 : matched / expected };
  };
  const all = totals(runs);
  const byTrial = trials.map((trial) => totals(runs.filter((run) => run.trial === trial)));
  return {
    reviews: runs.length,
    failedReviews: runs.filter((run) => run.error !== null).length,
    trials: trials.length,
    ...all,
    recallByTrial: byTrial.map((entry) => entry.recall),
    falsePositivesByTrial: byTrial.map((entry) => entry.falsePositives),
    trapsHitByTrial: byTrial.map((entry) => entry.trapsHit),
    cost: runs.reduce((total, run) => total + run.cost, 0),
    medianDurationSeconds: median(runs.map((run) => run.durationSeconds)),
  };
}
