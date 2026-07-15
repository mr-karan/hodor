import type { ReviewOutput, ReviewPriority } from "./types.js";

export interface ExpectedFinding {
  path: string;
  keywords: string[];
  maximumPriority: ReviewPriority;
}

export interface EvaluationScore {
  expected: number;
  matched: number;
  missed: ExpectedFinding[];
  falsePositives: number;
  recall: number;
}

export function scoreReview(
  review: ReviewOutput,
  expectedFindings: ExpectedFinding[],
): EvaluationScore {
  const unmatchedActual = new Set(review.findings.map((_, index) => index));
  const missed: ExpectedFinding[] = [];
  let matched = 0;

  for (const expected of expectedFindings) {
    const actualIndex = review.findings.findIndex((finding, index) => {
      if (!unmatchedActual.has(index)) return false;
      const pathMatches = finding.code_location.absolute_file_path.endsWith(expected.path);
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

  return {
    expected: expectedFindings.length,
    matched,
    missed,
    falsePositives: unmatchedActual.size,
    recall: expectedFindings.length === 0 ? 1 : matched / expectedFindings.length,
  };
}
