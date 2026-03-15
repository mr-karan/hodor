import { isAbsolute } from "node:path";
import { Type } from "@sinclair/typebox";
import type { ReviewOutput, ReviewPriority } from "./types.js";

const REVIEW_PRIORITY_TAGS = new Map<string, ReviewPriority>([
  ["[P0]", 0],
  ["[P1]", 1],
  ["[P2]", 2],
  ["[P3]", 3],
]);

export const REVIEW_LOCATION_SCHEMA = Type.Object(
  {
    absolute_file_path: Type.String({ minLength: 1 }),
    line_range: Type.Object(
      {
        start: Type.Integer({ minimum: 1 }),
        end: Type.Integer({ minimum: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

export const REVIEW_FINDING_SCHEMA = Type.Object(
  {
    title: Type.String({ minLength: 1 }),
    body: Type.String({ minLength: 1 }),
    priority: Type.Integer({ minimum: 0, maximum: 3 }),
    code_location: REVIEW_LOCATION_SCHEMA,
    suggestion: Type.Optional(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export const SUBMIT_REVIEW_SCHEMA = Type.Object(
  {
    findings: Type.Array(REVIEW_FINDING_SCHEMA),
    overall_correctness: Type.Union([
      Type.Literal("patch is correct"),
      Type.Literal("patch is incorrect"),
    ]),
    overall_explanation: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export function validateReviewOutput(review: ReviewOutput): ReviewOutput {
  if (review.overall_explanation.trim().length === 0) {
    throw new Error("submit_review overall_explanation must be non-empty");
  }

  for (const [index, finding] of review.findings.entries()) {
    const label = `submit_review finding ${index + 1}`;
    if (finding.title.trim().length === 0) {
      throw new Error(`${label} title must be non-empty`);
    }
    if (finding.body.trim().length === 0) {
      throw new Error(`${label} body must be non-empty`);
    }

    const taggedPriority = getPriorityFromTitle(finding.title);
    if (taggedPriority == null) {
      throw new Error(`${label} title must start with [P0], [P1], [P2], or [P3]`);
    }
    if (finding.priority !== taggedPriority) {
      throw new Error(
        `${label} priority ${finding.priority} does not match title tag ${taggedPriority}`,
      );
    }

    const { absolute_file_path: filePath, line_range: lineRange } = finding.code_location;
    if (!isAbsolute(filePath)) {
      throw new Error(`${label} code_location.absolute_file_path must be absolute`);
    }
    if (lineRange.start > lineRange.end) {
      throw new Error(`${label} code_location line_range start must be <= end`);
    }
  }

  return review;
}

function getPriorityFromTitle(title: string): ReviewPriority | null {
  const match = title.match(/^\[(P[0-3])\]/);
  if (!match) return null;
  return REVIEW_PRIORITY_TAGS.get(`[${match[1]}]`) ?? null;
}
