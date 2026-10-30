/**
 * A review key separates Hodor jobs that review the same MR or PR with
 * different instructions. A keyed job marks every note it writes and reads
 * back only notes with the same key. A job without a key reads back only
 * notes with no key, so its notes and outputs stay as they were.
 */

const REVIEW_KEY_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;
const MARKER_LINE_RE = /^\s*<!--.*-->\s*$/;
const KEY_MARKER_PREFIX_RE = /^\s*<!--\s*hodor:key:/i;

export type ReviewKey = string & { readonly __brand: "ReviewKey" };

function isReviewKey(value: string): value is ReviewKey {
  return REVIEW_KEY_RE.test(value);
}

export function parseReviewKey(value: string): ReviewKey {
  if (!isReviewKey(value)) {
    throw new Error(
      "--review-key must be 1 to 32 lowercase letters, digits, or hyphens, starting with a letter or digit",
    );
  }
  return value;
}

/**
 * Add the key marker after the leading marker lines. Markers that readers
 * expect first, such as `hodor:sha` or `hodor-review` then `hodor:superseded`,
 * keep their place.
 */
export function withReviewKey(body: string, key: ReviewKey | null): string {
  if (key === null) return body;
  const lines = body.split("\n");
  const end = lines.findIndex((line) => !MARKER_LINE_RE.test(line));
  lines.splice(end === -1 ? lines.length : end, 0, `<!-- hodor:key:${key} -->`);
  return lines.join("\n");
}

/**
 * True when a Hodor note carries this run's key, or no key marker when the
 * key is unset. Callers first establish that the publishing identity wrote
 * the note.
 */
export function belongsToReviewKey(body: string, key: ReviewKey | null): boolean {
  return readReviewKey(body) === key;
}

/** Read the key from the leading marker lines only, so review text cannot claim one. */
function readReviewKey(body: string): string | null {
  for (const line of body.split("\n")) {
    if (!MARKER_LINE_RE.test(line)) return null;
    const prefix = line.match(KEY_MARKER_PREFIX_RE);
    if (prefix) return line.slice(prefix[0].length, line.lastIndexOf("-->")).trim();
  }
  return null;
}

export function reviewCheckName(key: ReviewKey | null): string {
  return key === null ? "hodor" : `hodor/${key}`;
}
