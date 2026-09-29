import { fetchGiteaPublisherIdentity } from "./gitea.js";
import { fetchGithubUser } from "./github.js";
import { fetchGitlabPublisherIdentity, isHodorGeneratedNote } from "./gitlab.js";
import { logger } from "./utils/logger.js";
import type {
  GithubPublisherIdentity,
  NoteAuthor,
  NoteEntry,
  Platform,
  PublisherIdentity,
  TrustedHodorNote,
  UntrustedNote,
} from "./types.js";

const GITHUB_ACTIONS_BOT_LOGIN = "github-actions[bot]";

export interface PartitionedNotes {
  /** Hodor-marked notes written by the publishing identity. */
  hodor: TrustedHodorNote[];
  /** Everything else, including participant notes that copy a Hodor marker. */
  others: UntrustedNote[];
}

/**
 * Resolve the account Hodor posts as, or null when it cannot be determined.
 * A null identity means no historical note is trusted: the review runs fresh,
 * with no cache reuse and no incremental base.
 */
export async function resolvePublisherIdentity(
  platform: Platform,
  host: string,
): Promise<PublisherIdentity | null> {
  try {
    switch (platform) {
      case "gitlab":
        return await fetchGitlabPublisherIdentity(host);
      case "gitea":
        return await fetchGiteaPublisherIdentity(host);
      case "github":
        return await resolveGithubPublisherIdentity(host);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.warn(`Cannot resolve the ${platform} publishing identity; ignoring prior Hodor state: ${msg}`);
    return null;
  }
}

/**
 * GitHub identity rules, in order:
 * 1. HODOR_GITHUB_BOT_LOGIN, when set, names the account Hodor posts as.
 * 2. The account `gh api user` authenticates as.
 * 3. In GitHub Actions, where the installation token cannot call
 *    `gh api user`, `github-actions[bot]`.
 * Each is resolved to a numeric id through the API; the id differs on GHES.
 *
 * Caveat: github-actions[bot] is shared by every workflow in the repository.
 * Any workflow that can comment on the PR can write state that Hodor trusts.
 * Use a dedicated GitHub App or bot account to isolate Hodor state.
 */
async function resolveGithubPublisherIdentity(host: string): Promise<GithubPublisherIdentity> {
  const override = process.env.HODOR_GITHUB_BOT_LOGIN?.trim();
  if (override) return { platform: "github", ...(await fetchGithubUser(host, override)) };

  try {
    return { platform: "github", ...(await fetchGithubUser(host)) };
  } catch (err) {
    if (process.env.GITHUB_ACTIONS !== "true") throw err;
    logger.info(`gh api user failed in GitHub Actions; trusting ${GITHUB_ACTIONS_BOT_LOGIN}`);
    return { platform: "github", ...(await fetchGithubUser(host, GITHUB_ACTIONS_BOT_LOGIN)) };
  }
}

/** Numeric account ids only. Logins and display names are never compared. */
export function isAuthoredByPublisher(
  author: NoteAuthor | undefined,
  identity: PublisherIdentity,
): boolean {
  return author?.id != null && author.id === identity.userId;
}

/**
 * Split notes into authenticated Hodor state and untrusted context. This is
 * the only place that creates TrustedHodorNote values. Any previous
 * provenance on the input is ignored and recomputed.
 */
export function partitionNotesByProvenance(
  notes: readonly NoteEntry[] | undefined | null,
  identity: PublisherIdentity | null,
): PartitionedNotes {
  const result: PartitionedNotes = { hodor: [], others: [] };
  for (const note of notes ?? []) {
    const trusted =
      identity != null &&
      note.system !== true &&
      isHodorGeneratedNote(note.body) &&
      isAuthoredByPublisher(note.author, identity);
    if (trusted) {
      result.hodor.push({ ...note, provenance: "hodor" });
    } else {
      result.others.push({ ...note, provenance: "untrusted" });
    }
  }
  return result;
}
