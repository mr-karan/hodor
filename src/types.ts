export type Platform = "github" | "gitlab" | "gitea";

export interface ParsedPrUrl {
  owner: string;
  repo: string;
  prNumber: number;
  host: string;
}

export interface MrMetadata {
  title?: string;
  description?: string;
  source_branch?: string;
  target_branch?: string;
  changes_count?: number;
  labels?: Array<string | { name?: string }>;
  label_details?: Array<string | { name?: string }>;
  author?: {
    username?: string;
    name?: string;
  };
  pipeline?: {
    status?: string;
    web_url?: string;
  };
  Notes?: Array<NoteEntry>;
  state?: string;
}

export interface NoteAuthor {
  /** Numeric account id. Provenance compares only this field. */
  id?: number;
  username?: string;
  name?: string;
}

interface NoteFields {
  /** Platform note id, when the platform reports one. */
  id?: number;
  body?: string;
  author?: NoteAuthor;
  created_at?: string;
  updated_at?: string;
  system?: boolean;
}

/** A note as returned by the platform. Nobody has checked who wrote it. */
export interface UntrustedNote extends NoteFields {
  provenance?: "untrusted";
}

/**
 * A Hodor-marked note written by the resolved publishing identity. Only
 * partitionNotesByProvenance creates these. Machine state (review SHA,
 * cached review, prior review context) is read only from trusted notes.
 */
export interface TrustedHodorNote extends NoteFields {
  provenance: "hodor";
}

export type NoteEntry = UntrustedNote | TrustedHodorNote;

/** The account Hodor posts as on the reviewed platform and host. */
export type PublisherIdentity = GitlabPublisherIdentity | GiteaPublisherIdentity | GithubPublisherIdentity;

export interface GitlabPublisherIdentity {
  platform: "gitlab";
  userId: number;
}

export interface GiteaPublisherIdentity {
  platform: "gitea";
  userId: number;
}

export interface GithubPublisherIdentity {
  platform: "github";
  userId: number;
  /** For logs only. Logins do not distinguish a bot from a same-named user. */
  login: string;
}

export interface ReviewMetrics {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  totalTokens: number;
  cost: number;
  turns: number;
  toolCalls: number;
  /** Tool calls made from inside codemode scripts (included in toolCalls). */
  nestedToolCalls?: number;
  /** Codemode script executions (included in toolCalls). */
  codemodeCalls?: number;
  durationSeconds: number;
  reviewMode?: "full" | "incremental" | "snapshot" | "local" | "reused";
  reasoningEffort?: string;
  diffFiles?: number;
  diffAdditions?: number;
  diffDeletions?: number;
  diffBytes?: number;
  /** True when the diff was embedded in the prompt; false when served through git_diff. */
  diffEmbedded?: boolean;
  reused?: boolean;
  /** Whether the single-turn tiny-diff fast path gated tools to submit_review. */
  fastPath?: boolean;
}

export type ReviewPriority = 0 | 1 | 2 | 3;
export type ReviewCorrectness = "patch is correct" | "patch is incorrect";

export interface ReviewFinding {
  title: string;
  body: string;
  priority: ReviewPriority;
  code_location: {
    absolute_file_path: string;
    line_range: { start: number; end: number };
  };
  /** Verbatim copy of the source lines the finding refers to, used to resolve
   * line_range against the on-disk file. Optional; falls back to line_range. */
  existing_code?: string;
  suggestion?: string;
}

export interface ReviewStateFinding {
  fingerprint: string;
  title: string;
  body: string;
  priority: ReviewPriority;
  filePath?: string;
  lineRange?: {
    start: number;
    end: number;
  };
}

export interface ReviewOutput {
  findings: ReviewFinding[];
  overall_correctness: ReviewCorrectness;
  overall_explanation: string;
  /**
   * Ids of earlier Hodor findings the review confirmed fixed. After reviewPr
   * returns, only full fingerprints that passed trusted validation remain.
   */
  resolved_findings?: string[];
}

/** The commits a review compared. */
export interface ReviewRange {
  /** Null in local mode, which reviews the working tree. */
  headSha: string | null;
  /** Target branch, or the --diff-against ref in local mode. */
  targetBranch: string;
  /** The commit the diff starts from, when known. */
  baseSha: string | null;
}

/** What the review prompt carried besides the diff. */
export interface ReviewContextManifest {
  /** Hodor finding threads shown, by status, and threads left out by the limit. */
  hodorThreads: { open: number; fixedWaiting: number; resolved: number; droppedByLimit: number };
  /** Top-level human notes shown, and qualifying notes left out by the budget. */
  humanComments: { included: number; droppedByBudget: number };
  /** Prior Hodor summaries shown for deduplication. */
  priorHodorReviews: number;
}

export interface PostCommentResult {
  success: boolean;
  platform?: Platform;
  prNumber?: number;
  mrNumber?: number;
  error?: string;
  errors?: string[];
  summaryPosted?: boolean;
  /** Web URL of the new GitLab summary note, when GitLab returned its id. */
  summaryUrl?: string;
  inlineCreated?: number;
  inlineFailed?: number;
  draftsPublished?: boolean;
  commitStatusPosted?: boolean;
  /** "Fixed in <sha>" replies posted on threads this review verified fixed. */
  fixedReplies?: number;
  /** Open threads confirmed fixed and waiting for a human to resolve them. */
  fixedAwaiting?: number;
  reviewFindings?: ReviewStateFinding[];
  reviewStateComplete?: boolean;
}
