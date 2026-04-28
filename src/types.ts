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

export interface NoteEntry {
  body?: string;
  author?: {
    username?: string;
    name?: string;
  };
  created_at?: string;
  system?: boolean;
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
  durationSeconds: number;
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
  suggestion?: string;
}

export interface ReviewOutput {
  findings: ReviewFinding[];
  overall_correctness: ReviewCorrectness;
  overall_explanation: string;
}

export interface PostCommentResult {
  success: boolean;
  platform?: Platform;
  prNumber?: number;
  mrNumber?: number;
  error?: string;
}
