import { logger } from "./utils/logger.js";
import { isRecord } from "./utils/json.js";
import type { GiteaPublisherIdentity, MrMetadata, UntrustedNote } from "./types.js";

export interface GiteaPrCheckoutInfo {
  sourceBranch: string;
  targetBranch: string;
  sourceCloneUrl?: string;
}

export class GiteaAPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GiteaAPIError";
  }
}

function normalizeBaseUrl(host?: string | null): string {
  const candidate =
    host ||
    process.env.GITEA_HOST ||
    process.env.FORGEJO_HOST;
  if (!candidate) {
    throw new GiteaAPIError(
      "No Gitea/Forgejo host configured. Set GITEA_HOST or FORGEJO_HOST, " +
      "or provide a full PR URL that includes the hostname.",
    );
  }
  const trimmed = candidate.trim();
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed.replace(/\/+$/, "");
  }
  return `https://${trimmed}`.replace(/\/+$/, "");
}

function giteaToken(): string | null {
  return process.env.GITEA_TOKEN ?? process.env.FORGEJO_TOKEN ?? null;
}

function requireGiteaToken(): string {
  const token = giteaToken();
  if (!token) {
    throw new GiteaAPIError(
      "No Gitea/Forgejo token found. Set GITEA_TOKEN or FORGEJO_TOKEN environment variable.",
    );
  }
  return token;
}

async function giteaFetch<T>(
  host: string | null | undefined,
  path: string,
  options?: { method?: string; body?: unknown },
): Promise<T> {
  const baseUrl = normalizeBaseUrl(host);
  const url = `${baseUrl}/api/v1/${path}`;
  const token = giteaToken();

  const headers: Record<string, string> = {
    Accept: "application/json",
    "Content-Type": "application/json",
  };
  if (token) {
    headers.Authorization = `token ${token}`;
  }

  const init: RequestInit = {
    method: options?.method ?? "GET",
    headers,
  };

  if (options?.body) {
    init.body = JSON.stringify(options.body);
  }

  const response = await fetch(url, init);

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    if (response.status === 401 || response.status === 403) {
      throw new GiteaAPIError(
        `Authentication failed (${response.status}): check GITEA_TOKEN/FORGEJO_TOKEN. ${text}`,
      );
    }
    if (response.status === 429) {
      throw new GiteaAPIError(`Rate limited by Gitea API (429). ${text}`);
    }
    throw new GiteaAPIError(
      `Gitea API error ${response.status} for ${options?.method ?? "GET"} ${path}: ${text}`,
    );
  }

  return (await response.json()) as T;
}

/**
 * Resolve the numeric id of the account the configured token belongs to.
 * Hodor trusts only comments written by this id.
 */
export async function fetchGiteaPublisherIdentity(
  host?: string | null,
): Promise<GiteaPublisherIdentity> {
  requireGiteaToken();
  const user = await giteaFetch<unknown>(host, "user");
  const userId = isRecord(user) ? user.id : undefined;
  if (typeof userId !== "number" || !Number.isSafeInteger(userId)) {
    throw new GiteaAPIError("Authenticated Gitea user has no numeric id");
  }
  return { platform: "gitea", userId };
}

/**
 * Fetch pull request metadata from Gitea/Forgejo.
 */
export async function fetchGiteaPrInfo(
  owner: string,
  repo: string,
  prNumber: number | string,
  host?: string | null,
  options?: { includeComments?: boolean },
): Promise<MrMetadata> {
  let prData: Record<string, unknown>;
  try {
    prData = await giteaFetch<Record<string, unknown>>(
      host,
      `repos/${owner}/${repo}/pulls/${prNumber}`,
    );
  } catch (err) {
    if (err instanceof GiteaAPIError) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    throw new GiteaAPIError(`Failed to fetch PR #${prNumber}: ${msg}`);
  }

  const user = (prData.user as Record<string, string>) ?? {};
  const head = (prData.head as Record<string, unknown>) ?? {};
  const base = (prData.base as Record<string, unknown>) ?? {};
  const labels = (prData.labels as Array<Record<string, string>>) ?? [];

  const metadata: MrMetadata = {
    title: prData.title as string | undefined,
    description: (prData.body as string) ?? "",
    source_branch: head.ref as string | undefined,
    target_branch: base.ref as string | undefined,
    changes_count: prData.changed_files as number | undefined,
    labels: labels.map((lbl) => ({ name: lbl.name })),
    author: {
      username: user.login,
      name: user.full_name || user.login,
    },
    state: prData.state as string | undefined,
  };

  if (options?.includeComments) {
    try {
      metadata.Notes = await fetchGiteaPrComments(owner, repo, prNumber, host);
    } catch (err) {
      logger.warn(`Failed to fetch PR comments: ${err instanceof Error ? err.message : err}`);
    }
  }

  return metadata;
}

/**
 * Fetch the PR branch metadata needed to checkout same-repo and fork PRs.
 */
export async function fetchGiteaPrCheckoutInfo(
  owner: string,
  repo: string,
  prNumber: number | string,
  host?: string | null,
): Promise<GiteaPrCheckoutInfo> {
  const prData = await giteaFetch<Record<string, unknown>>(
    host,
    `repos/${owner}/${repo}/pulls/${prNumber}`,
  );
  const head = (prData.head as Record<string, unknown>) ?? {};
  const base = (prData.base as Record<string, unknown>) ?? {};
  const headRepo = (head.repo as Record<string, unknown> | null) ?? null;
  const sourceBranch = head.ref as string | undefined;

  if (!sourceBranch) {
    throw new GiteaAPIError(`Could not determine source branch for PR #${prNumber}`);
  }

  return {
    sourceBranch,
    targetBranch: (base.ref as string | undefined) ?? "main",
    sourceCloneUrl: headRepo?.clone_url as string | undefined,
  };
}

/**
 * Fetch comments on a Gitea/Forgejo pull request.
 * PRs are a superset of issues in Gitea, so comments use the issues endpoint.
 */
export async function fetchGiteaPrComments(
  owner: string,
  repo: string,
  prNumber: number | string,
  host?: string | null,
): Promise<UntrustedNote[]> {
  const comments: Array<Record<string, unknown>> = [];
  const pageSize = 100;

  for (let page = 1; ; page++) {
    const batch = await giteaFetch<Array<Record<string, unknown>>>(
      host,
      `repos/${owner}/${repo}/issues/${prNumber}/comments?page=${page}&limit=${pageSize}`,
    );
    comments.push(...batch);
    if (batch.length < pageSize) break;
  }

  return comments.map((c) => {
    const user = isRecord(c.user) ? c.user : {};
    const login = typeof user.login === "string" ? user.login : undefined;
    const fullName = typeof user.full_name === "string" ? user.full_name : undefined;
    return {
      body: typeof c.body === "string" ? c.body : "",
      author: {
        id: typeof user.id === "number" && Number.isSafeInteger(user.id) ? user.id : undefined,
        username: login,
        name: fullName || login,
      },
      created_at: typeof c.created_at === "string" ? c.created_at : undefined,
      updated_at: typeof c.updated_at === "string" ? c.updated_at : undefined,
      system: false,
    };
  });
}

/**
 * Post a comment on a Gitea/Forgejo pull request.
 */
export async function postGiteaPrComment(
  owner: string,
  repo: string,
  prNumber: number | string,
  body: string,
  host?: string | null,
): Promise<void> {
  // Posting requires authentication
  requireGiteaToken();

  try {
    await giteaFetch<unknown>(
      host,
      `repos/${owner}/${repo}/issues/${prNumber}/comments`,
      { method: "POST", body: { body } },
    );
  } catch (err) {
    if (err instanceof GiteaAPIError) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    throw new GiteaAPIError(`Failed to post comment to PR #${prNumber}: ${msg}`);
  }
}
