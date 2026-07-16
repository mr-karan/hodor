import { execJson } from "./utils/exec.js";
import type { MrMetadata, NoteEntry } from "./types.js";

export class GitHubAPIError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GitHubAPIError";
  }
}

export async function fetchGithubPrInfo(
  owner: string,
  repo: string,
  prNumber: number | string,
): Promise<Record<string, unknown>> {
  const fields = [
    "number",
    "title",
    "body",
    "author",
    "baseRefName",
    "headRefName",
    "baseRefOid",
    "headRefOid",
    "changedFiles",
    "labels",
    "comments",
    "reviews",
    "state",
    "isDraft",
    "createdAt",
    "updatedAt",
    "mergeable",
    "url",
  ];

  const repoFullPath = `${owner}/${repo}`;
  try {
    return await execJson<Record<string, unknown>>("gh", [
      "pr",
      "view",
      String(prNumber),
      "-R",
      repoFullPath,
      "--json",
      fields.join(","),
    ]);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new GitHubAPIError(msg);
  }
}

export function normalizeGithubMetadata(
  raw: Record<string, unknown>,
): MrMetadata {
  const author = (raw.author as Record<string, string>) ?? {};
  const labels = (raw.labels as Array<Record<string, string>>) ?? [];
  const comments = raw.comments as
    | Record<string, unknown>
    | Array<Record<string, unknown>>
    | undefined;
  const reviews = raw.reviews as
    | Record<string, unknown>
    | Array<Record<string, unknown>>
    | undefined;

  return {
    title: raw.title as string | undefined,
    description: (raw.body as string) ?? "",
    source_branch: raw.headRefName as string | undefined,
    target_branch: raw.baseRefName as string | undefined,
    changes_count: raw.changedFiles as number | undefined,
    labels: labels.map((lbl) => ({ name: lbl.name ?? lbl.id })),
    author: {
      username: author.login ?? author.name,
      name: author.name,
    },
    Notes: [
      ...githubCommentsToNotes(comments),
      ...githubCommentsToNotes(reviews),
    ],
  };
}

function githubCommentsToNotes(
  comments:
    | Record<string, unknown>
    | Array<Record<string, unknown>>
    | undefined
    | null,
): NoteEntry[] {
  if (!comments) return [];

  let nodes: Array<Record<string, unknown>>;
  if (Array.isArray(comments)) {
    nodes = comments;
  } else if (typeof comments === "object") {
    nodes =
      (comments.nodes as Array<Record<string, unknown>>) ??
      (comments.edges as Array<Record<string, unknown>>) ??
      [];
    // Handle GraphQL edge format
    if (
      nodes.length > 0 &&
      typeof nodes[0] === "object" &&
      "node" in nodes[0]
    ) {
      nodes = nodes.map(
        (edge) =>
          (edge.node as Record<string, unknown>) ?? {},
      );
    }
  } else {
    nodes = [];
  }

  return nodes.map((node) => {
    const author = (node.author as Record<string, string>) ?? {};
    return {
      body: (node.body as string) ?? "",
      author: {
        username: author.login ?? author.name,
        name: author.name,
      },
      created_at: node.createdAt as string | undefined,
    };
  });
}
