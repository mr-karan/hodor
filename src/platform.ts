import type { ParsedPrUrl, Platform } from "./types.js";

export function detectPlatform(prUrl: string): Platform {
  const url = new URL(prUrl);
  const hostname = url.hostname;
  if (prUrl.includes("/-/merge_requests/") || hostname.includes("gitlab")) {
    return "gitlab";
  }
  if (
    prUrl.includes("/pulls/") ||
    hostname.includes("gitea") ||
    hostname.includes("forgejo") ||
    hostname.includes("codeberg")
  ) {
    return "gitea";
  }
  if (prUrl.includes("/pull/") || hostname.includes("github")) {
    return "github";
  }
  throw new Error(
    `Cannot detect platform for URL: ${prUrl}. Expected a GitHub (/pull/), GitLab (/-/merge_requests/), or Gitea/Forgejo (/pulls/) URL.`,
  );
}

export function parsePrUrl(prUrl: string): ParsedPrUrl {
  const url = new URL(prUrl);
  const pathParts = url.pathname.split("/").filter(Boolean);
  const host = url.host;

  if (pathParts.length >= 4 && pathParts[2] === "pull") {
    return {
      owner: pathParts[0],
      repo: pathParts[1],
      prNumber: parsePositiveNumber(pathParts[3], "PR", prUrl, "/pull/"),
      host,
    };
  }

  if (pathParts.length >= 4 && pathParts[2] === "pulls") {
    return {
      owner: pathParts[0],
      repo: pathParts[1],
      prNumber: parsePositiveNumber(pathParts[3], "PR", prUrl, "/pulls/"),
      host,
    };
  }

  const mrIndex = pathParts.indexOf("merge_requests");
  if (mrIndex >= 0) {
    if (mrIndex < 2 || mrIndex + 1 >= pathParts.length) {
      throw new Error(
        `Invalid GitLab MR URL format: ${prUrl}. Expected .../-/merge_requests/<number>`,
      );
    }
    if (pathParts[mrIndex - 1] !== "-") {
      throw new Error(
        `Invalid GitLab MR URL format: ${prUrl}. Missing '/-/' segment before merge_requests.`,
      );
    }

    const repo = pathParts[mrIndex - 2];
    const ownerParts = pathParts.slice(0, mrIndex - 2);
    const owner = ownerParts.length > 0 ? ownerParts.join("/") : pathParts[0];
    return {
      owner,
      repo,
      prNumber: parsePositiveNumber(
        pathParts[mrIndex + 1],
        "MR",
        prUrl,
        "/merge_requests/",
      ),
      host,
    };
  }

  throw new Error(
    `Invalid PR/MR URL format: ${prUrl}. Expected GitHub (/pull/), GitLab (/-/merge_requests/), or Gitea/Forgejo (/pulls/) URL.`,
  );
}

function parsePositiveNumber(
  raw: string,
  kind: "PR" | "MR",
  prUrl: string,
  segment: string,
): number {
  const number = Number(raw);
  if (!Number.isSafeInteger(number) || number <= 0) {
    throw new Error(
      `Invalid ${kind} number in URL: ${prUrl}. Expected a positive integer after ${segment}.`,
    );
  }
  return number;
}
