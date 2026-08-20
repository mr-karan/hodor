import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { exec, execJson } from "./utils/exec.js";
import { logger } from "./utils/logger.js";
import { fetchGitlabMrInfo } from "./gitlab.js";
import { fetchGiteaPrCheckoutInfo } from "./gitea.js";
import type { GiteaPrCheckoutInfo } from "./gitea.js";
import type { MrMetadata, Platform } from "./types.js";

export class WorkspaceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WorkspaceError";
  }
}

export interface WorkspaceResult {
  workspace: string;
  targetBranch: string;
  diffBaseSha: string | null;
  isTemporary: boolean;
}

// ---------------------------------------------------------------------------
// CI detection
// ---------------------------------------------------------------------------

interface CiWorkspace {
  path: string | null;
  targetBranch: string | null;
  diffBaseSha: string | null;
}

function envOrNull(name: string): string | null {
  const value = process.env[name]?.trim();
  return value ? value : null;
}

async function detectCiWorkspace(owner: string, repo: string): Promise<CiWorkspace> {
  const expected = `${owner}/${repo}`;

  // GitLab CI
  if (process.env.GITLAB_CI === "true") {
    const projectDir = envOrNull("CI_PROJECT_DIR");
    const projectPath = envOrNull("CI_PROJECT_PATH");
    const targetBranch = envOrNull("CI_MERGE_REQUEST_TARGET_BRANCH_NAME");
    const diffBaseSha = envOrNull("CI_MERGE_REQUEST_DIFF_BASE_SHA");

    if (projectDir && projectPath && (projectPath === expected || projectPath.endsWith(`/${expected}`))) {
      if (await isSameRepo(projectDir, owner, repo)) {
        logger.info(`Detected GitLab CI environment (target: ${targetBranch ?? "unknown"})`);
        return { path: projectDir, targetBranch, diffBaseSha };
      }
      logger.warn(
        `Detected GitLab CI for ${projectPath}, but ${projectDir} is not a git checkout of ${expected}; falling back to clone`,
      );
    }
  }

  // Gitea Actions / Forgejo Actions
  // Must check BEFORE GITHUB_ACTIONS since Gitea/Forgejo Actions sets GITHUB_ACTIONS=true for compat
  if (process.env.GITEA_ACTIONS === "true" || process.env.FORGEJO_ACTIONS === "true") {
    const workspaceDir = envOrNull("GITHUB_WORKSPACE");
    const repository = envOrNull("GITHUB_REPOSITORY");
    const baseRef = envOrNull("GITHUB_BASE_REF");

    if (workspaceDir && repository === expected) {
      const ciType = process.env.FORGEJO_ACTIONS ? "Forgejo" : "Gitea";
      if (await isSameRepo(workspaceDir, owner, repo)) {
        logger.info(`Detected ${ciType} Actions environment (base: ${baseRef ?? "unknown"})`);
        return { path: workspaceDir, targetBranch: baseRef, diffBaseSha: null };
      }
      logger.warn(
        `Detected ${ciType} Actions for ${repository}, but ${workspaceDir} is not a git checkout of ${expected}; falling back to clone`,
      );
    }
  }

  // GitHub Actions
  if (process.env.GITHUB_ACTIONS === "true") {
    const workspaceDir = envOrNull("GITHUB_WORKSPACE");
    const repository = envOrNull("GITHUB_REPOSITORY");
    const baseRef = envOrNull("GITHUB_BASE_REF");

    if (workspaceDir && repository === expected) {
      if (await isSameRepo(workspaceDir, owner, repo)) {
        logger.info(`Detected GitHub Actions environment (base: ${baseRef ?? "unknown"})`);
        return { path: workspaceDir, targetBranch: baseRef, diffBaseSha: null };
      }
      logger.warn(
        `Detected GitHub Actions for ${repository}, but ${workspaceDir} is not a git checkout of ${expected}; falling back to clone`,
      );
    }
  }

  return { path: null, targetBranch: null, diffBaseSha: null };
}

async function resolveGitlabDiffBaseSha(
  workspace: string,
  targetBranch: string | null,
  fallbackSha: string | null,
): Promise<string | null> {
  if (!targetBranch) return fallbackSha;

  try {
    await exec("git", ["fetch", "--no-tags", "origin", targetBranch], { cwd: workspace });
    const { stdout } = await exec("git", ["merge-base", "HEAD", "FETCH_HEAD"], { cwd: workspace });
    const mergeBase = stdout.trim();
    if (mergeBase) {
      logger.info(`Calculated current GitLab MR diff base: ${mergeBase.slice(0, 8)}`);
      return mergeBase;
    }
  } catch (err) {
    logger.warn(`Could not calculate current GitLab MR diff base: ${err}`);
  }

  try {
    const { stdout } = await exec("git", ["merge-base", "HEAD", `origin/${targetBranch}`], { cwd: workspace });
    const mergeBase = stdout.trim();
    if (mergeBase) {
      logger.info(`Calculated GitLab MR diff base from origin/${targetBranch}: ${mergeBase.slice(0, 8)}`);
      return mergeBase;
    }
  } catch {
    // Keep the CI-provided SHA as a last-resort compatibility fallback.
  }

  if (fallbackSha) logger.warn(`Falling back to CI_MERGE_REQUEST_DIFF_BASE_SHA: ${fallbackSha.slice(0, 8)}`);
  return fallbackSha;
}

// ---------------------------------------------------------------------------
// Repo identity check
// ---------------------------------------------------------------------------

function normalizeGitRemotePath(remoteUrl: string): string {
  const trimmed = remoteUrl.trim().replace(/\.git$/, "");

  try {
    const url = new URL(trimmed);
    return url.pathname.replace(/^\/+/, "").replace(/\.git$/, "");
  } catch {
    // Not a URL; try common SSH/scp-like forms below.
  }

  const scpLikeMatch = trimmed.match(/^[^@\s]+@[^:\s]+:(.+)$/);
  if (scpLikeMatch) {
    return scpLikeMatch[1].replace(/^\/+/, "").replace(/\.git$/, "");
  }

  return trimmed.replace(/^\/+/, "").replace(/\.git$/, "");
}

/**
 * Check if workspace is already cloned from the expected owner/repo.
 * Parses the remote URL to compare exact owner/repo, avoiding substring false positives.
 */
async function isSameRepo(
  workspace: string,
  owner: string,
  repo: string,
): Promise<boolean> {
  try {
    const { stdout } = await exec("git", ["remote", "get-url", "origin"], { cwd: workspace });
    const remoteUrl = stdout.trim();
    const expectedPath = `${owner}/${repo}`.replace(/\.git$/, "");
    const remotePath = normalizeGitRemotePath(remoteUrl);
    return remotePath === expectedPath || remotePath.toLowerCase() === expectedPath.toLowerCase();
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// GitHub helpers
// ---------------------------------------------------------------------------

async function getGithubBaseBranch(workspace: string, prNumber: string): Promise<string> {
  try {
    const prInfo = await execJson<Record<string, string>>(
      "gh",
      ["pr", "view", prNumber, "--json", "headRefName,baseRefName"],
      { cwd: workspace },
    );
    const baseBranch = prInfo.baseRefName ?? "main";
    logger.info(`Base branch: ${baseBranch}`);
    return baseBranch;
  } catch {
    logger.warn("Could not fetch PR metadata for base branch detection");
    return "main";
  }
}

async function fetchAndCheckoutGithubPr(
  workspace: string,
  prNumber: string,
): Promise<string> {
  logger.info(`Fetching and checking out PR #${prNumber} in existing workspace`);
  await exec("git", ["fetch", "origin"], { cwd: workspace });

  try {
    await exec("gh", ["pr", "checkout", prNumber], { cwd: workspace });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to checkout PR #${prNumber}: ${msg}`);
  }

  return getGithubBaseBranch(workspace, prNumber);
}

async function cloneAndCheckoutGithubPr(
  workspace: string,
  owner: string,
  repo: string,
  prNumber: string,
): Promise<string> {
  logger.info(`Setting up GitHub workspace for ${owner}/${repo}/pull/${prNumber}`);

  try {
    await exec("gh", ["version"]);
  } catch {
    throw new WorkspaceError("GitHub CLI (gh) is not available. Install it: https://cli.github.com");
  }

  logger.info(`Cloning repository ${owner}/${repo}...`);
  try {
    await exec("gh", ["repo", "clone", `${owner}/${repo}`, workspace]);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to clone repository ${owner}/${repo}: ${msg}`);
  }

  logger.info(`Checking out PR #${prNumber}...`);
  try {
    await exec("gh", ["pr", "checkout", prNumber], { cwd: workspace });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to checkout PR #${prNumber}: ${msg}`);
  }

  return getGithubBaseBranch(workspace, prNumber);
}

// ---------------------------------------------------------------------------
// GitLab helpers
// ---------------------------------------------------------------------------

async function getGitlabMrBranches(
  owner: string,
  repo: string,
  prNumber: string,
  host?: string,
): Promise<{ sourceBranch: string; targetBranch: string }> {
  const gitlabHost = host || process.env.GITLAB_HOST || "gitlab.com";
  let mrInfo: MrMetadata;
  try {
    mrInfo = await fetchGitlabMrInfo(owner, repo, Number(prNumber), gitlabHost);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to fetch MR info for !${prNumber}: ${msg}`);
  }

  const sourceBranch = mrInfo.source_branch;
  if (!sourceBranch) {
    throw new WorkspaceError(`Could not determine source branch for MR !${prNumber}`);
  }

  return { sourceBranch, targetBranch: mrInfo.target_branch ?? "main" };
}

async function checkoutGitlabBranch(workspace: string, sourceBranch: string): Promise<void> {
  try {
    await exec("git", ["checkout", "-b", sourceBranch, `origin/${sourceBranch}`], {
      cwd: workspace,
    });
  } catch {
    try {
      await exec("git", ["checkout", sourceBranch], { cwd: workspace });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new WorkspaceError(`Failed to checkout MR branch '${sourceBranch}': ${msg}`);
    }
  }
}

async function fetchAndCheckoutGitlabMr(
  workspace: string,
  owner: string,
  repo: string,
  prNumber: string,
  host?: string,
): Promise<string> {
  logger.info(`Fetching and checking out MR !${prNumber} in existing workspace`);
  await exec("git", ["fetch", "origin"], { cwd: workspace });

  const { sourceBranch, targetBranch } = await getGitlabMrBranches(owner, repo, prNumber, host);
  logger.info(`Source branch: ${sourceBranch}, Target branch: ${targetBranch}`);
  await checkoutGitlabBranch(workspace, sourceBranch);

  return targetBranch;
}

async function cloneAndCheckoutGitlabMr(
  workspace: string,
  owner: string,
  repo: string,
  prNumber: string,
  host?: string,
): Promise<string> {
  const gitlabHost = host || process.env.GITLAB_HOST || "gitlab.com";
  logger.info(`Setting up GitLab workspace for ${owner}/${repo}/merge_requests/${prNumber}`);

  try {
    await exec("glab", ["version"]);
  } catch {
    throw new WorkspaceError(
      "GitLab CLI (glab) is not available. Install it: https://gitlab.com/gitlab-org/cli",
    );
  }

  const cloneUrl = `https://${gitlabHost}/${owner}/${repo}.git`;
  logger.info(`Cloning from ${cloneUrl}...`);
  try {
    await exec("git", ["clone", cloneUrl, workspace]);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes("Permission denied") || msg.includes("publickey")) {
      throw new WorkspaceError(
        `Failed to clone ${owner}/${repo}: SSH authentication failed. ` +
        `Ensure your SSH key is available (ssh-add) or configure a GITLAB_TOKEN ` +
        `and use HTTPS: git config --global url."https://oauth2:$GITLAB_TOKEN@${gitlabHost}/".insteadOf "git@${gitlabHost}:"`,
      );
    }
    throw new WorkspaceError(`Failed to clone ${owner}/${repo}: ${msg}`);
  }

  const { sourceBranch, targetBranch } = await getGitlabMrBranches(owner, repo, prNumber, host);
  logger.info(`Source branch: ${sourceBranch}, Target branch: ${targetBranch}`);
  await checkoutGitlabBranch(workspace, sourceBranch);

  return targetBranch;
}

// ---------------------------------------------------------------------------
// Gitea helpers
// ---------------------------------------------------------------------------

function normalizeGiteaBaseUrl(host?: string): string {
  const giteaHost = host || process.env.GITEA_HOST || process.env.FORGEJO_HOST;
  if (!giteaHost) {
    throw new WorkspaceError("No Gitea/Forgejo host configured. Set GITEA_HOST or FORGEJO_HOST.");
  }
  const trimmed = giteaHost.trim().replace(/\/+$/, "");
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  return `https://${trimmed}`;
}

function giteaGitArgs(args: string[]): string[] {
  const token = process.env.GITEA_TOKEN || process.env.FORGEJO_TOKEN;
  return token ? ["-c", `http.extraHeader=Authorization: token ${token}`, ...args] : args;
}

async function getGiteaPrCheckoutInfo(
  owner: string,
  repo: string,
  prNumber: string,
  host?: string,
): Promise<GiteaPrCheckoutInfo> {
  try {
    return await fetchGiteaPrCheckoutInfo(owner, repo, Number(prNumber), host);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to fetch PR info for #${prNumber}: ${msg}`);
  }
}

async function checkoutGiteaBranch(
  workspace: string,
  prNumber: string,
  checkoutInfo: GiteaPrCheckoutInfo,
): Promise<void> {
  const { sourceBranch, sourceCloneUrl } = checkoutInfo;
  const localBranch = `hodor-pr-${prNumber}`;

  if (sourceCloneUrl) {
    try {
      await exec("git", giteaGitArgs(["fetch", sourceCloneUrl, sourceBranch]), { cwd: workspace });
      await exec("git", ["checkout", "-B", localBranch, "FETCH_HEAD"], { cwd: workspace });
      return;
    } catch (err) {
      logger.warn(`Failed to fetch Gitea PR branch from source repo, falling back to origin: ${err}`);
    }
  }

  try {
    await exec("git", ["checkout", "-B", localBranch, `origin/${sourceBranch}`], {
      cwd: workspace,
    });
  } catch {
    try {
      await exec("git", ["checkout", sourceBranch], { cwd: workspace });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new WorkspaceError(`Failed to checkout PR branch '${sourceBranch}': ${msg}`);
    }
  }
}

async function fetchAndCheckoutGiteaPr(
  workspace: string,
  owner: string,
  repo: string,
  prNumber: string,
  host?: string,
): Promise<string> {
  logger.info(`Fetching and checking out PR #${prNumber} in existing workspace`);
  await exec("git", giteaGitArgs(["fetch", "origin"]), { cwd: workspace });

  const checkoutInfo = await getGiteaPrCheckoutInfo(owner, repo, prNumber, host);
  logger.info(`Source branch: ${checkoutInfo.sourceBranch}, Target branch: ${checkoutInfo.targetBranch}`);
  await checkoutGiteaBranch(workspace, prNumber, checkoutInfo);

  return checkoutInfo.targetBranch;
}

async function cloneAndCheckoutGiteaPr(
  workspace: string,
  owner: string,
  repo: string,
  prNumber: string,
  host?: string,
): Promise<string> {
  logger.info(`Setting up Gitea workspace for ${owner}/${repo}/pulls/${prNumber}`);

  const cloneUrl = `${normalizeGiteaBaseUrl(host)}/${owner}/${repo}.git`;
  logger.info(`Cloning from ${cloneUrl}...`);

  try {
    await exec("git", giteaGitArgs(["clone", cloneUrl, workspace]));
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to clone ${owner}/${repo}: ${msg}`);
  }

  const checkoutInfo = await getGiteaPrCheckoutInfo(owner, repo, prNumber, host);
  logger.info(`Source branch: ${checkoutInfo.sourceBranch}, Target branch: ${checkoutInfo.targetBranch}`);
  await checkoutGiteaBranch(workspace, prNumber, checkoutInfo);

  return checkoutInfo.targetBranch;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function setupWorkspace(opts: {
  platform: Platform;
  owner: string;
  repo: string;
  prNumber: string;
  host?: string;
  workingDir?: string;
  reuse?: boolean;
}): Promise<WorkspaceResult> {
  const { platform, owner, repo, prNumber, host, workingDir, reuse = true } = opts;

  try {
    const ci = await detectCiWorkspace(owner, repo);
    let detectedTargetBranch = ci.targetBranch;
    let detectedDiffBaseSha = ci.diffBaseSha;

    let workspace: string;
    let isTemporary = false;

    if (ci.path) {
      workspace = ci.path;
      if (platform === "gitlab" && ci.targetBranch) {
        detectedDiffBaseSha = await resolveGitlabDiffBaseSha(
          workspace,
          ci.targetBranch,
          detectedDiffBaseSha,
        );
      }
      if (platform === "github" && !detectedTargetBranch) {
        detectedTargetBranch = await getGithubBaseBranch(workspace, prNumber);
      }
    } else if (!workingDir) {
      workspace = await mkdtemp(join(tmpdir(), "hodor-review-"));
      isTemporary = true;
      logger.info(`Created temporary workspace: ${workspace}`);
    } else {
      workspace = workingDir;
      const { mkdir } = await import("node:fs/promises");
      await mkdir(workspace, { recursive: true });

      if (reuse && (await isSameRepo(workspace, owner, repo))) {
        logger.info(`Reusing existing workspace: ${workspace}`);
        // Repo already cloned — just fetch and checkout the PR/MR branch
        if (platform === "github") {
          const tb = await fetchAndCheckoutGithubPr(workspace, prNumber);
          if (!detectedTargetBranch) detectedTargetBranch = tb;
        } else if (platform === "gitlab") {
          const tb = await fetchAndCheckoutGitlabMr(workspace, owner, repo, prNumber, host);
          if (!detectedTargetBranch) detectedTargetBranch = tb;
        } else if (platform === "gitea") {
          const tb = await fetchAndCheckoutGiteaPr(workspace, owner, repo, prNumber, host);
          if (!detectedTargetBranch) detectedTargetBranch = tb;
        }
        const finalTargetBranch = detectedTargetBranch ?? "main";
        logger.info(
          `Workspace ready at: ${workspace} (target: ${finalTargetBranch}, ` +
          `diff_base_sha: ${detectedDiffBaseSha?.slice(0, 8) ?? "N/A"})`,
        );
        return { workspace, targetBranch: finalTargetBranch, diffBaseSha: detectedDiffBaseSha, isTemporary: false };
      }
    }

    if (!ci.path) {
      if (platform === "github") {
        const tb = await cloneAndCheckoutGithubPr(workspace, owner, repo, prNumber);
        if (!detectedTargetBranch) detectedTargetBranch = tb;
      } else if (platform === "gitlab") {
        const tb = await cloneAndCheckoutGitlabMr(workspace, owner, repo, prNumber, host);
        if (!detectedTargetBranch) detectedTargetBranch = tb;
      } else if (platform === "gitea") {
        const tb = await cloneAndCheckoutGiteaPr(workspace, owner, repo, prNumber, host);
        if (!detectedTargetBranch) detectedTargetBranch = tb;
      } else {
        throw new WorkspaceError(`Unsupported platform: ${platform}`);
      }
    }

    const finalTargetBranch = detectedTargetBranch ?? "main";
    logger.info(
      `Workspace ready at: ${workspace} (target: ${finalTargetBranch}, ` +
      `diff_base_sha: ${detectedDiffBaseSha?.slice(0, 8) ?? "N/A"})`,
    );
    return { workspace, targetBranch: finalTargetBranch, diffBaseSha: detectedDiffBaseSha, isTemporary };
  } catch (err) {
    if (err instanceof WorkspaceError) throw err;
    const msg = err instanceof Error ? err.message : String(err);
    throw new WorkspaceError(`Failed to setup workspace: ${msg}`);
  }
}

export async function cleanupWorkspace(workspace: string): Promise<void> {
  try {
    await rm(workspace, { recursive: true, force: true });
    logger.info(`Cleaned up workspace: ${workspace}`);
  } catch (err) {
    logger.warn(`Failed to cleanup workspace ${workspace}: ${err}`);
  }
}
