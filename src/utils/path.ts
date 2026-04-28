/**
 * Strip workspace/CI prefixes from an absolute path so it becomes repo-relative.
 *
 * Used both for inline-comment paths posted to GitLab (must match the diff path)
 * and for the CodeClimate code quality artifact (GitLab compares against repo
 * paths, not the temp-dir absolute paths Hodor sees during a review).
 *
 * Honors `CI_PROJECT_DIR` first when set (GitLab CI sets it to the checkout root).
 * Otherwise, falls through generic patterns: GitLab `/builds/<group>/<project>/...`,
 * a `/workspace/...` segment, and Hodor's own `/tmp/hodor-review-<id>/...` temp dirs.
 */
export function relativizeWorkspacePath(absolutePath: string, workspacePrefix?: string): string {
  let filePath = absolutePath;

  const prefix = workspacePrefix ?? process.env.CI_PROJECT_DIR;
  if (prefix) {
    const trimmed = prefix.replace(/\/+$/, "");
    if (filePath.startsWith(`${trimmed}/`)) {
      return filePath.slice(trimmed.length + 1);
    }
  }

  const buildsMatch = filePath.match(/\/builds\/[^/]+\/[^/]+\/(.+)/);
  if (buildsMatch) return buildsMatch[1];

  if (filePath.includes("/workspace/")) {
    return filePath.slice(filePath.indexOf("/workspace/") + "/workspace/".length);
  }

  const stripped = filePath.replace(/^.*\/hodor-review-[^/]+\//, "");
  return stripped !== filePath ? stripped : filePath;
}
