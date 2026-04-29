import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { relativizeWorkspacePath } from "../src/utils/path.js";

describe("relativizeWorkspacePath", () => {
  const originalEnv = process.env.CI_PROJECT_DIR;

  beforeEach(() => {
    delete process.env.CI_PROJECT_DIR;
  });

  afterEach(() => {
    if (originalEnv === undefined) delete process.env.CI_PROJECT_DIR;
    else process.env.CI_PROJECT_DIR = originalEnv;
  });

  it("strips an explicit workspace prefix (with or without trailing slash)", () => {
    expect(relativizeWorkspacePath("/home/me/repo/src/app.ts", "/home/me/repo")).toBe("src/app.ts");
    expect(relativizeWorkspacePath("/home/me/repo/src/app.ts", "/home/me/repo/")).toBe("src/app.ts");
  });

  it("falls back to CI_PROJECT_DIR when no explicit prefix is passed", () => {
    process.env.CI_PROJECT_DIR = "/builds/acme/app";
    expect(relativizeWorkspacePath("/builds/acme/app/src/auth.ts")).toBe("src/auth.ts");
  });

  it("explicit prefix wins over CI_PROJECT_DIR", () => {
    process.env.CI_PROJECT_DIR = "/builds/acme/app";
    expect(
      relativizeWorkspacePath("/home/me/clone/src/auth.ts", "/home/me/clone"),
    ).toBe("src/auth.ts");
  });

  it("uses /builds/<group>/<project>/ pattern when CI_PROJECT_DIR is unset", () => {
    expect(relativizeWorkspacePath("/builds/acme/app/src/api.ts")).toBe("src/api.ts");
  });

  it("strips /workspace/ segment", () => {
    expect(relativizeWorkspacePath("/home/runner/workspace/src/db.ts")).toBe("src/db.ts");
  });

  it("strips hodor temp dirs", () => {
    expect(relativizeWorkspacePath("/tmp/hodor-review-abc123/src/foo.ts")).toBe("src/foo.ts");
  });

  it("returns the original path when no pattern matches", () => {
    // No prefix, no env, no recognizable pattern — input is returned untouched.
    // This is the case the GitLab inline-post regression guarded against:
    // when called WITHOUT a workspacePath, we used to silently emit absolute paths.
    expect(relativizeWorkspacePath("/opt/work/my-clone/src/x.ts")).toBe(
      "/opt/work/my-clone/src/x.ts",
    );
  });

  it("returns relative path when a custom workspace IS passed (the regression fix)", () => {
    expect(
      relativizeWorkspacePath("/opt/work/my-clone/src/x.ts", "/opt/work/my-clone"),
    ).toBe("src/x.ts");
  });
});
