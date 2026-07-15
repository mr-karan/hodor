import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { reviewPr } from "../src/agent.js";
import { scoreReview, type ExpectedFinding } from "../src/evaluation.js";

const execFileAsync = promisify(execFile);

interface EvalCase {
  id: string;
  description: string;
  baseFiles: Record<string, string>;
  changedFiles: Record<string, string>;
  expectedFindings: ExpectedFinding[];
}

const modelArg = process.argv.indexOf("--model");
const model =
  (modelArg >= 0 ? process.argv[modelArg + 1] : undefined) ??
  process.env.HODOR_EVAL_MODEL ??
  "anthropic/claude-sonnet-4-5-20250929";
const evalDir = join(process.cwd(), "evals");
const caseFiles = (await readdir(evalDir))
  .filter((file) => file.endsWith(".json"))
  .sort();

if (process.argv.includes("--list")) {
  for (const caseFile of caseFiles) {
    const evalCase = JSON.parse(
      await readFile(join(evalDir, caseFile), "utf-8"),
    ) as EvalCase;
    console.log(`${evalCase.id}: ${evalCase.description}`);
  }
  process.exit(0);
}

let expected = 0;
let matched = 0;
let falsePositives = 0;

for (const caseFile of caseFiles) {
  const evalCase = JSON.parse(
    await readFile(join(evalDir, caseFile), "utf-8"),
  ) as EvalCase;
  const workspace = await mkdtemp(join(tmpdir(), `hodor-eval-${evalCase.id}-`));
  try {
    await execFileAsync("git", ["init", "--quiet", "--initial-branch", "main"], {
      cwd: workspace,
    });
    await execFileAsync("git", ["config", "user.email", "hodor-eval@example.invalid"], {
      cwd: workspace,
    });
    await execFileAsync("git", ["config", "user.name", "Hodor Eval"], { cwd: workspace });
    await writeFiles(workspace, evalCase.baseFiles);
    await execFileAsync("git", ["add", "."], { cwd: workspace });
    await execFileAsync("git", ["commit", "--quiet", "-m", "base"], { cwd: workspace });
    await writeFiles(workspace, evalCase.changedFiles);
    await execFileAsync("git", ["add", "."], { cwd: workspace });
    await execFileAsync("git", ["commit", "--quiet", "-m", "change"], { cwd: workspace });

    const result = await reviewPr({
      localMode: true,
      workspaceDir: workspace,
      diffAgainst: "HEAD~1",
      cleanup: false,
      model,
    });
    const score = scoreReview(result.review, evalCase.expectedFindings);
    expected += score.expected;
    matched += score.matched;
    falsePositives += score.falsePositives;
    console.log(
      `${evalCase.id}: ${score.matched}/${score.expected} expected, ${score.falsePositives} false positive(s)`,
    );
    for (const miss of score.missed) {
      console.error(`  missed: ${miss.path} (${miss.keywords.join(", ")})`);
    }
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
}

const recall = expected === 0 ? 1 : matched / expected;
console.log(
  `Aggregate: recall ${(recall * 100).toFixed(1)}%, ${falsePositives} false positive(s)`,
);
if (matched !== expected || falsePositives > 0) process.exitCode = 1;

async function writeFiles(root: string, files: Record<string, string>): Promise<void> {
  for (const [relativePath, contents] of Object.entries(files)) {
    const path = join(root, relativePath);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, contents, "utf-8");
  }
}
