import { appendFile, mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { reviewPr } from "../src/agent.js";
import {
  parseEvalCase,
  scoreReview,
  summarizeRuns,
  type EvalCase,
  type EvalRun,
} from "../src/evaluation.js";

const execFileAsync = promisify(execFile);
const args = process.argv.slice(2);

function flagValues(name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < args.length; index++) {
    if (args[index] !== name) continue;
    const value = args[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`${name} needs a value`);
    values.push(value);
  }
  return values;
}

function flagValue(name: string): string | undefined {
  return flagValues(name).at(-1);
}

function positiveInteger(name: string, fallback: number): number {
  const raw = flagValue(name);
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  return value;
}

const model = flagValue("--model") ?? process.env.HODOR_EVAL_MODEL ?? "anthropic/claude-opus-5-5";
const casesDir = resolve(flagValue("--cases") ?? join(process.cwd(), "evals"));
const instructionPaths = flagValues("--instructions").map((path) => resolve(path));
const focus = flagValue("--focus") ?? null;
const reasoningEffort = flagValue("--reasoning-effort");
const codemode = args.includes("--codemode");
const trials = positiveInteger("--trials", 1);
const concurrency = positiveInteger("--concurrency", 1);
const filter = flagValue("--filter");
const label = flagValue("--label") ?? "run";
const outPath = flagValue("--out");

const caseFiles = (await readdir(casesDir)).filter((file) => file.endsWith(".json")).sort();
const cases: EvalCase[] = [];
for (const caseFile of caseFiles) {
  const source = join(casesDir, caseFile);
  const evalCase = parseEvalCase(JSON.parse(await readFile(source, "utf-8")), source);
  if (evalCase.kind === "repo" && !isAbsolute(evalCase.repo)) evalCase.repo = resolve(casesDir, evalCase.repo);
  if (filter === undefined || new RegExp(filter).test(evalCase.id)) cases.push(evalCase);
}

if (args.includes("--list")) {
  for (const evalCase of cases) console.log(`${evalCase.id}: ${evalCase.description}`);
  process.exit(0);
}

const instructions = await Promise.all(instructionPaths.map((path) => readFile(path, "utf-8")));
const expectedByCase = new Map(cases.map((evalCase) => [evalCase.id, evalCase.expectedFindings.length]));
const jobs = Array.from({ length: trials }, (_, trial) => cases.map((evalCase) => ({ evalCase, trial: trial + 1 }))).flat();
const runs: EvalRun[] = [];

console.log(
  `${label}: ${cases.length} case(s) x ${trials} trial(s) = ${jobs.length} review(s) with ${model}` +
    `${instructionPaths.length > 0 ? `, instructions ${instructionPaths.join(", ")}` : ""}` +
    `${focus !== null ? `, focus "${focus}"` : ""}`,
);

async function prepareWorkspace(evalCase: EvalCase): Promise<{ workspace: string; diffAgainst: string }> {
  const workspace = await mkdtemp(join(tmpdir(), `hodor-eval-${evalCase.id}-`));
  if (evalCase.kind === "repo") {
    await execFileAsync("git", ["clone", "--quiet", "--no-checkout", evalCase.repo, workspace]);
    await execFileAsync("git", ["checkout", "--quiet", "--detach", evalCase.head], { cwd: workspace });
    return { workspace, diffAgainst: evalCase.base };
  }
  const git = (...gitArgs: string[]) => execFileAsync("git", gitArgs, { cwd: workspace });
  await git("init", "--quiet", "--initial-branch", "main");
  await git("config", "user.email", "hodor-eval@example.invalid");
  await git("config", "user.name", "Hodor Eval");
  await writeFiles(workspace, evalCase.baseFiles);
  await git("add", ".");
  await git("commit", "--quiet", "-m", "base");
  await writeFiles(workspace, evalCase.changedFiles);
  await git("add", ".");
  await git("commit", "--quiet", "-m", "change");
  return { workspace, diffAgainst: "HEAD~1" };
}

async function runJob({ evalCase, trial }: { evalCase: EvalCase; trial: number }): Promise<void> {
  const { workspace, diffAgainst } = await prepareWorkspace(evalCase);
  const started = Date.now();
  let run: EvalRun;
  let record: Record<string, unknown>;
  try {
    const result = await reviewPr({
      localMode: true,
      workspaceDir: workspace,
      diffAgainst,
      cleanup: false,
      model,
      instructions,
      focus,
      reasoningEffort,
      codemode,
    });
    const score = scoreReview(result.review, evalCase.expectedFindings);
    run = { caseId: evalCase.id, trial, score, error: null, cost: result.metrics.cost, durationSeconds: result.metrics.durationSeconds };
    record = {
      findings: result.review.findings.map((finding) => ({
        title: finding.title,
        body: finding.body,
        priority: finding.priority,
        path: finding.code_location.absolute_file_path.slice(workspace.length + 1),
        lines: finding.code_location.line_range,
      })),
      overallCorrectness: result.review.overall_correctness,
      metrics: result.metrics,
    };
    console.log(
      `${evalCase.id} #${trial}: ${score.matched}/${score.expected} expected, ${score.falsePositives} false positive(s), ` +
        `$${result.metrics.cost.toFixed(3)}, ${result.metrics.durationSeconds.toFixed(0)}s`,
    );
    for (const miss of score.missed) console.error(`  missed: ${miss.path} (${miss.keywords.join(", ")})`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    run = { caseId: evalCase.id, trial, score: null, error: message, cost: 0, durationSeconds: (Date.now() - started) / 1000 };
    record = {};
    console.error(`${evalCase.id} #${trial}: review failed: ${message}`);
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
  runs.push(run);
  if (outPath !== undefined) {
    await appendFile(outPath, `${JSON.stringify({ label, model, focus, instructions: instructionPaths, reasoningEffort: reasoningEffort ?? null, codemode, ...run, ...record })}\n`);
  }
}

const queue = [...jobs];
await Promise.all(
  Array.from({ length: Math.min(concurrency, queue.length) }, async () => {
    for (let job = queue.shift(); job !== undefined; job = queue.shift()) await runJob(job);
  }),
);

const summary = summarizeRuns(runs, expectedByCase);
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
console.log(
  `${label} aggregate: recall ${percent(summary.recall)} (per trial ${summary.recallByTrial.map(percent).join(", ")}), ` +
    `${summary.falsePositives} false positive(s) (per trial ${summary.falsePositivesByTrial.join(", ")}), ` +
    `${summary.reviews - summary.failedReviews}/${summary.reviews} reviews completed, ` +
    `$${summary.cost.toFixed(2)}, median ${summary.medianDurationSeconds.toFixed(0)}s`,
);
if (summary.failedReviews > 0 || summary.matched !== summary.expected || summary.falsePositives > 0) process.exitCode = 1;

async function writeFiles(root: string, files: Record<string, string>): Promise<void> {
  for (const [relativePath, contents] of Object.entries(files)) {
    const path = join(root, relativePath);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, contents, "utf-8");
  }
}
