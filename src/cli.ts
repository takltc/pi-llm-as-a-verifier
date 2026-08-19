#!/usr/bin/env bun
/** Headless CLI for the OMP LLM-as-a-Verifier extension. */

import { isAbsolute, resolve } from "node:path";
import {
  DEFAULT_MODEL_SELECTOR,
  MissingAPIKeyError,
  VerifierClient,
  formatUsage,
} from "./client.ts";
import { GROUND_TRUTH_NOTE, TERMINAL_BENCH_CRITERIA } from "./prompt.ts";
import { detectInputLayout, loadCandidateDir, loadTerminalDir, type Tasks } from "./loader.ts";
import { renderReport, runBenchmark } from "./run.ts";

interface CliArgs {
  positionals: string[];
  opts: Record<string, string>;
}

const USAGE = [
  "Usage: llm-verifier <traj_dir> [options]",
  "",
  "Options:",
  "  --pivots N       tournament pivots (Terminal-Bench 2.1 Bo5 default: 1)",
  "  --k N            repeated evaluations per criterion (default: 2)",
  "  --seed N         deterministic ring seed (default: 0)",
  "  --workers N      concurrent verifier requests (default: 16)",
  "  --trials N       keep the first N trials per task",
  "  --tasks a,b      select named Terminal-Bench tasks",
  "  --cache PATH     persistent score cache",
  `  --model MODEL    OMP selector (default: ${DEFAULT_MODEL_SELECTOR})`,
  "  --effort LEVEL   reasoning effort override",
  "  --max-tokens N   answer plus reasoning token budget",
].join("\n");

function parseCli(argv: string[]): CliArgs {
  const positionals: string[] = [];
  const opts: Record<string, string> = {};
  let optionsEnded = false;
  for (let index = 0; index < argv.length; index++) {
    const token = argv[index];
    if (!optionsEnded && token === "--") {
      optionsEnded = true;
      continue;
    }
    if (!optionsEnded && token.startsWith("--")) {
      const raw = token.slice(2);
      const equals = raw.indexOf("=");
      const name = equals >= 0 ? raw.slice(0, equals) : raw;
      if (!name) throw new Error(`Invalid option: ${token}`);
      if (equals >= 0) opts[name] = raw.slice(equals + 1);
      else if (index + 1 < argv.length && !argv[index + 1].startsWith("--")) {
        opts[name] = argv[++index];
      } else opts[name] = "true";
    } else positionals.push(token);
  }
  return { positionals, opts };
}

function integerOption(opts: Record<string, string>, name: string): number | undefined {
  const raw = opts[name];
  if (raw === undefined) return undefined;
  if (raw === "true") throw new Error(`--${name} requires an integer value`);
  if (!/^-?\d+$/.test(raw)) throw new Error(`--${name} must be an integer`);
  const value = Number(raw);
  if (!Number.isSafeInteger(value)) throw new Error(`--${name} is outside the safe integer range`);
  return value;
}

function assertKnownOptions(opts: Record<string, string>): void {
  const known = new Set([
    "help",
    "pivots",
    "k",
    "evaluations",
    "seed",
    "workers",
    "trials",
    "tasks",
    "cache",
    "model",
    "effort",
    "max-tokens",
    "note",
  ]);
  for (const name of Object.keys(opts)) {
    if (!known.has(name)) throw new Error(`Unknown option: --${name}`);
  }
}

function absolutePath(value: string): string {
  return isAbsolute(value) ? value : resolve(process.cwd(), value);
}

function loadTasks(dir: string): Tasks {
  return detectInputLayout(dir) === "terminal"
    ? loadTerminalDir(dir).tasks
    : loadCandidateDir(dir, "task").tasks;
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const { positionals, opts } = parseCli(argv);
  assertKnownOptions(opts);
  if (opts.help || positionals.length === 0) {
    console.log(USAGE);
    return opts.help ? 0 : 2;
  }
  if (positionals.length !== 1) throw new Error("Exactly one trajectory directory is required");

  const dir = absolutePath(positionals[0]);
  let tasks = loadTasks(dir);
  const trialLimit = integerOption(opts, "trials");
  if (trialLimit !== undefined) {
    if (trialLimit < 2) throw new Error("--trials must be at least 2");
    tasks = Object.fromEntries(
      Object.entries(tasks).map(([name, trials]) => [name, trials.slice(0, trialLimit)]),
    );
  }
  if (opts.tasks) {
    const keep = new Set(opts.tasks.split(",").map((name) => name.trim()).filter(Boolean));
    if (keep.size === 0) throw new Error("--tasks requires at least one task name");
    tasks = Object.fromEntries(
      Object.entries(tasks).filter(([name]) => keep.has(name)),
    );
    if (Object.keys(tasks).length === 0) throw new Error("--tasks did not match any loaded task");
  }

  const client = new VerifierClient({
    model: opts.model,
    effort: opts.effort,
    maxTokens: integerOption(opts, "max-tokens"),
  });
  if (!client.ready) {
    throw new MissingAPIKeyError(
      "No verifier API key. Set OPENCODE_API_KEY or log in to opencode-go in omp (`/login opencode-go`).",
    );
  }

  console.log(`Loading ${Object.keys(tasks).length} task(s) from ${dir} …`);
  const stats = await runBenchmark(tasks, TERMINAL_BENCH_CRITERIA, {
    pivots: integerOption(opts, "pivots"),
    nEvaluations: integerOption(opts, "k") ?? integerOption(opts, "evaluations"),
    seed: integerOption(opts, "seed"),
    maxWorkers: integerOption(opts, "workers"),
    cacheFile: absolutePath(opts.cache || ".verifier-cache.json"),
    client,
    groundTruthNote: opts.note === undefined ? GROUND_TRUTH_NOTE : opts.note,
  });
  const winners = Object.entries(stats.bestPerTask).map(
    ([task, result]) =>
      `  ${task}: trial #${result.index} (reward=${result.reward}, ` +
      `w=${result.w.toFixed(3)}, c=${result.c})`,
  );
  console.log(
    [
      renderReport(stats),
      formatUsage(stats.usage).join("\n"),
      winners.length ? "\nWinners per task (verifier choice, ground-truth reward):" : "",
      ...winners,
    ]
      .filter(Boolean)
      .join("\n"),
  );
  return 0;
}

if (import.meta.main) {
  try {
    process.exitCode = await main();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`Error: ${message}`);
    process.exitCode = error instanceof MissingAPIKeyError ? 1 : 2;
  }
}
