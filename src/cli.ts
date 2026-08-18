#!/usr/bin/env bun
/**
 * CLI for the omp LLM-as-a-Verifier plugin.
 *
 * Self-verification over a trajectory directory (Terminal-Bench layout:
 * <dir>/<task>/*_trajectory.json) or a flat directory of candidate JSON
 * files. Picks the best trial per task with the Probabilistic Pivot
 * Tournament and reports Pass@1 / verifier / oracle, mirroring
 * `scripts/run_bo5.py` from llm-as-a-verifier.
 *
 * Usage:
 *   bun run src/cli.ts <traj_dir> [--pivots N] [--k N] [--seed N]
 *                      [--workers N] [--effort xhigh] [--max-tokens N]
 *                      [--cache <path>] [--trials N] [--tasks a,b]
 */

import { statSync, readdirSync } from "node:fs";
import { VerifierClient, formatUsage, MissingAPIKeyError } from "./client.ts";
import { TERMINAL_BENCH_CRITERIA } from "./prompt.ts";
import { loadCandidateDir, loadTerminalDir } from "./loader.ts";
import { runBenchmark, renderReport } from "./run.ts";

function opt(args: Record<string, string>, name: string): number | undefined {
  const v = args[name];
  if (v === undefined) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : undefined;
}

function parseCli(argv: string[]): {
  positionals: string[];
  opts: Record<string, string>;
} {
  const positionals: string[] = [];
  const opts: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t.startsWith("--")) {
      const name = t.slice(2);
      const eq = name.indexOf("=");
      if (eq >= 0) opts[name.slice(0, eq)] = name.slice(eq + 1);
      else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
        opts[name] = argv[++i];
      } else opts[name] = "true";
    } else positionals.push(t);
  }
  return { positionals, opts };
}

async function main(): Promise<number> {
  const { positionals, opts } = parseCli(process.argv.slice(2));
  const dir = positionals[0];
  if (!dir) {
    console.error("Usage: bun run src/cli.ts <traj_dir> [options]");
    return 2;
  }

  const client = new VerifierClient({
    effort: opts["effort"],
    maxTokens: opt(opts, "max-tokens"),
  });
  if (!client.ready) {
    console.error(
      "Error: no verifier API key. Set OPENCODE_API_KEY or log in to " +
        "opencode-go in omp (`/login opencode-go`).",
    );
    return 1;
  }

  let isDir = false;
  try {
    isDir = statSync(dir).isDirectory();
  } catch {
    isDir = false;
  }
  if (!isDir) {
    console.error(`Error: not a directory: ${dir}`);
    return 2;
  }
  const hasSub = readdirSync(dir, { withFileTypes: true }).some((e) =>
    e.isDirectory(),
  );
  const { tasks } = hasSub
    ? loadTerminalDir(dir)
    : loadCandidateDir(dir, "task");
  if (Object.keys(tasks).length === 0) {
    console.error("Error: no trajectory JSON files found.");
    return 2;
  }

  const trialLimit = opt(opts, "trials");
  if (trialLimit) {
    for (const name of Object.keys(tasks)) {
      tasks[name] = tasks[name].slice(0, trialLimit);
    }
  }
  const taskFilter = opts["tasks"];
  if (taskFilter) {
    const keep = new Set(taskFilter.split(",").map((s) => s.trim()));
    for (const name of Object.keys(tasks)) {
      if (!keep.has(name)) delete tasks[name];
    }
  }

  const cacheFile = opts["cache"] || `${process.cwd()}/.verifier-cache.json`;
  console.log(`Loading ${Object.keys(tasks).length} task(s) from ${dir} …`);
  const stats = await runBenchmark(tasks, TERMINAL_BENCH_CRITERIA, {
    pivots: opt(opts, "pivots"),
    nEvaluations: opt(opts, "k") ?? opt(opts, "evaluations"),
    seed: opt(opts, "seed"),
    maxWorkers: opt(opts, "workers"),
    cacheFile,
    client,
    groundTruthNote: opts["note"] ?? "",
  });

  const lines = [
    renderReport(stats),
    formatUsage(stats.usage).join("\n"),
    "",
    "Winners per task (verifier choice, ground-truth reward):",
    ...Object.entries(stats.bestPerTask).map(
      ([task, b]) =>
        `  ${task}: trial #${b.index} (reward=${b.reward}, ` +
        `w=${b.w.toFixed(3)}, c=${b.c})`,
    ),
  ];
  console.log(lines.join("\n"));
  return 0;
}

try {
  process.exitCode = await main();
} catch (e) {
  if (e instanceof MissingAPIKeyError) {
    console.error(`Error: ${e.message}`);
  } else {
    console.error(e);
  }
  process.exitCode = 1;
}
