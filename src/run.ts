/**
 * End-to-end verification pipeline (port of `scripts/run.py` +
 * `llm_verifier/fine_grained_reward.py#score_directed_pairs`).
 *
 * Pipeline per swing task (a task with both successes and failures):
 *   1. Sample one ring per swing task (deterministic given seed).
 *   2. Phase A: score all ring pairs (cached comparisons skipped).
 *   3. Pivots = ring-pass leaders; collect pivot-round pairs.
 *   4. Phase B: score all pivot-round pairs.
 *   5. Aggregate ring + pivot rounds into w_i / c_i; select the winner.
 *
 * Prefix-cache optimization: the backend's prefix cache is only populated
 * once a request returns, so one job per distinct prompt prefix runs first;
 * the rest then fan out against the warm cache.
 */

import { VerifierClient, USAGE, type UsageSnapshot } from "./client.ts";
import type { Criterion } from "./prompt.ts";
import { buildPrompt } from "./prompt.ts";
import { extractScore, type VerifierReply } from "./scale.ts";
import { bradleyTerry, ringCycle, selectPivots, pivotRoundPairs } from "./ppt.ts";
import {
  cacheKey,
  directedReward,
  loadCache,
  saveCache,
  type ScoreCache,
} from "./cache.ts";
import type { Tasks } from "./loader.ts";
import { classify } from "./loader.ts";

export interface VerifyOptions {
  pivots?: number;
  nEvaluations?: number; // K repeated verifications
  seed?: number;
  maxWorkers?: number;
  cacheFile?: string;
  criteria?: Criterion[];
  groundTruthNote?: string;
  onError?: "tie" | "raise";
  progress?: boolean;
  signal?: AbortSignal;
  client?: VerifierClient;
}

export interface RunStats {
  tasks: Tasks;
  criteriaIds: string[];
  nReps: number;
  k: number;
  seed: number;
  allPass: string[];
  swing: string[];
  nTasks: number;
  nRuns: number;
  pass1: number;
  verifier: number;
  oracle: number;
  avgComparisons: number;
  totalComparisons: number;
  usage: UsageSnapshot;
  bestPerTask: Record<string, { index: number; reward: 0 | 1; w: number; c: number }>;
}

interface Job {
  key: string;
  problem: string;
  traceA: string;
  traceB: string;
  crit: Criterion;
  swap: boolean;
  prefix: string;
}

const DEFAULT_MAX_WORKERS = 16;

export function defaultMaxWorkers(opts: VerifyOptions): number {
  return opts.maxWorkers ?? DEFAULT_MAX_WORKERS;
}

/**
 * Score every (criterion, rep) for the requested directed (task, a, b)
 * pairs and merge into the cache. Only comparisons missing from the cache
 * trigger API calls; odd reps swap the prompt slots (scores are recorded
 * back in candidate order), so with nReps >= 2 slot bias cancels within
 * each comparison.
 *
 * Returns the merged scores dict (error ties go into the returned dict but
 * are never persisted).
 */
export async function scoreDirectedPairs(
  client: VerifierClient,
  tasks: Tasks,
  neededPairs: Record<string, Array<[number, number]>>,
  criteria: Criterion[],
  groundTruthNote: string,
  nReps: number,
  maxWorkers: number,
  cacheFile: string | undefined,
  opts: { onError?: "tie" | "raise"; progress?: boolean; signal?: AbortSignal } = {},
): Promise<ScoreCache> {
  const cached = loadCache(cacheFile);
  const jobs: Job[] = [];
  for (const [taskName, pairs] of Object.entries(neededPairs)) {
    const trials = tasks[taskName];
    for (const [a, b] of pairs) {
      for (const crit of criteria) {
        for (let rep = 0; rep < nReps; rep++) {
          const key = cacheKey(crit.id, taskName, a, b, rep);
          if (key in cached) continue;
          const swap = rep % 2 === 1;
          const [sa, sb] = swap ? [b, a] : [a, b];
          jobs.push({
            key,
            problem: trials[a].problem,
            traceA: trials[swap ? b : a].trace,
            traceB: trials[swap ? a : b].trace,
            crit,
            swap,
            prefix: `${taskName}|${sa},${sb}`,
          });
        }
      }
    }
  }

  const log = opts.progress === false ? () => {} : console.log;
  if (jobs.length === 0) {
    log(`  All scores cached (${Object.keys(cached).length} entries)`);
    return cached;
  }

  // Warm-up wave: one job per distinct prompt prefix, then the rest.
  const seen = new Set<string>();
  const warm: Job[] = [];
  const rest: Job[] = [];
  for (const job of jobs) {
    if (seen.has(job.prefix)) rest.push(job);
    else {
      seen.add(job.prefix);
      warm.push(job);
    }
  }
  log(
    `  ${jobs.length} scoring jobs (${Object.keys(cached).length} cached); ` +
      `warming ${warm.length} prefixes`,
  );

  const results: ScoreCache = { ...cached };
  let errors = 0;

  async function scoreOne(job: Job): Promise<void> {
    const prompt = buildPrompt(job.problem, job.traceA, job.traceB, job.crit, groundTruthNote);
    let reply: VerifierReply;
    try {
      reply = await client.scoreReply(prompt, { signal: opts.signal });
    } catch (e) {
      if (opts.onError === "raise") throw e;
      errors += 1;
      results[job.key] = { score_A: 0.5, score_B: 0.5 };
      if (errors <= 3) log(`\n  Error: ${String(e)}`);
      return;
    }
    let ra = extractScore(reply, "<score_A>");
    let rb = extractScore(reply, "<score_B>");
    if (job.swap) [ra, rb] = [rb, ra]; // scores back in candidate order
    const entry = { score_A: ra, score_B: rb };
    cached[job.key] = entry;
    results[job.key] = entry;
  }

  async function runPhase(phaseJobs: Job[]): Promise<void> {
    if (phaseJobs.length === 0) return;
    let idx = 0;
    const workers = Math.min(maxWorkers, phaseJobs.length);
    async function worker(): Promise<void> {
      while (idx < phaseJobs.length) {
        const job = phaseJobs[idx++];
        if (opts.signal?.aborted) throw new DOMException("aborted", "AbortError");
        await scoreOne(job);
        if (cacheFile && idx % Math.max(1, Math.floor(phaseJobs.length / 20)) === 0) {
          saveCache(cacheFile, cached);
        }
      }
    }
    await Promise.all(Array.from({ length: workers }, () => worker()));
  }

  try {
    await runPhase(warm);
    await runPhase(rest);
  } finally {
    if (cacheFile) saveCache(cacheFile, cached);
  }
  log(`  Done (${errors} errors)`);
  return results;
}

/** Score one benchmark end to end and return metrics + per-task winners. */
export async function runBenchmark(
  tasks: Tasks,
  criteria: Criterion[],
  opts: VerifyOptions = {},
): Promise<RunStats> {
  const k = opts.pivots ?? 2;
  const nReps = opts.nEvaluations ?? 2;
  const seed = opts.seed ?? 0;
  const maxWorkers = defaultMaxWorkers(opts);
  const note = opts.groundTruthNote ?? "";
  const client = opts.client ?? new VerifierClient();
  const criteriaIds = criteria.map((c) => c.id);

  const { allPass, swing } = classify(tasks);
  const nTasks = Object.keys(tasks).length;
  const nRuns = Math.max(0, ...Object.values(tasks).map((t) => t.length));

  console.log(
    `  tasks=${nTasks}  all-pass=${allPass.length}  swing=${swing.length}  N(trials)=${nRuns}`,
  );
  console.log(
    `  criteria=${criteriaIds.join(",")}  K=${nReps}  pivots=${k}  seed=${seed}  max_workers=${maxWorkers}`,
  );

  // Deterministic RNG (mulberry32) so `seed` reproduces the same ring.
  const rng = mulberry32(seed);
  const rings: Record<string, Array<[number, number]>> = {};
  for (const task of swing) {
    rings[task] = ringCycle(tasks[task].length, rng);
  }

  console.log("Phase A: ring pass");
  let scores = await scoreDirectedPairs(
    client,
    tasks,
    rings,
    criteria,
    note,
    nReps,
    maxWorkers,
    opts.cacheFile,
    opts,
  );

  const directed = (task: string, a: number, b: number): [number, number] =>
    directedReward(scores, task, a, b, criteriaIds, nReps);

  const pivotsByTask: Record<string, number[]> = {};
  const prPairs: Record<string, Array<[number, number]>> = {};
  for (const task of swing) {
    const n = tasks[task].length;
    const w = new Array<number>(n).fill(0);
    const c = new Array<number>(n).fill(0);
    for (const [a, b] of rings[task]) {
      const [ra, rb] = directed(task, a, b);
      const p = bradleyTerry(ra, rb);
      w[a] += p; c[a] += 1; w[b] += 1 - p; c[b] += 1;
    }
    const pivots = selectPivots(w, c, k);
    pivotsByTask[task] = pivots;
    prPairs[task] = pivotRoundPairs(n, pivots);
  }

  console.log("Phase B: pivot rounds");
  scores = await scoreDirectedPairs(
    client,
    tasks,
    prPairs,
    criteria,
    note,
    nReps,
    maxWorkers,
    opts.cacheFile,
    opts,
  );

  let selected = 0;
  let totalComparisons = 0;
  const bestPerTask: RunStats["bestPerTask"] = {};
  for (const task of swing) {
    const n = tasks[task].length;
    const w = new Array<number>(n).fill(0);
    const c = new Array<number>(n).fill(0);
    const accumulate = (pairs: Array<[number, number]>) => {
      for (const [a, b] of pairs) {
        const [ra, rb] = directed(task, a, b);
        const p = bradleyTerry(ra, rb);
        w[a] += p; c[a] += 1; w[b] += 1 - p; c[b] += 1;
      }
    };
    accumulate(rings[task]);
    accumulate(prPairs[task]);
    let best = 0;
    for (let i = 1; i < n; i++) {
      const bi = c[i] ? w[i] / c[i] : 0;
      const bb = c[best] ? w[best] / c[best] : 0;
      if (bi > bb) best = i;
    }
    totalComparisons += rings[task].length + prPairs[task].length;
    if (tasks[task][best].reward === 1) selected += 1;
    bestPerTask[task] = {
      index: best,
      reward: tasks[task][best].reward,
      w: w[best],
      c: c[best],
    };
  }

  const pass1 = allPass.length + swing.reduce(
    (s, tn) => s + tasks[tn].reduce((x, t) => x + t.reward, 0) / tasks[tn].length,
    0,
  );
  const verifier = allPass.length + selected;
  const oracle = allPass.length + swing.length;

  return {
    tasks,
    criteriaIds,
    nReps,
    k,
    seed,
    allPass,
    swing,
    nTasks,
    nRuns,
    pass1,
    verifier,
    oracle,
    avgComparisons: totalComparisons / Math.max(1, swing.length),
    totalComparisons,
    usage: USAGE.snapshot(),
    bestPerTask,
  };
}

export function renderReport(stats: RunStats): string {
  const { nTasks, nRuns, swing, criteriaIds, nReps, k, seed } = stats;
  const lines = [
    "",
    "=".repeat(72),
    "SELF-VERIFICATION  (LLM-as-a-Verifier)",
    `  g20  criteria=${criteriaIds.join(",")}  K=${nReps}  pivots=${k}  seed=${seed}`,
    `  tasks=${nTasks}  swing=${swing.length}  N(trials)=${nRuns}  ` +
      `comparisons/task=${stats.avgComparisons.toFixed(1)}`,
    "=".repeat(72),
    `${"Method".padEnd(26)}  ${"Score".padStart(14)}  ${"Rate".padStart(7)}`,
    "-".repeat(72),
    `${"Pass@1".padEnd(26)}  ${stats.pass1.toFixed(2).padStart(14)}  ` +
      `${((100 * stats.pass1) / nTasks).toFixed(1).padStart(6)}%`,
    `${"LLM-as-a-Verifier".padEnd(26)}  ${String(stats.verifier).padStart(14)}  ` +
      `${((100 * stats.verifier) / nTasks).toFixed(1).padStart(6)}%`,
    `${`Oracle (Bo${nRuns})`.padEnd(26)}  ${String(stats.oracle).padStart(14)}  ` +
      `${((100 * stats.oracle) / nTasks).toFixed(1).padStart(6)}%`,
    "-".repeat(72),
  ];
  return lines.join("\n");
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
