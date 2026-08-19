/** Single-task public API for Probabilistic Pivot Tournament selection. */

import { VerifierClient, USAGE, diffUsage, type UsageSnapshot } from "./client.ts";
import {
  GROUND_TRUTH_NOTE,
  normalizeCriteria,
  type Criterion,
} from "./prompt.ts";
import { bradleyTerry, pivotRoundPairs, ringCycle, selectPivots } from "./ppt.ts";
import { directedReward, mergeCaches, type ScoreCache } from "./cache.ts";
import {
  contextResolver,
  mulberry32,
  scoreDirectedPairs,
  validateVerifyOptions,
} from "./run.ts";

export interface Candidate {
  name?: string;
  trace: string;
}

export interface SelectOptions {
  criteria?: Criterion[] | Record<string, string> | string[] | string;
  pivots?: number;
  nEvaluations?: number;
  seed?: number;
  maxWorkers?: number;
  cacheFile?: string;
  groundTruthNote?: string;
  onError?: "tie" | "raise";
  progress?: boolean;
  signal?: AbortSignal;
  client?: VerifierClient;
  taskName?: string;
}

export interface SelectResult {
  index: number;
  best: string;
  scores: number[];
  ranking: number[];
  nComparisons: number;
  criteria: string[];
  usage: UsageSnapshot;
}

export async function select(
  problem: string,
  candidates: Candidate[],
  opts: SelectOptions = {},
): Promise<SelectResult> {
  if (typeof problem !== "string" || !problem.trim()) {
    throw new Error("select: problem must be a non-empty string");
  }
  if (!Array.isArray(candidates) || candidates.length < 2) {
    throw new Error("select: at least two candidates are required");
  }
  for (const [index, candidate] of candidates.entries()) {
    if (!candidate || typeof candidate !== "object" || typeof candidate.trace !== "string") {
      throw new Error(`select: candidate ${index} trace must be a string`);
    }
    if (candidate.name !== undefined && typeof candidate.name !== "string") {
      throw new Error(`select: candidate ${index} name must be a string`);
    }
    if (!candidate.trace.trim()) throw new Error(`select: candidate ${index} has an empty trace`);
  }

  const criteria = normalizeCriteria(opts.criteria ?? "terminal_bench");
  const taskName = opts.taskName?.trim() || "task";
  const client = opts.client;
  if (!client) {
    throw new Error("select requires a verifier client for the OMP default model.");
  }
  const note = opts.groundTruthNote === undefined ? GROUND_TRUTH_NOTE : opts.groundTruthNote;
  const tasks = {
    [taskName]: candidates.map((candidate, index) => ({
      trialName: candidate.name?.trim() || `candidate_${index}`,
      reward: 0 as const,
      problem,
      trace: candidate.trace,
    })),
  };
  const { k, nReps, seed, maxWorkers } = validateVerifyOptions(tasks, criteria, {
    pivots: opts.pivots,
    nEvaluations: opts.nEvaluations,
    seed: opts.seed,
    maxWorkers: opts.maxWorkers,
  });
  const usageBefore = USAGE.snapshot();
  const rng = mulberry32(seed);
  const ring = ringCycle(candidates.length, rng);

  if (opts.progress !== false) console.log(`Phase A: ring pass (${ring.length} comparisons)`);
  let scores = await scoreDirectedPairs(
    client,
    tasks,
    { [taskName]: ring },
    criteria,
    note,
    nReps,
    maxWorkers,
    opts.cacheFile,
    {
      onError: opts.onError,
      progress: opts.progress,
      signal: opts.signal,
    },
  );

  const directed = (scoreCache: ScoreCache, a: number, b: number): [number, number] =>
    directedReward(
      scoreCache,
      taskName,
      a,
      b,
      criteria.map((criterion) => criterion.id),
      nReps,
      contextResolver(client, tasks[taskName], taskName, a, b, note, criteria),
    );

  const n = candidates.length;
  const w = new Array<number>(n).fill(0);
  const c = new Array<number>(n).fill(0);
  for (const [a, b] of ring) {
    const [ra, rb] = directed(scores, a, b);
    const preference = bradleyTerry(ra, rb);
    w[a] += preference;
    c[a] += 1;
    w[b] += 1 - preference;
    c[b] += 1;
  }
  const pivots = selectPivots(w, c, Math.min(k, n));
  const pivotPairs = pivotRoundPairs(n, pivots);
  if (opts.progress !== false) console.log(`Phase B: pivot rounds (${pivotPairs.length} comparisons)`);
  const phaseB = await scoreDirectedPairs(
    client,
    tasks,
    { [taskName]: pivotPairs },
    criteria,
    note,
    nReps,
    maxWorkers,
    opts.cacheFile,
    {
      onError: opts.onError,
      progress: opts.progress,
      signal: opts.signal,
      initialCache: scores,
    },
  );
  scores = mergeCaches(scores, phaseB);

  for (const [a, b] of pivotPairs) {
    const [ra, rb] = directed(scores, a, b);
    const preference = bradleyTerry(ra, rb);
    w[a] += preference;
    c[a] += 1;
    w[b] += 1 - preference;
    c[b] += 1;
  }

  const scoresPerCandidate = w.map((wins, index) => (c[index] ? wins / c[index] : 0));
  let best = 0;
  for (let index = 1; index < n; index++) {
    if (scoresPerCandidate[index] > scoresPerCandidate[best]) best = index;
  }
  const ranking = Array.from({ length: n }, (_, index) => index).sort(
    (a, b) => scoresPerCandidate[b] - scoresPerCandidate[a] || a - b,
  );
  return {
    index: best,
    best: candidates[best].name?.trim() || `candidate_${best}`,
    scores: scoresPerCandidate,
    ranking,
    nComparisons: ring.length + pivotPairs.length,
    criteria: criteria.map((criterion) => criterion.id),
    usage: diffUsage(USAGE.snapshot(), usageBefore),
  };
}
