/** Single-task public API for Probabilistic Pivot Tournament selection. */

import type { ImageContent } from "@oh-my-pi/pi-ai";
import { VerifierClient, USAGE, diffUsage, type UsageSnapshot } from "./client.ts";
import {
  GROUND_TRUTH_NOTE,
  normalizeCriteria,
  type Criterion,
} from "./prompt.ts";
import { accumulate, pivotRoundPairs, ringCycle, selectPivots } from "./ppt.ts";
import {
  directedReward,
  type ScoreCache,
  type ScoreDistributionQuality,
  type ScoreSourceCounts,
} from "./cache.ts";
import {
  contextResolver,
  createUnsupportedBreaker,
  mulberry32,
  scoreDirectedPairs,
  summarizeScoredPairs,
  validateVerifyOptions,
} from "./run.ts";

export interface Candidate {
  name?: string;
  trace: string;
  /** Images produced inside this candidate trajectory. */
  images?: readonly ImageContent[];
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
  images?: readonly ImageContent[];
}

export interface SelectResult {
  index: number;
  best: string;
  scores: number[];
  ranking: number[];
  nComparisons: number;
  criteria: string[];
  scoreSources: ScoreSourceCounts;
  scoreDistribution: ScoreDistributionQuality;
  paperEquivalent: boolean;
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
    if (candidate.images !== undefined && !Array.isArray(candidate.images)) {
      throw new Error(`select: candidate ${index} images must be an array`);
    }
  }

  const criteria = normalizeCriteria(opts.criteria ?? "terminal_bench");
  const taskName = opts.taskName?.trim() || "task";
  const client = opts.client;
  if (!client) {
    throw new Error("select requires a verifier client for the OMP default model.");
  }
  const images = opts.images ?? [];
  const nImages = images.length + candidates.reduce(
    (total, candidate) => total + (candidate.images?.length ?? 0),
    0,
  );
  if (nImages > 0 && !client.supportsImages) {
    throw new Error(
      `select: verifier model ${client.provider}/${client.model} cannot inspect ${nImages} context image(s)`,
    );
  }
  const note = opts.groundTruthNote === undefined ? GROUND_TRUTH_NOTE : opts.groundTruthNote;
  const tasks = {
    [taskName]: candidates.map((candidate, index) => ({
      trialName: candidate.name?.trim() || `candidate_${index}`,
      reward: 0 as const,
      problem,
      trace: candidate.trace,
      images,
      trajectoryImages: candidate.images,
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
  // A no-logprobs rejection is a property of the shared request shape, so the
  // two phases share one breaker: a ring-pass confirmation skips the pivot
  // round instead of spending fresh provider calls on the same capability.
  const unsupportedBreaker = createUnsupportedBreaker();
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
      unsupportedBreaker,
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
  accumulate(ring, (a, b) => directed(scores, a, b), w, c);
  const pivots = selectPivots(w, c, Math.min(k, n));
  const pivotPairs = pivotRoundPairs(n, pivots, ring);
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
      unsupportedBreaker,
    },
  );
  // Phase B was seeded with the phase-A cache, so its result already carries
  // every ring score.
  scores = phaseB;

  accumulate(pivotPairs, (a, b) => directed(scores, a, b), w, c);

  const scoresPerCandidate = w.map((wins, index) => (c[index] ? wins / c[index] : 0));
  let best = 0;
  for (let index = 1; index < n; index++) {
    if (scoresPerCandidate[index] > scoresPerCandidate[best]) best = index;
  }
  const ranking = Array.from({ length: n }, (_, index) => index).sort(
    (a, b) => scoresPerCandidate[b] - scoresPerCandidate[a] || a - b,
  );
  const { scoreSources, scoreDistribution, paperEquivalent } = summarizeScoredPairs(
    client,
    tasks,
    { [taskName]: [...ring, ...pivotPairs] },
    criteria,
    note,
    nReps,
    scores,
  );
  return {
    index: best,
    best: candidates[best].name?.trim() || `candidate_${best}`,
    scores: scoresPerCandidate,
    ranking,
    nComparisons: ring.length + pivotPairs.length,
    criteria: criteria.map((criterion) => criterion.id),
    scoreSources,
    scoreDistribution,
    paperEquivalent,
    usage: diffUsage(USAGE.snapshot(), usageBefore),
  };
}
