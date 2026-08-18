/**
 * Single-task best-of-N selection (public `select` API).
 *
 * Scores directed pairs with the fine-grained reward and aggregates them
 * with a Probabilistic Pivot Tournament — O(Nk) verifier comparisons
 * instead of a full O(N²) round-robin. Identical inputs with the same
 * `seed` run the identical tournament.
 */

import { VerifierClient, USAGE, type UsageSnapshot } from "./client.ts";
import { TERMINAL_BENCH_CRITERIA, type Criterion } from "./prompt.ts";
import { extractScore, type VerifierReply } from "./scale.ts";
import { ringCycle, bradleyTerry, selectPivots, pivotRoundPairs } from "./ppt.ts";
import { directedReward } from "./cache.ts";
import { mulberry32, scoreDirectedPairs } from "./run.ts";
export interface Candidate {
  name: string;
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
  scores: number[]; // per-candidate mean preference w_i / c_i
  ranking: number[];
  nComparisons: number;
  criteria: string[];
  usage: UsageSnapshot;
}

export function select(
  problem: string,
  candidates: Candidate[],
  opts: SelectOptions = {},
): Promise<SelectResult> {
  if (candidates.length === 0) {
    throw new Error("select: no candidates given");
  }
  const criteria = normalizeCriteriaArg(opts.criteria);
  const criteriaIds = criteria.map((c) => c.id);
  const k = opts.pivots ?? 2;
  const nReps = opts.nEvaluations ?? 2;
  const seed = opts.seed ?? 0;
  const maxWorkers = opts.maxWorkers ?? 16;
  const note = opts.groundTruthNote ?? "";
  const client = opts.client ?? new VerifierClient();
  const taskName = opts.taskName ?? "task";
  const rng = mulberry32(seed);

  return (async () => {
    // One task with N trials; rewards are unknown here, so keep the task in
    // the swing set to force a full PPT regardless of outcome.
    const tasks = {
      [taskName]: candidates.map((c, i) => ({
        trialName: c.name || `candidate_${i}`,
        reward: 0 as const,
        problem,
        trace: c.trace,
      })),
    };
    const ring = ringCycle(candidates.length, rng);

    console.log(`Phase A: ring pass (${ring.length} comparisons)`);
    let scores = await scoreDirectedPairs(
      client,
      tasks,
      { [taskName]: ring },
      criteria,
      note,
      nReps,
      maxWorkers,
      opts.cacheFile,
      { onError: opts.onError, progress: opts.progress, signal: opts.signal },
    );

    const directed = (a: number, b: number): [number, number] =>
      directedReward(scores, taskName, a, b, criteriaIds, nReps);

    const n = candidates.length;
    const w = new Array<number>(n).fill(0);
    const c = new Array<number>(n).fill(0);
    const accumulate = (pairs: Array<[number, number]>) => {
      for (const [a, b] of pairs) {
        const [ra, rb] = directed(a, b);
        const p = bradleyTerry(ra, rb);
        w[a] += p; c[a] += 1; w[b] += 1 - p; c[b] += 1;
      }
    };
    accumulate(ring);

    const pivots = selectPivots(w, c, k);
    const prPairs = pivotRoundPairs(n, pivots);
    console.log(`Phase B: pivot rounds (${prPairs.length} comparisons)`);
    scores = await scoreDirectedPairs(
      client,
      tasks,
      { [taskName]: prPairs },
      criteria,
      note,
      nReps,
      maxWorkers,
      opts.cacheFile,
      { onError: opts.onError, progress: opts.progress, signal: opts.signal },
    );
    accumulate(prPairs);

    let best = 0;
    const scoresPerCandidate = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      scoresPerCandidate[i] = c[i] ? w[i] / c[i] : 0;
      // Strictly greater wins, so ties keep the earlier index — same
      // tie-break as the reference (`max(range(n), key=(w/c, -i))` picks
      // the lower index too).
      if (i > 0 && scoresPerCandidate[i] > scoresPerCandidate[best]) {
        best = i;
      }
    }
    const ranking = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => scoresPerCandidate[b] - scoresPerCandidate[a] || a - b,
    );
    return {
      index: best,
      best: candidates[best].name,
      scores: scoresPerCandidate,
      ranking,
      nComparisons: ring.length + prPairs.length,
      criteria: criteriaIds,
      usage: USAGE.snapshot(),
    };
  })();
}

function normalizeCriteriaArg(
  criteria: SelectOptions["criteria"],
): Criterion[] {
  if (criteria === undefined) return TERMINAL_BENCH_CRITERIA;
  if (Array.isArray(criteria)) {
    return criteria.map((c) =>
      typeof c === "string"
        ? { id: c.toLowerCase().replace(/\s+/g, "_"), name: c, description: c }
        : c,
    );
  }
  if (typeof criteria === "string") {
    if (criteria === "terminal_bench" || criteria === "terminal_bench_2.1") {
      return TERMINAL_BENCH_CRITERIA;
    }
    return [{ id: criteria, name: criteria, description: criteria }];
  }
  return Object.entries(criteria).map(([name, description]) => ({
    id: name.toLowerCase().replace(/\s+/g, "_"),
    name,
    description,
  }));
}
