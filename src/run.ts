/**
 * End-to-end LLM-as-a-Verifier pipeline.
 *
 * This is the two-phase Terminal-Bench self-verification flow from the
 * reference implementation: a deterministic ring pass chooses pivots, then
 * pivot rounds are scored and aggregated with the ring scores.
 */

import {
  VerifierClient,
  USAGE,
  diffUsage,
  type UsageSnapshot,
} from "./client.ts";
import {
  GROUND_TRUTH_NOTE,
  PROMPT_VERSION,
  buildPrompt,
  validateCriteria,
  type Criterion,
} from "./prompt.ts";
import { extractScore, hasExtractableScore, type VerifierReply } from "./scale.ts";
import { bradleyTerry, pivotRoundPairs, ringCycle, selectPivots } from "./ppt.ts";
import {
  cacheKey,
  directedReward,
  loadCache,
  mergeCaches,
  saveCache,
  stableFingerprint,
  type CacheContext,
  type ScoreCache,
} from "./cache.ts";
import type { Tasks, Trial } from "./loader.ts";
import { classify } from "./loader.ts";

export interface VerifyOptions {
  pivots?: number;
  nEvaluations?: number;
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
  criterion: Criterion;
  swap: boolean;
  prefix: string;
  context: CacheContext;
}

export const SELF_VERIFICATION_DEFAULTS = {
  pivots: 1,
  nEvaluations: 2,
  seed: 0,
  maxWorkers: 8,
} as const;

export function defaultMaxWorkers(opts: VerifyOptions): number {
  return opts.maxWorkers ?? SELF_VERIFICATION_DEFAULTS.maxWorkers;
}

function positiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  if (!Number.isInteger(resolved) || resolved < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return resolved;
}

function validateTasks(tasks: Tasks): void {
  if (!tasks || typeof tasks !== "object" || Object.keys(tasks).length === 0) {
    throw new Error("At least one task is required");
  }
  for (const [taskName, trials] of Object.entries(tasks)) {
    if (!taskName.trim()) throw new Error("Task names must be non-empty");
    if (!Array.isArray(trials) || trials.length < 2) {
      throw new Error(`Task ${taskName} needs at least two trials`);
    }
    for (const [index, trial] of trials.entries()) {
      if (!trial || typeof trial !== "object") {
        throw new Error(`Task ${taskName} trial ${index} is invalid`);
      }
      if (trial.reward !== 0 && trial.reward !== 1) {
        throw new Error(`Task ${taskName} trial ${index} reward must be 0 or 1`);
      }
      if (typeof trial.problem !== "string" || typeof trial.trace !== "string") {
        throw new Error(`Task ${taskName} trial ${index} problem and trace must be strings`);
      }
      if (!trial.problem.trim() || !trial.trace.trim()) {
        throw new Error(`Task ${taskName} trial ${index} has empty problem or trace`);
      }
    }
  }
}

export function cacheContext(
  client: VerifierClient,
  trials: Trial[],
  a: number,
  b: number,
  rep: number,
  groundTruthNote: string,
  criterion: Criterion,
): CacheContext {
  const swap = rep % 2 === 1;
  const slotA = swap ? b : a;
  const slotB = swap ? a : b;
  return {
    criterionId: criterion.id,
    criterionName: criterion.name,
    criterionDescription: criterion.description,
    problem: trials[a].problem,
    traceA: trials[slotA].trace,
    traceB: trials[slotB].trace,
    provider: String(client.provider ?? ""),
    api: String(client.api ?? ""),
    model: String(client.model ?? ""),
    effort: String(client.effort ?? ""),
    maxTokens: Number(client.maxTokens ?? 0),
    baseUrl: String(client.baseUrl ?? ""),
    requestIdentity: String(client.requestIdentity ?? ""),
    groundTruthNote,
    promptVersion: PROMPT_VERSION,
  };
}

export function contextResolver(
  client: VerifierClient,
  trials: Trial[],
  taskName: string,
  a: number,
  b: number,
  groundTruthNote: string,
  criteria: Criterion[],
): (criterionId: string, rep: number) => CacheContext {
  const byId = new Map(criteria.map((criterion) => [criterion.id, criterion]));
  return (criterionId, rep) => {
    const criterion = byId.get(criterionId);
    if (!criterion) throw new Error(`Unknown criterion: ${criterionId}`);
    return cacheContext(client, trials, a, b, rep, groundTruthNote, criterion);
  };
}

function ensurePair(trials: Trial[], a: number, b: number, taskName: string): void {
  if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a >= trials.length || b >= trials.length || a === b) {
    throw new Error(`Invalid comparison pair ${taskName}:${a},${b}`);
  }
}

/**
 * Score every requested directed pair. Failed calls produce an in-memory tie
 * and remain absent from persistent cache, so transient outages cannot become
 * durable evidence.
 */
export async function scoreDirectedPairs(
  client: VerifierClient,
  tasks: Tasks,
  neededPairs: Record<string, Array<[number, number]>>,
  criteria: Criterion[],
  groundTruthNote: string = GROUND_TRUTH_NOTE,
  nReps = 2,
  maxWorkers: number = SELF_VERIFICATION_DEFAULTS.maxWorkers,
  cacheFile?: string,
  opts: {
    onError?: "tie" | "raise";
    progress?: boolean;
    signal?: AbortSignal;
    initialCache?: ScoreCache;
  } = {},
): Promise<ScoreCache> {
  validateTasks(tasks);
  validateCriteria(criteria);
  const repetitions = positiveInteger(
    nReps,
    SELF_VERIFICATION_DEFAULTS.nEvaluations,
    "nEvaluations",
  );
  const workersLimit = positiveInteger(
    maxWorkers,
    SELF_VERIFICATION_DEFAULTS.maxWorkers,
    "maxWorkers",
  );
  const note = groundTruthNote ?? GROUND_TRUTH_NOTE;
  const cached = mergeCaches(loadCache(cacheFile), opts.initialCache ?? {});
  const jobs: Job[] = [];
  const requested = new Set<string>();

  for (const [taskName, pairs] of Object.entries(neededPairs).sort(([a], [b]) => a.localeCompare(b))) {
    const trials = tasks[taskName];
    if (!trials) throw new Error(`Unknown task in comparison set: ${taskName}`);
    for (const [a, b] of pairs) {
      ensurePair(trials, a, b, taskName);
      for (const criterion of criteria) {
        for (let rep = 0; rep < repetitions; rep++) {
          const context = cacheContext(client, trials, a, b, rep, note, criterion);
          const key = cacheKey(criterion.id, taskName, a, b, rep, context);
          if (requested.has(key) || cached[key]) continue;
          requested.add(key);
          const swap = rep % 2 === 1;
          const slotA = swap ? b : a;
          const slotB = swap ? a : b;
          jobs.push({
            key,
            problem: trials[a].problem,
            traceA: trials[slotA].trace,
            traceB: trials[slotB].trace,
            criterion,
            swap,
            context,
            prefix: stableFingerprint({
              problem: trials[a].problem,
              traceA: trials[slotA].trace,
              traceB: trials[slotB].trace,
              provider: context.provider,
              api: context.api,
              model: context.model,
              effort: context.effort,
              maxTokens: context.maxTokens,
              baseUrl: context.baseUrl,
              requestIdentity: context.requestIdentity,
              groundTruthNote: note,
              promptVersion: PROMPT_VERSION,
            }),
          });
        }
      }
    }
  }

  const log = opts.progress === false ? (_message: string) => {} : console.log;
  if (jobs.length === 0) {
    log(`  All scores cached (${Object.keys(cached).length} entries)`);
    return cached;
  }

  const seenPrefixes = new Set<string>();
  const warm: Job[] = [];
  const rest: Job[] = [];
  for (const job of jobs) {
    if (seenPrefixes.has(job.prefix)) rest.push(job);
    else {
      seenPrefixes.add(job.prefix);
      warm.push(job);
    }
  }
  log(
    `  ${jobs.length} scoring jobs (${Object.keys(cached).length} cached); warming ${warm.length} prefixes`,
  );

  const results: ScoreCache = mergeCaches(cached);
  let errors = 0;
  let completed = 0;
  let firstError: unknown;
  const executionAbort = new AbortController();
  const externalAbort = opts.signal;
  const abortReason = (error?: unknown): unknown =>
    error ?? new DOMException("The operation was aborted", "AbortError");
  const abortExecution = (error?: unknown): void => {
    if (!executionAbort.signal.aborted) executionAbort.abort(abortReason(error));
  };
  const onExternalAbort = (): void => {
    firstError ??= abortReason(externalAbort?.reason);
    abortExecution(firstError);
  };
  if (externalAbort?.aborted) onExternalAbort();
  else externalAbort?.addEventListener("abort", onExternalAbort, { once: true });

  async function scoreOne(job: Job): Promise<void> {
    if (executionAbort.signal.aborted) {
      throw abortReason(firstError ?? executionAbort.signal.reason);
    }
    const prompt = buildPrompt(
      job.problem,
      job.traceA,
      job.traceB,
      job.criterion,
      note,
    );
    let reply: VerifierReply;
    try {
      reply = await client.scoreReply(prompt, { signal: executionAbort.signal });
      if (!hasExtractableScore(reply, "<score_A>") || !hasExtractableScore(reply, "<score_B>")) {
        throw new Error("Verifier response did not contain usable score tags");
      }
    } catch (error) {
      if (executionAbort.signal.aborted) {
        throw abortReason(firstError ?? error);
      }
      if (opts.onError === "raise") {
        firstError ??= error;
        abortExecution(firstError);
        throw error;
      }
      errors += 1;
      results[job.key] = { score_A: 0.5, score_B: 0.5 };
      if (errors <= 3) log(`\n  Error: ${String(error)}`);
      return;
    }
    let scoreA = extractScore(reply, "<score_A>");
    let scoreB = extractScore(reply, "<score_B>");
    if (!Number.isFinite(scoreA) || !Number.isFinite(scoreB)) {
      if (opts.onError === "raise") {
        const error = new Error("Verifier returned non-finite scores");
        firstError ??= error;
        abortExecution(firstError);
        throw error;
      }
      errors += 1;
      results[job.key] = { score_A: 0.5, score_B: 0.5 };
      return;
    }
    if (job.swap) [scoreA, scoreB] = [scoreB, scoreA];
    const entry = { score_A: scoreA, score_B: scoreB };
    cached[job.key] = entry;
    results[job.key] = entry;
  }

  async function runPhase(phaseJobs: Job[]): Promise<void> {
    if (phaseJobs.length === 0) return;
    let next = 0;
    const workers = Math.min(workersLimit, phaseJobs.length);
    const checkpoint = Math.max(1, Math.floor(phaseJobs.length / 20));
    async function worker(): Promise<void> {
      while (true) {
        if (executionAbort.signal.aborted) return;
        const index = next++;
        if (index >= phaseJobs.length) return;
        try {
          await scoreOne(phaseJobs[index]);
          completed += 1;
          if (cacheFile && completed % checkpoint === 0 && Object.keys(cached).length > 0) {
            saveCache(cacheFile, cached);
          }
        } catch (error) {
          firstError ??= error;
          abortExecution(firstError);
          return;
        }
      }
    }
    await Promise.allSettled(Array.from({ length: workers }, () => worker()));
    if (firstError) throw firstError;
  }

  try {
    if (firstError) throw firstError;
    await runPhase(warm);
    await runPhase(rest);
  } finally {
    externalAbort?.removeEventListener("abort", onExternalAbort);
    if (cacheFile && Object.keys(cached).length > 0) saveCache(cacheFile, cached);
  }
  log(`  Done (${errors} errors)`);
  return results;
}

export function validateVerifyOptions(
  tasks: Tasks,
  criteria: Criterion[],
  opts: VerifyOptions,
): { k: number; nReps: number; seed: number; maxWorkers: number } {
  validateTasks(tasks);
  validateCriteria(criteria);
  const maxCandidates = Math.max(...Object.values(tasks).map((trials) => trials.length));
  const requestedK = positiveInteger(
    opts.pivots,
    SELF_VERIFICATION_DEFAULTS.pivots,
    "pivots",
  );
  const k = Math.min(requestedK, maxCandidates);
  const nReps = positiveInteger(
    opts.nEvaluations,
    SELF_VERIFICATION_DEFAULTS.nEvaluations,
    "nEvaluations",
  );
  const maxWorkers = positiveInteger(
    opts.maxWorkers,
    SELF_VERIFICATION_DEFAULTS.maxWorkers,
    "maxWorkers",
  );
  const seed = opts.seed ?? SELF_VERIFICATION_DEFAULTS.seed;
  if (!Number.isInteger(seed)) throw new Error("seed must be an integer");
  return { k, nReps, seed, maxWorkers };
}

/** Score one benchmark and return metrics plus per-task winners. */
export async function runBenchmark(
  tasks: Tasks,
  criteria: Criterion[],
  opts: VerifyOptions = {},
): Promise<RunStats> {
  const { k, nReps, seed, maxWorkers } = validateVerifyOptions(tasks, criteria, opts);
  const note = opts.groundTruthNote === undefined ? GROUND_TRUTH_NOTE : opts.groundTruthNote;
  const client = opts.client;
  if (!client) {
    throw new Error("runBenchmark requires a verifier client for the OMP default model.");
  }
  const criteriaIds = criteria.map((criterion) => criterion.id);
  const usageBefore = USAGE.snapshot();
  const { allPass, swing } = classify(tasks);
  const nTasks = Object.keys(tasks).length;
  const nRuns = Math.max(...Object.values(tasks).map((trials) => trials.length));

  console.log(
    `  tasks=${nTasks}  all-pass=${allPass.length}  swing=${swing.length}  N(trials)=${nRuns}`,
  );
  console.log(
    `  criteria=${criteriaIds.join(",")}  K=${nReps}  pivots=${k}  seed=${seed}  max_workers=${maxWorkers}`,
  );

  const rng = mulberry32(seed);
  const rings: Record<string, Array<[number, number]>> = {};
  for (const task of swing) rings[task] = ringCycle(tasks[task].length, rng);

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

  const directed = (scoreCache: ScoreCache, taskName: string, a: number, b: number): [number, number] =>
    directedReward(
      scoreCache,
      taskName,
      a,
      b,
      criteriaIds,
      nReps,
      contextResolver(client, tasks[taskName], taskName, a, b, note, criteria),
    );

  const pivotPairs: Record<string, Array<[number, number]>> = {};
  for (const task of swing) {
    const n = tasks[task].length;
    const w = new Array<number>(n).fill(0);
    const c = new Array<number>(n).fill(0);
    for (const [a, b] of rings[task]) {
      const [ra, rb] = directed(scores, task, a, b);
      const preference = bradleyTerry(ra, rb);
      w[a] += preference;
      c[a] += 1;
      w[b] += 1 - preference;
      c[b] += 1;
    }
    pivotPairs[task] = pivotRoundPairs(n, selectPivots(w, c, Math.min(k, n)));
  }

  console.log("Phase B: pivot rounds");
  const phaseB = await scoreDirectedPairs(
    client,
    tasks,
    pivotPairs,
    criteria,
    note,
    nReps,
    maxWorkers,
    opts.cacheFile,
    { ...opts, initialCache: scores },
  );
  scores = mergeCaches(scores, phaseB);

  let selected = 0;
  let totalComparisons = 0;
  const bestPerTask: RunStats["bestPerTask"] = {};
  for (const task of swing) {
    const n = tasks[task].length;
    const w = new Array<number>(n).fill(0);
    const c = new Array<number>(n).fill(0);
    for (const [a, b] of [...rings[task], ...pivotPairs[task]]) {
      const [ra, rb] = directed(scores, task, a, b);
      const preference = bradleyTerry(ra, rb);
      w[a] += preference;
      c[a] += 1;
      w[b] += 1 - preference;
      c[b] += 1;
    }
    let best = 0;
    for (let index = 1; index < n; index++) {
      const current = c[index] ? w[index] / c[index] : 0;
      const incumbent = c[best] ? w[best] / c[best] : 0;
      if (current > incumbent) best = index;
    }
    totalComparisons += rings[task].length + pivotPairs[task].length;
    if (tasks[task][best].reward === 1) selected += 1;
    bestPerTask[task] = {
      index: best,
      reward: tasks[task][best].reward,
      w: w[best],
      c: c[best],
    };
  }

  const pass1 = allPass.length + swing.reduce(
    (sum, taskName) =>
      sum + tasks[taskName].reduce((taskSum, trial) => taskSum + trial.reward, 0) / tasks[taskName].length,
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
    usage: diffUsage(USAGE.snapshot(), usageBefore),
    bestPerTask,
  };
}

export function renderReport(stats: RunStats): string {
  const { nTasks, nRuns, swing, criteriaIds, nReps, k, seed } = stats;
  const denominator = Math.max(1, nTasks);
  return [
    "",
    "=".repeat(72),
    "SELF-VERIFICATION  (LLM-as-a-Verifier)",
    `  g20  criteria=${criteriaIds.join(",")}  K=${nReps}  pivots=${k}  seed=${seed}`,
    `  tasks=${nTasks}  swing=${swing.length}  N(trials)=${nRuns}  comparisons/task=${stats.avgComparisons.toFixed(1)}`,
    "=".repeat(72),
    `${"Method".padEnd(26)}  ${"Score".padStart(14)}  ${"Rate".padStart(7)}`,
    "-".repeat(72),
    `${"Pass@1".padEnd(26)}  ${stats.pass1.toFixed(2).padStart(14)}  ${((100 * stats.pass1) / denominator).toFixed(1).padStart(6)}%`,
    `${"LLM-as-a-Verifier".padEnd(26)}  ${String(stats.verifier).padStart(14)}  ${((100 * stats.verifier) / denominator).toFixed(1).padStart(6)}%`,
    `${`Oracle (Bo${nRuns})`.padEnd(26)}  ${String(stats.oracle).padStart(14)}  ${((100 * stats.oracle) / denominator).toFixed(1).padStart(6)}%`,
    "-".repeat(72),
  ].join("\n");
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
