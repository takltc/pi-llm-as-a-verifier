/**
 * Durable cache for directed verifier scores.
 *
 * A score is valid only for the exact prompt that produced it. The cache key
 * therefore carries a content fingerprint for the task, both candidate
 * traces, verifier configuration, and prompt version. Writes use a lock, a
 * fresh read/merge, and an atomic rename so concurrent benchmark processes
 * preserve one another's entries and readers never observe partial JSON.
 */

import { createHash, randomBytes } from "node:crypto";
import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  realpathSync,
  renameSync,
  statSync,
  unlinkSync,
  writeSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { FileLock } from "@oh-my-pi/pi-natives";

export const CACHE_VERSION = 7;
const LOCK_TIMEOUT_MS = 30_000;
export interface CachedEntry {
  score_A: number;
  score_B: number;
  source_A?: CachedScoreSource;
  source_B?: CachedScoreSource;
  support_A?: number;
  support_B?: number;
  probability_mass_A?: number;
  probability_mass_B?: number;
}

export type CachedScoreSource =
  | "logprobs"
  | "text_fallback"
  | "neutral_tie"
  | "unknown";

export interface ScoreSourceCounts {
  logprobs: number;
  textFallback: number;
  neutralTie: number;
  unknown: number;
}

export interface ScoreDistributionQuality {
  logprobScores: number;
  minSupport: number;
  meanSupport: number;
  minProbabilityMass: number;
  meanProbabilityMass: number;
}

export type ScoreCache = Record<string, CachedEntry>;

/** All inputs that can change the text or token distribution being scored. */
export interface CacheContext {
  criterionId: string;
  criterionName: string;
  criterionDescription: string;
  problem: string;
  traceA: string;
  traceB: string;
  imagesFingerprint: string;
  trajectoryImagesAFingerprint: string;
  trajectoryImagesBFingerprint: string;
  provider: string;
  api: string;
  model: string;
  effort: string;
  maxTokens: number;
  baseUrl: string;
  requestIdentity: string;
  groundTruthNote: string;
  promptVersion: string;
}

export type CacheContextResolver =
  | CacheContext
  | ((criterionId: string, rep: number) => CacheContext | undefined);

/** Stable JSON encoding with sorted object keys for deterministic hashes. */
function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, stableValue(item)]),
    );
  }
  return value;
}

export function stableFingerprint(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(stableValue(value)))
    .digest("hex")
    .slice(0, 32);
}

/**
 * Directed comparisons keep (a,b) and (b,a) separate. The optional context
 * prevents stale scores from being reused after inputs, prompts, or models
 * change. The five-argument form remains readable for callers without a
 * context, while production scoring paths pass one.
 */
export function cacheKey(
  critId: string,
  taskName: string,
  a: number,
  b: number,
  rep: number,
  context?: CacheContext,
): string {
  const base = `${critId}|${taskName}|${a},${b}|${rep}`;
  return context
    ? `v${CACHE_VERSION}|${base}|${stableFingerprint(context)}`
    : base;
}

function isCachedEntry(value: unknown): value is CachedEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.score_A === "number" &&
    Number.isFinite(entry.score_A) &&
    entry.score_A >= 0 &&
    entry.score_A <= 1 &&
    typeof entry.score_B === "number" &&
    Number.isFinite(entry.score_B) &&
    entry.score_B >= 0 &&
    entry.score_B <= 1 &&
    isCachedScoreSource(entry.source_A) &&
    isCachedScoreSource(entry.source_B) &&
    isOptionalNonNegativeNumber(entry.support_A) &&
    isOptionalNonNegativeNumber(entry.support_B) &&
    isOptionalNonNegativeNumber(entry.probability_mass_A) &&
    isOptionalNonNegativeNumber(entry.probability_mass_B)
  );
}

function isDurableCachedEntry(value: unknown): value is CachedEntry {
  return isCachedEntry(value) &&
    value.source_A !== "neutral_tie" &&
    value.source_B !== "neutral_tie";
}

function isCachedScoreSource(value: unknown): value is CachedScoreSource | undefined {
  return value === undefined || value === "logprobs" || value === "text_fallback" ||
    value === "neutral_tie" || value === "unknown";
}

function isOptionalNonNegativeNumber(value: unknown): value is number | undefined {
  return value === undefined ||
    (typeof value === "number" && Number.isFinite(value) && value >= 0);
}

function resolvedScoreSource(
  entry: CachedEntry | undefined,
  side: "A" | "B",
): CachedScoreSource {
  const source = side === "A" ? entry?.source_A : entry?.source_B;
  if (source !== "logprobs") return source ?? "unknown";
  const support = side === "A" ? entry?.support_A : entry?.support_B;
  const mass = side === "A" ? entry?.probability_mass_A : entry?.probability_mass_B;
  return typeof support === "number" && Number.isInteger(support) && support >= 1 &&
      typeof mass === "number" && Number.isFinite(mass) && mass > 0
    ? "logprobs"
    : "unknown";
}

/** Count score-tag evidence sources for theory-equivalence telemetry. */
export function summarizeScoreSources(
  entries: Iterable<CachedEntry | undefined>,
): ScoreSourceCounts {
  const counts: ScoreSourceCounts = {
    logprobs: 0,
    textFallback: 0,
    neutralTie: 0,
    unknown: 0,
  };
  const add = (source: CachedScoreSource): void => {
    if (source === "logprobs") counts.logprobs += 1;
    else if (source === "text_fallback") counts.textFallback += 1;
    else if (source === "neutral_tie") counts.neutralTie += 1;
    else counts.unknown += 1;
  };
  for (const entry of entries) {
    add(resolvedScoreSource(entry, "A"));
    add(resolvedScoreSource(entry, "B"));
  }
  return counts;
}

/** Aggregate returned A-T support and probability mass for logprob-backed tags. */
export function summarizeScoreDistribution(
  entries: Iterable<CachedEntry | undefined>,
): ScoreDistributionQuality {
  let count = 0;
  let minSupport = Infinity;
  let minMass = Infinity;
  let totalSupport = 0;
  let totalMass = 0;
  for (const entry of entries) {
    for (const side of ["A", "B"] as const) {
      if (resolvedScoreSource(entry, side) !== "logprobs") continue;
      const support = (side === "A" ? entry?.support_A : entry?.support_B)!;
      const mass = (side === "A" ? entry?.probability_mass_A : entry?.probability_mass_B)!;
      count += 1;
      minSupport = Math.min(minSupport, support);
      minMass = Math.min(minMass, mass);
      totalSupport += support;
      totalMass += mass;
    }
  }
  return {
    logprobScores: count,
    minSupport: count ? minSupport : 0,
    meanSupport: count ? totalSupport / count : 0,
    minProbabilityMass: count ? minMass : 0,
    meanProbabilityMass: count ? totalMass / count : 0,
  };
}

function parseCache(text: string): ScoreCache {
  const parsed: unknown = JSON.parse(text);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {};
  }
  const cache: ScoreCache = {};
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (isDurableCachedEntry(value)) cache[key] = value;
  }
  return cache;
}

function readCacheFile(cacheFile: string): ScoreCache {
  try {
    return parseCache(readFileSync(cacheFile, "utf8"));
  } catch {
    return {};
  }
}

export function loadCache(cacheFile?: string): ScoreCache {
  return cacheFile ? readCacheFile(cacheFile) : {};
}

function writeAll(fd: number, data: Buffer): void {
  let offset = 0;
  while (offset < data.length) {
    const written = writeSync(fd, data, offset, data.length - offset, offset);
    if (written <= 0) throw new Error("Unable to write cache data");
    offset += written;
  }
}

function sleepSync(milliseconds: number): void {
  const signal = new Int32Array(new SharedArrayBuffer(4));
  Atomics.wait(signal, 0, 0, milliseconds);
}

/** OS ownership survives contention and is released automatically on process exit. */
function acquireLock(cacheFile: string): FileLock {
  const lockPath = join(realpathSync(dirname(cacheFile)), `${basename(cacheFile)}.lock`);
  try {
    if (statSync(lockPath).isDirectory()) {
      throw new Error(
        `Legacy cache lock directory: ${lockPath}. Stop old verifier processes before removing it and retrying`,
      );
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  const started = Date.now();
  while (true) {
    // Keep the lock file in place: unlinking it would split Unix inode ownership.
    const lock = FileLock.tryAcquire(lockPath);
    if (lock.acquired) return lock;
    if (Date.now() - started >= LOCK_TIMEOUT_MS) {
      throw new Error(`Timed out waiting for cache lock: ${lockPath}`);
    }
    sleepSync(10);
  }
}

function writeAtomic(cacheFile: string, cache: ScoreCache): void {
  const directory = dirname(cacheFile);
  mkdirSync(directory, { recursive: true });
  const tempFile = join(
    directory,
    `.${basename(cacheFile)}.tmp-${process.pid}-${randomBytes(8).toString("hex")}`,
  );
  const payload = `${JSON.stringify(cache)}\n`;
  const payloadBuffer = Buffer.from(payload);
  let fd: number | undefined;
  try {
    fd = openSync(tempFile, "wx", 0o600);
    writeAll(fd, payloadBuffer);
    fsyncSync(fd);
    closeSync(fd);
    fd = undefined;
    renameSync(tempFile, cacheFile);
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {
        // Preserve the original write error.
      }
    }
    try {
      unlinkSync(tempFile);
    } catch {
      // The rename succeeded or the temporary file was never created.
    }
  }
}

/** Merge with the latest on-disk state while holding an inter-process lock. */
export function saveCache(cacheFile: string, cache: ScoreCache): void {
  mkdirSync(dirname(cacheFile), { recursive: true });
  const lock = acquireLock(cacheFile);
  try {
    const merged: ScoreCache = { ...readCacheFile(cacheFile) };
    for (const [key, entry] of Object.entries(cache)) {
      if (isDurableCachedEntry(entry)) merged[key] = entry;
    }
    writeAtomic(cacheFile, merged);
  } finally {
    lock.release();
  }
}

export function mergeCaches(...caches: ScoreCache[]): ScoreCache {
  const merged: ScoreCache = {};
  for (const cache of caches) {
    for (const [key, entry] of Object.entries(cache)) {
      if (isCachedEntry(entry)) merged[key] = entry;
    }
  }
  return merged;
}

/** Fine-grained rewards for a directed comparison, averaged over criteria and repeats. */
export function directedReward(
  scores: ScoreCache,
  taskName: string,
  a: number,
  b: number,
  criteriaIds: string[],
  nReps: number,
  context?: CacheContextResolver,
): [number, number] {
  if (a === b) return [0.5, 0.5];
  let sa = 0;
  let sb = 0;
  let count = 0;
  for (const criterionId of criteriaIds) {
    for (let rep = 0; rep < nReps; rep++) {
      const resolvedContext =
        typeof context === "function"
          ? context(criterionId, rep)
          : context;
      const entry =
        scores[cacheKey(criterionId, taskName, a, b, rep, resolvedContext)] ?? {
          score_A: 0.5,
          score_B: 0.5,
        };
      sa += entry.score_A;
      sb += entry.score_B;
      count += 1;
    }
  }
  return count > 0 ? [sa / count, sb / count] : [0.5, 0.5];
}
