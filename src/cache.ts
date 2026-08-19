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
  renameSync,
  rmdirSync,
  statSync,
  unlinkSync,
  writeSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";

export const CACHE_VERSION = 4;
const LOCK_TIMEOUT_MS = 30_000;
const LOCK_STALE_MS = 120_000;

interface LockHandle {
  lockDir: string;
  token: string;
}

interface LockMetadata {
  pid: number;
  token: string;
  createdAt: number;
}

export interface CachedEntry {
  score_A: number;
  score_B: number;
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
    entry.score_B <= 1
  );
}

function parseCache(text: string): ScoreCache {
  const parsed: unknown = JSON.parse(text);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {};
  }
  const cache: ScoreCache = {};
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (isCachedEntry(value)) cache[key] = value;
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

function readLockMetadata(lockDir: string): LockMetadata | undefined {
  try {
    const parsed: unknown = JSON.parse(readFileSync(join(lockDir, "owner"), "utf8"));
    if (!parsed || typeof parsed !== "object") return undefined;
    const metadata = parsed as Record<string, unknown>;
    if (
      typeof metadata.pid !== "number" ||
      !Number.isInteger(metadata.pid) ||
      metadata.pid < 1 ||
      typeof metadata.token !== "string" ||
      metadata.token.length === 0 ||
      typeof metadata.createdAt !== "number" ||
      !Number.isFinite(metadata.createdAt)
    ) {
      return undefined;
    }
    return {
      pid: metadata.pid,
      token: metadata.token,
      createdAt: metadata.createdAt,
    };
  } catch {
    return undefined;
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** Move a stale lock aside atomically before removing it. */
function reclaimStaleLock(lockDir: string): boolean {
  let lockStat: ReturnType<typeof statSync>;
  try {
    lockStat = statSync(lockDir);
  } catch {
    return true;
  }
  if (Date.now() - lockStat.mtimeMs <= LOCK_STALE_MS) return false;
  const metadata = readLockMetadata(lockDir);
  if (metadata && processIsAlive(metadata.pid)) return false;
  const quarantine = `${lockDir}.reclaim-${process.pid}-${randomBytes(8).toString("hex")}`;
  try {
    renameSync(lockDir, quarantine);
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "ENOENT";
  }
  try {
    unlinkSync(join(quarantine, "owner"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  rmdirSync(quarantine);
  return true;
}

function acquireLock(lockDir: string): LockHandle {
  const started = Date.now();
  mkdirSync(dirname(lockDir), { recursive: true });
  while (true) {
    try {
      mkdirSync(lockDir);
      const token = `${process.pid}:${Date.now()}:${randomBytes(16).toString("hex")}`;
      const fd = openSync(join(lockDir, "owner"), "wx", 0o600);
      try {
        writeAll(
          fd,
          Buffer.from(
            JSON.stringify({ pid: process.pid, token, createdAt: Date.now() }) + "\n",
          ),
        );
        fsyncSync(fd);
        return { lockDir, token };
      } finally {
        closeSync(fd);
      }
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST") {
        try {
          unlinkSync(join(lockDir, "owner"));
          rmdirSync(lockDir);
        } catch {
          // Preserve the original lock initialization error.
        }
        throw error;
      }
      if (reclaimStaleLock(lockDir)) continue;
      if (Date.now() - started >= LOCK_TIMEOUT_MS) {
        throw new Error(`Timed out waiting for cache lock: ${lockDir}`);
      }
      sleepSync(10);
    }
  }
}

function releaseLock(handle: LockHandle): void {
  try {
    const metadata = readLockMetadata(handle.lockDir);
    if (metadata?.token !== handle.token) return;
    unlinkSync(join(handle.lockDir, "owner"));
    rmdirSync(handle.lockDir);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
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
  const lock = acquireLock(`${cacheFile}.lock`);
  try {
    const merged: ScoreCache = { ...readCacheFile(cacheFile) };
    for (const [key, entry] of Object.entries(cache)) {
      if (isCachedEntry(entry)) merged[key] = entry;
    }
    writeAtomic(cacheFile, merged);
  } finally {
    releaseLock(lock);
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
