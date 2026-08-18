/**
 * On-disk score cache.
 *
 * Comparisons are *directed*: (a, b) and (b, a) are distinct cache entries,
 * which the ring pass relies on to cancel the verifier's slot bias. Odd
 * reps swap the prompt slots, so with K >= 2 the bias also cancels within
 * one directed comparison. "score_A"/"score_B" always mean candidate a's /
 * b's reward, whichever slot they occupied.
 */

import { readFileSync, writeFileSync, readdirSync } from "node:fs";

export interface CachedEntry {
  score_A: number;
  score_B: number;
}

export type ScoreCache = Record<string, CachedEntry>;

export function cacheKey(
  critId: string,
  taskName: string,
  a: number,
  b: number,
  rep: number,
): string {
  return `${critId}|${taskName}|${a},${b}|${rep}`;
}

export function loadCache(cacheFile?: string): ScoreCache {
  if (!cacheFile) return {};
  try {
    return JSON.parse(readFileSync(cacheFile, "utf8")) as ScoreCache;
  } catch {
    return {};
  }
}

export function saveCache(cacheFile: string, cache: ScoreCache): void {
  writeFileSync(cacheFile, JSON.stringify(cache));
}

/** Fine-grained rewards (R_a, R_b) for the directed comparison (a, b),
 * averaged over criteria and repeats. Missing entries default to 0.5. */
export function directedReward(
  scores: ScoreCache,
  taskName: string,
  a: number,
  b: number,
  criteriaIds: string[],
  nReps: number,
): [number, number] {
  if (a === b) return [0.5, 0.5];
  let sa = 0;
  let sb = 0;
  let cnt = 0;
  for (const cid of criteriaIds) {
    for (let rep = 0; rep < nReps; rep++) {
      const entry = scores[cacheKey(cid, taskName, a, b, rep)] ?? {
        score_A: 0.5,
        score_B: 0.5,
      };
      sa += entry.score_A;
      sb += entry.score_B;
      cnt += 1;
    }
  }
  return cnt > 0 ? [sa / cnt, sb / cnt] : [0.5, 0.5];
}
