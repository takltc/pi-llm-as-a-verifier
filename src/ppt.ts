/**
 * Probabilistic Pivot Tournament (PPT): O(Nk) best-of-N selection.
 *
 * Instead of a full O(N^2) round-robin, PPT selects the best candidate in
 * three steps (paper §3.2 and Appendix B.2):
 *
 *   1) Ring pass: score the N adjacent directed pairs of a random
 *      Hamiltonian cycle. Every candidate appears once in each prompt slot,
 *      so the verifier's slot bias cancels around the ring.
 *   2) Pivot selection: the top-k candidates by ring-pass mean preference
 *      w_i / c_i become the pivot set P.
 *   3) Pivot rounds: score every non-pivot-vs-pivot and pivot-vs-pivot
 *      pair that is absent from the directed ring; aggregate all unique
 *      comparisons into w_i, c_i and return argmax_i w_i / c_i.
 *
 * N + k(N - k) + C(k, 2) is the paper's O(Nk) upper bound. Algorithm 1
 * removes directed ring overlaps from the pivot edge set, so the exact count
 * can be lower by |E_ring ∩ E_piv|.
 * Each comparison's rewards (R_a, R_b) become a soft win via Bradley-Terry,
 * p(a beats b) = sigmoid(R_a - R_b).
 */

export function ringCycle(n: number, rng: () => number): Array<[number, number]> {
  if (n <= 1) return [];
  const perm = Array.from({ length: n }, (_, i) => i);
  // Fisher-Yates shuffle with an injected RNG (deterministic given seed).
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  return perm.map((v, t) => [v, perm[(t + 1) % n]] as [number, number]);
}

export function bradleyTerry(ra: number, rb: number): number {
  return 1 / (1 + Math.exp(-(ra - rb)));
}

export type DirectedScore = (a: number, b: number) => [ra: number, rb: number];

export function accumulate(
  pairs: Array<[number, number]>,
  score: DirectedScore,
  w: number[],
  c: number[],
): void {
  for (const [a, b] of pairs) {
    const [ra, rb] = score(a, b);
    const p = bradleyTerry(ra, rb);
    w[a] += p;
    c[a] += 1;
    w[b] += 1 - p;
    c[b] += 1;
  }
}

export function selectPivots(w: number[], c: number[], k: number): number[] {
  const n = w.length;
  const kk = Math.min(k, n);
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => {
    const wi = c[i] ? w[i] / c[i] : 0;
    const wj = c[j] ? w[j] / c[j] : 0;
    return wj - wi || i - j;
  });
  return order.slice(0, kk);
}

export function pivotRoundPairs(
  n: number,
  pivots: number[],
  ring: Array<[number, number]> = [],
): Array<[number, number]> {
  const pivotSet = new Set(pivots);
  const nonPivots = Array.from({ length: n }, (_, i) => i).filter(
    (i) => !pivotSet.has(i),
  );
  const pairs: Array<[number, number]> = [];
  for (const i of nonPivots) {
    for (const p of pivots) pairs.push([i, p]);
  }
  const sorted = [...pivots].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) {
      pairs.push([sorted[i], sorted[j]]);
    }
  }
  const ringPairs = new Set(ring.map(([a, b]) => `${a},${b}`));
  return pairs.filter(([a, b]) => !ringPairs.has(`${a},${b}`));
}

export interface PPTResult {
  bestIndex: number;
  nComparisons: number;
  w: number[];
  c: number[];
}

/** Run the full PPT given a pre-sampled ring and a directed score fn. */
export function selectBest(
  n: number,
  ring: Array<[number, number]>,
  k: number,
  score: DirectedScore,
): PPTResult {
  const w = new Array<number>(n).fill(0);
  const c = new Array<number>(n).fill(0);

  accumulate(ring, score, w, c);

  const pivots = selectPivots(w, c, k);

  const prPairs = pivotRoundPairs(n, pivots, ring);
  accumulate(prPairs, score, w, c);

  let best = 0;
  for (let i = 1; i < n; i++) {
    const bi = c[i] ? w[i] / c[i] : 0;
    const bb = c[best] ? w[best] / c[best] : 0;
    if (bi > bb) best = i;
  }
  return {
    bestIndex: best,
    nComparisons: ring.length + prPairs.length,
    w,
    c,
  };
}
