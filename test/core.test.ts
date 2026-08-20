import { describe, expect, test } from "bun:test";
import {
  extractScore,
  extractScorePair,
  findTagLogprobs,
  GRANULARITY,
  hasExtractableScore,
  SCALE,
  type PositionLogprobs,
  type VerifierReply,
} from "../src/scale.ts";
import { accumulate, bradleyTerry, pivotRoundPairs, ringCycle, selectBest, selectPivots } from "../src/ppt.ts";
import {
  CACHE_VERSION,
  cacheKey,
  directedReward,
  type CacheContext,
  type ScoreCache,
} from "../src/cache.ts";
import { mulberry32, scoreDirectedPairs } from "../src/run.ts";
import type { VerifierClient } from "../src/client.ts";


/**
 * Naive reference: accumulate the joined text token-by-token and, after each
 * token, suffix-match the trimmed accumulated text (the reference
 * `_find_tag_logprobs` loop, exact tag first, then the fused `>`-less form).
 */
function referenceFindTagLogprobs(
  reply: VerifierReply,
  tag: string,
): PositionLogprobs | undefined {
  const { tokens, positionLogprobs } = reply;
  if (!tokens || !positionLogprobs || tokens.length === 0) return undefined;
  for (const suffix of [tag, tag.slice(0, -1)]) {
    let found: PositionLogprobs | undefined;
    let textSoFar = "";
    for (let i = 0; i < tokens.length; i++) {
      textSoFar += tokens[i];
      if (pythonRstrip(tokens[i]!).length === 0) continue;
      if (pythonRstrip(textSoFar).endsWith(suffix)) {
        if (i + 1 < positionLogprobs.length) {
          found = positionLogprobs[i + 1];
        }
      }
    }
    if (found !== undefined) return found;
  }
  return undefined;
}

function pythonRstrip(value: string): string {
  const whitespace =
    /[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/;
  let end = value.length;
  while (end > 0 && whitespace.test(value[end - 1]!)) end -= 1;
  return value.slice(0, end);
}

function randomTokenStream(seed: number): { tokens: string[]; positions: PositionLogprobs[] } {
  const rng = mulberry32(seed);
  const tokenPool = [
    "analysis", " ", "\n", "<score_A>", "<score_A", "A", ">A", ">T", "B",
    "</score_A>", "<score_B>", "<score_B", "C", "</score_B>", " ", "  ", "\n\n",
    'Format: "<score_A>', "LETTER_A_TO_T", '">\n<score_A>',
  ];
  const letters = "ABCDEFGHIJKLMNOPQRST";
  const n = Math.floor(rng() * 14);
  const tokens: string[] = [];
  const positions: PositionLogprobs[] = [];
  for (let i = 0; i < n; i++) {
    tokens.push(tokenPool[Math.floor(rng() * tokenPool.length)]!);
    // 20% chance the position has a real letter distribution; else generic.
    if (rng() < 0.2) {
      const chosen = letters[Math.floor(rng() * letters.length)]!;
      positions.push([
        [chosen, -Math.log(1 + rng() * 10)],
        [letters[(letters.indexOf(chosen) + 1) % letters.length]!, -Math.log(1 + rng() * 10)],
      ]);
    } else {
      positions.push([[tokenPool[Math.floor(rng() * tokenPool.length)]!, -Math.log(1 + rng())]]);
    }
  }
  return { tokens, positions };
}

describe("linear tag scanner matches the reference semantics", () => {
  test("findTagLogprobs agrees with the naive reference over random streams", () => {
    for (let seed = 0; seed < 400; seed++) {
      const { tokens, positions } = randomTokenStream(seed);
      const reply: VerifierReply = { text: tokens.join(""), tokens, positionLogprobs: positions };
      for (const tag of ["<score_A>", "<score_B>"]) {
        const reference = referenceFindTagLogprobs(reply, tag);
        const linear = findTagLogprobs(reply, tag);
        expect(linear, `seed=${seed} tag=${tag}`).toEqual(reference);
      }
    }
  });

  test("extractScorePair agrees with per-tag extractScore + hasExtractableScore", () => {
    for (let seed = 0; seed < 300; seed++) {
      const { tokens, positions } = randomTokenStream(seed);
      const reply: VerifierReply = { text: tokens.join(""), tokens, positionLogprobs: positions };
      const pair = extractScorePair(reply);
      expect(pair.scoreA, `seed=${seed}`).toBe(extractScore(reply, "<score_A>"));
      expect(pair.scoreB, `seed=${seed}`).toBe(extractScore(reply, "<score_B>"));
      expect(pair.extractableA, `seed=${seed}`).toBe(hasExtractableScore(reply, "<score_A>"));
      expect(pair.extractableB, `seed=${seed}`).toBe(hasExtractableScore(reply, "<score_B>"));
    }
  });

  test("last-match wins with trailing whitespace and mid-quoted tags", () => {
    // Exact-tag trailing match wins over an earlier exact match; whitespace
    // after the closing tag must not break the match (trimEnd semantics).
    const reply: VerifierReply = {
      text: '<score_A> A </score_A>\n<score_A> B </score_A>   ',
      tokens: ["<score_A>", " ", "A", " </score_A>\n<score_A>", " ", "B", " </score_A>", "   "],
      positionLogprobs: [
        [["<score_A>", 0]],
        [[" ", 0]],
        [["A", Math.log(0.5)], ["B", Math.log(0.5)]],
        [[" </score_A>\n<score_A>", 0]],
        [[" ", 0]],
        [["B", Math.log(1.0)], ["A", -50]],
        [[" </score_A>", 0]],
        [["   ", 0]],
      ],
    };
    const reference = referenceFindTagLogprobs(reply, "<score_A>");
    const linear = findTagLogprobs(reply, "<score_A>");
    expect(linear).toEqual(reference);
    // B has all mass -> (19-1)/19
    expect(extractScore(reply, "<score_A>")).toBeCloseTo((19 - 1) / 19, 9);
  });

  test("fused '>A' token and empty position arrays", () => {
    const reply: VerifierReply = {
      text: "<score_A> A </score_A>",
      tokens: ["<score_A>", ">A", " </score_A>"],
      positionLogprobs: [
        [["<score_A>", 0]],
        [[" >A", Math.log(0.8)], [" >T", Math.log(0.2)]],
        [[" </score_A>", 0]],
      ],
    };
    expect(findTagLogprobs(reply, "<score_A>")).toEqual(referenceFindTagLogprobs(reply, "<score_A>"));
    expect(extractScore(reply, "<score_A>")).toBeCloseTo((0.8 * 20 + 0.2 * 1 - 1) / 19, 9);
    // No tags at all: every position is a non-letter generic token.
    const empty: VerifierReply = {
      text: "no tags here",
      tokens: ["no", "tags", "here"],
      positionLogprobs: [[["no", 0]], [["tags", 0]], [["here", 0]]],
    };
    expect(findTagLogprobs(empty, "<score_A>")).toBeUndefined();
    expect(hasExtractableScore(empty, "<score_A>")).toBe(false);
    expect(extractScore(empty, "<score_A>")).toBe(0.5);
  });

  test("uses Python rstrip whitespace semantics at tag boundaries", () => {
    const nel: VerifierReply = {
      text: "<score_A>\u0085A",
      tokens: ["<score_A>\u0085", "A"],
      positionLogprobs: [
        [["<score_A>\u0085", 0]],
        [["A", 0]],
      ],
    };
    expect(findTagLogprobs(nel, "<score_A>")).toEqual([["A", 0]]);
    expect(findTagLogprobs(nel, "<score_A>")).toEqual(
      referenceFindTagLogprobs(nel, "<score_A>"),
    );

    const bom: VerifierReply = {
      text: "<score_A>\ufeffA",
      tokens: ["<score_A>\ufeff", "A"],
      positionLogprobs: [
        [["<score_A>\ufeff", 0]],
        [["A", 0]],
      ],
    };
    expect(findTagLogprobs(bom, "<score_A>")).toBeUndefined();
    expect(findTagLogprobs(bom, "<score_A>")).toEqual(
      referenceFindTagLogprobs(bom, "<score_A>"),
    );
  });

  test("whitespace tokens do not shadow the first post-tag distribution", () => {
    const firstDistribution: PositionLogprobs = [
      [" A", Math.log(0.8)],
      [" T", Math.log(0.2)],
    ];
    const reply: VerifierReply = {
      text: "<score_A> A",
      tokens: ["<score_A>", " ", "A"],
      positionLogprobs: [
        [["<score_A>", 0]],
        firstDistribution,
        [["A", 0]],
      ],
    };
    expect(findTagLogprobs(reply, "<score_A>")).toEqual(firstDistribution);
    expect(findTagLogprobs(reply, "<score_A>")).toEqual(
      referenceFindTagLogprobs(reply, "<score_A>"),
    );
    expect(extractScore(reply, "<score_A>")).toBeCloseTo((0.8 * 20 + 0.2 - 1) / 19, 10);
  });
});

describe("extractScore", () => {
  test("scale has 20 letters A-T mapping to 20..1", () => {
    expect(GRANULARITY).toBe(20);
    expect(SCALE.scoreFormat).toBe("LETTER_A_TO_T");
  });

  test("expectation over logprob distribution at <score_A>", () => {
    // Distribution: 90% mass on 'A' (raw 20), 10% on 'T' (raw 1).
    const reply: VerifierReply = {
      text: "analysis\n<score_A> A </score_A>\n<score_B> T </score_B>",
      tokens: ["analysis", "\n<score_A>", "A", " </score_A>\n<score_B>", "T", " </score_B>"],
      positionLogprobs: [
        [["analysis", 0]],
        [["\n<score_A>", 0]],
        [
          ["A", Math.log(0.9)],
          ["T", Math.log(0.1)],
        ],
        [[" </score_A>\n<score_B>", 0]],
        [
          ["T", Math.log(0.95)],
          ["A", Math.log(0.05)],
        ],
        [[" </score_B>", 0]],
      ],
    };
    const ra = extractScore(reply, "<score_A>");
    const rb = extractScore(reply, "<score_B>");
    const pair = extractScorePair(reply);
    const expectedA = (0.9 * 20 + 0.1 * 1 - 1) / (20 - 1); // (18.1-1)/19
    const expectedB = (0.95 * 1 + 0.05 * 20 - 1) / (19); // (1.95-1)/19
    expect(ra).toBeCloseTo(expectedA, 10);
    expect(rb).toBeCloseTo(expectedB, 10);
    expect(ra).toBeGreaterThan(rb);
    expect(pair.sourceA).toBe("logprobs");
    expect(pair.sourceB).toBe("logprobs");
    expect(pair.supportA).toBe(2);
    expect(pair.supportB).toBe(2);
    expect(pair.probabilityMassA).toBeCloseTo(1, 10);
    expect(pair.probabilityMassB).toBeCloseTo(1, 10);
  });

  test("fused '>A' token (DeepSeek tokenizer)", () => {
    const reply: VerifierReply = {
      text: "<score_A> A </score_A>",
      tokens: ["<score_A>", ">A", " </score_A>"],
      positionLogprobs: [
        [["<score_A>", 0]],
        [
          [">A", Math.log(0.8)],
          [">T", Math.log(0.2)],
        ],
        [[" </score_A>", 0]],
      ],
    };
    const ra = extractScore(reply, "<score_A>");
    const expected = (0.8 * 20 + 0.2 * 1 - 1) / 19;
    expect(ra).toBeCloseTo(expected, 10);
  });

  test("literal-text fallback when no logprobs", () => {
    const reply: VerifierReply = { text: "analysis\n<score_A> E </score_A>\n<score_B> Q </score_B>" };
    // E = raw 16, Q = raw 4
    expect(extractScore(reply, "<score_A>")).toBeCloseTo((16 - 1) / 19, 10);
    expect(extractScore(reply, "<score_B>")).toBeCloseTo((4 - 1) / 19, 10);
    const pair = extractScorePair(reply);
    expect(pair.sourceA).toBe("text_fallback");
    expect(pair.sourceB).toBe("text_fallback");
    expect(pair.supportA).toBe(0);
    expect(pair.probabilityMassA).toBe(0);
  });

  test("last match wins when the model quotes the format mid-analysis", () => {
    const reply: VerifierReply = {
      text: 'Format: "<score_A> LETTER_A_TO_T </score_A>"\n<score_A> B </score_A>',
      tokens: [
        'Format: "<score_A>',
        " LETTER_A_TO_T ",
        '</score_A>"\n<score_A>',
        "B",
        " </score_A>",
      ],
      positionLogprobs: [
        [['Format: "<score_A>', 0]],
        [[" LETTER_A_TO_T ", 0]],
        [['</score_A>"\n<score_A>', 0]],
        [
          ["B", Math.log(1.0)],
          ["T", -50],
        ],
        [[" </score_A>", 0]],
      ],
    };
    // B = raw 19 → (19-1)/19
    expect(extractScore(reply, "<score_A>")).toBeCloseTo((19 - 1) / 19, 6);
  });

  test("missing tag returns neutral 0.5", () => {
    const reply: VerifierReply = { text: "no tags here", tokens: ["no tags here"] };
    expect(extractScore(reply, "<score_A>")).toBe(0.5);
  });
});

describe("PPT", () => {
  test("ring cycle covers every candidate once in each slot", () => {
    const rng = mulberry32(0);
    const ring = ringCycle(5, rng);
    expect(ring.length).toBe(5);
    const aCount = new Map<number, number>();
    const bCount = new Map<number, number>();
    for (const [a, b] of ring) {
      aCount.set(a, (aCount.get(a) ?? 0) + 1);
      bCount.set(b, (bCount.get(b) ?? 0) + 1);
    }
    for (let i = 0; i < 5; i++) {
      expect(aCount.get(i)).toBe(1);
      expect(bCount.get(i)).toBe(1);
    }
  });

  test("deterministic given seed", () => {
    const r1 = ringCycle(7, mulberry32(42));
    const r2 = ringCycle(7, mulberry32(42));
    expect(r1).toEqual(r2);
  });

  test("bradleyTerry saturates on large gaps", () => {
    expect(bradleyTerry(0.9, 0.1)).toBeCloseTo(1 / (1 + Math.exp(-0.8)), 10);
    // sigmoid(1) ≈ 0.731 — a full-scale reward gap is a strong but not
    // absolute preference.
    expect(bradleyTerry(1, 0)).toBeCloseTo(1 / (1 + Math.exp(-1)), 10);
    expect(bradleyTerry(0, 1)).toBeCloseTo(1 / (1 + Math.exp(1)), 10);
    expect(bradleyTerry(0.99, 0.01)).toBeCloseTo(1 / (1 + Math.exp(-0.98)), 10);
  });

  test("pivotRoundPairs reaches the k(N-k) + C(k,2) upper bound without a ring", () => {
    const pairs = pivotRoundPairs(5, [2, 3]);
    expect(pairs.length).toBe(2 * 3 + 1);
    // non-pivots take slot A, pivots slot B
    for (const [a, b] of pairs.slice(0, 6)) {
      expect([0, 1, 4]).toContain(a);
      expect([2, 3]).toContain(b);
    }
  });

  test("pivotRoundPairs implements Algorithm 1 E_piv minus E_ring", () => {
    const ring: Array<[number, number]> = [[0, 2], [2, 1], [1, 0]];
    const pairs = pivotRoundPairs(3, [0], ring);
    expect(pairs).toEqual([[2, 0]]);

    const ringSet = new Set(ring.map(([a, b]) => `${a},${b}`));
    expect(pairs.every(([a, b]) => !ringSet.has(`${a},${b}`))).toBe(true);
  });

  test("Algorithm 1 pivot edge sets stay unique and within the O(Nk) bound", () => {
    for (let n = 2; n <= 8; n += 1) {
      for (let seed = 0; seed < 20; seed += 1) {
        const ring = ringCycle(n, mulberry32(seed));
        const ringSet = new Set(ring.map(([a, b]) => `${a},${b}`));
        for (let k = 1; k <= n; k += 1) {
          const pivots = Array.from({ length: k }, (_, index) => index);
          const unfiltered = pivotRoundPairs(n, pivots);
          const filtered = pivotRoundPairs(n, pivots, ring);
          const filteredKeys = filtered.map(([a, b]) => `${a},${b}`);
          expect(new Set(filteredKeys).size).toBe(filtered.length);
          expect(filteredKeys.every((key) => !ringSet.has(key))).toBe(true);
          expect(filtered).toEqual(
            unfiltered.filter(([a, b]) => !ringSet.has(`${a},${b}`)),
          );
          expect(ring.length + filtered.length).toBeLessThanOrEqual(
            n + k * (n - k) + (k * (k - 1)) / 2,
          );
        }
      }
    }
  });

  test("pins the paper/reference Bo3 edge-count divergence", () => {
    const ring: Array<[number, number]> = [[0, 2], [2, 1], [1, 0]];
    const w = [0, 0, 0];
    const c = [0, 0, 0];
    accumulate(ring, () => [0.5, 0.5], w, c);
    const pivots = selectPivots(w, c, 1);
    expect(pivots).toEqual([0]);
    expect(ring.length + pivotRoundPairs(3, pivots, ring).length).toBe(4);
    expect(ring.length + pivotRoundPairs(3, pivots).length).toBe(5);
  });

  test("selectBest picks the ground-truth winner with perfect scoring", () => {
    // Candidate 2 is the best (highest intrinsic quality).
    const intrinsic = [0.2, 0.4, 0.9, 0.6, 0.1];
    const score: (a: number, b: number) => [number, number] = (a, b) => {
      const diff = (intrinsic[a] - intrinsic[b]) / 2 + 0.5;
      return [diff, 1 - diff];
    };
    const rng = mulberry32(0);
    const ring = ringCycle(5, rng);
    const { bestIndex, nComparisons } = selectBest(5, ring, 2, score);
    expect(bestIndex).toBe(2);
    // Algorithm 1 removes directed edges already observed in the ring, while
    // the paper's N + k(N-k) + C(k,2) expression remains an upper bound.
    expect(nComparisons).toBeLessThanOrEqual(5 + 2 * 3 + 1);
  });

  test("selectPivots picks ring leaders", () => {
    const w = [0.2, 0.8, 0.5];
    const c = [2, 2, 2];
    expect(selectPivots(w, c, 2)).toEqual([1, 2]);
  });
});

describe("directedReward", () => {
  test("averages fine-grained rewards over criteria and repeats (1/CK)", () => {
    const scores: ScoreCache = {
      [cacheKey("c1", "task", 0, 1, 0)]: { score_A: 1, score_B: 0 },
      [cacheKey("c1", "task", 0, 1, 1)]: { score_A: 0.5, score_B: 0.5 },
      [cacheKey("c2", "task", 0, 1, 0)]: { score_A: 0, score_B: 1 },
      [cacheKey("c2", "task", 0, 1, 1)]: { score_A: 1, score_B: 0 },
    };
    const [ra, rb] = directedReward(scores, "task", 0, 1, ["c1", "c2"], 2);
    expect(ra).toBeCloseTo((1 + 0.5 + 0 + 1) / 4, 10);
    expect(rb).toBeCloseTo((0 + 0.5 + 1 + 0) / 4, 10);
  });

  test("missing entries default to the neutral 0.5 tie", () => {
    const scores: ScoreCache = {
      [cacheKey("c1", "task", 0, 1, 0)]: { score_A: 1, score_B: 0 },
    };
    // rep 0 present as (1, 0); rep 1 missing and counted as (0.5, 0.5).
    const [ra, rb] = directedReward(scores, "task", 0, 1, ["c1"], 2);
    expect(ra).toBeCloseTo((1 + 0.5) / 2, 10);
    expect(rb).toBeCloseTo((0 + 0.5) / 2, 10);
    expect(directedReward({}, "task", 0, 1, ["c1"], 3)).toEqual([0.5, 0.5]);
  });

  test("self-comparison is neutral and reads no cache entries", () => {
    expect(directedReward({}, "task", 2, 2, ["c1"], 1)).toEqual([0.5, 0.5]);
  });

  test("resolves context-fingerprinted keys the way production scoring writes them", () => {
    // The writer (scoreDirectedPairs) and reader (directedReward) must agree
    // on the key for the same (criterion, task, a, b, rep, context).
    const context = {
      criterionId: "c1",
      criterionName: "C1",
      criterionDescription: "desc",
      problem: "p",
      traceA: "ta",
      traceB: "tb",
      imagesFingerprint: "",
      trajectoryImagesAFingerprint: "",
      trajectoryImagesBFingerprint: "",
      provider: "prov",
      api: "openai-completions",
      model: "m",
      effort: "high",
      maxTokens: 32768,
      baseUrl: "https://example.test/v1",
      requestIdentity: "id",
      groundTruthNote: "note",
      promptVersion: "v",
    };
    const scores: ScoreCache = {
      [cacheKey("c1", "task", 0, 1, 0, context)]: { score_A: 0.8, score_B: 0.2 },
    };
    const [ra, rb] = directedReward(scores, "task", 0, 1, ["c1"], 1, context);
    expect(ra).toBeCloseTo(0.8, 10);
    expect(rb).toBeCloseTo(0.2, 10);
    // A different context must NOT see the entry (no stale-score reuse).
    const drifted = { ...context, model: "other-model" };
    expect(directedReward(scores, "task", 0, 1, ["c1"], 1, drifted)).toEqual([0.5, 0.5]);
  });
});

describe("classify", () => {
  const { classify } = require("../src/loader.ts");
  test("all-pass and swing tasks are split; all-fail needs no verification", () => {
    const tasks = {
      all_pass: [
        { reward: 1 as const },
        { reward: 1 as const },
      ],
      swing: [
        { reward: 1 as const },
        { reward: 0 as const },
      ],
      all_fail: [
        { reward: 0 as const },
        { reward: 0 as const },
      ],
    };
    expect(classify(tasks)).toEqual({
      allPass: ["all_pass"],
      swing: ["swing"],
    });
  });
});

describe("theory-gate cache identity and aggregation invariants", () => {
  function baseContext(): CacheContext {
    return {
      criterionId: "c1",
      criterionName: "C1",
      criterionDescription: "desc",
      problem: "p",
      traceA: "ta",
      traceB: "tb",
      imagesFingerprint: "",
      trajectoryImagesAFingerprint: "",
      trajectoryImagesBFingerprint: "",
      provider: "prov",
      api: "openai-completions",
      model: "m",
      effort: "high",
      maxTokens: 32768,
      baseUrl: "https://example.test/v1",
      requestIdentity: "id",
      groundTruthNote: "note",
      promptVersion: "pairwise-granularity20-v5",
    };
  }

  test("directed (a,b) and (b,a) comparisons are distinct cache identities", () => {
    const context = baseContext();
    const ab = cacheKey("c1", "task", 0, 1, 0, context);
    const ba = cacheKey("c1", "task", 1, 0, 0, context);
    expect(ab).not.toBe(ba);
    // Slot swap: the same directed pair with A/B traces exchanged is a
    // different prompt and therefore a different key.
    expect(ab).not.toBe(
      cacheKey("c1", "task", 0, 1, 0, { ...context, traceA: "tb", traceB: "ta" }),
    );
  });

  test("repetitions keep distinct cache keys so K observations never collapse", () => {
    const context = baseContext();
    const rep0 = cacheKey("c1", "task", 0, 1, 0, context);
    const rep1 = cacheKey("c1", "task", 0, 1, 1, context);
    expect(rep0).not.toBe(rep1);
  });

  test("cache keys embed CACHE_VERSION so a version bump invalidates old scores", () => {
    const key = cacheKey("c1", "task", 0, 1, 0, baseContext());
    expect(key.startsWith(`v${CACHE_VERSION}|`)).toBe(true);
  });

  test("criteria aggregation is order-invariant and equal-weight", () => {
    const scores: ScoreCache = {
      [cacheKey("c1", "task", 0, 1, 0)]: { score_A: 1, score_B: 0 },
      [cacheKey("c1", "task", 0, 1, 1)]: { score_A: 0.5, score_B: 0.5 },
      [cacheKey("c2", "task", 0, 1, 0)]: { score_A: 0, score_B: 1 },
      [cacheKey("c2", "task", 0, 1, 1)]: { score_A: 1, score_B: 0 },
    };
    const [ra, rb] = directedReward(scores, "task", 0, 1, ["c1", "c2"], 2);
    const [raSwapped, rbSwapped] = directedReward(scores, "task", 0, 1, ["c2", "c1"], 2);
    expect([ra, rb]).toEqual([raSwapped, rbSwapped]);
    // Each criterion contributes exactly 1/(C*K) weight.
    expect(ra).toBeCloseTo((1 + 0.5 + 0 + 1) / 4, 10);
    expect(rb).toBeCloseTo((0 + 0.5 + 1 + 0) / 4, 10);
  });

  test("K repetitions alternate A/B slots and scores stay bound to candidate identity", async () => {
    const captured: string[] = [];
    const client = {
      provider: "prov",
      api: "openai-completions",
      model: "m",
      effort: "off",
      maxTokens: 4096,
      baseUrl: "https://example.test/v1",
      requestIdentity: "id",
      supportsImages: true,
      scoreReply: async (prompt: string) => {
        captured.push(prompt);
        const slotAIsTrailA = prompt.includes("**Trajectory A:**\ntrace a");
        const scoreA = slotAIsTrailA ? "A" : "T";
        const scoreB = slotAIsTrailA ? "T" : "A";
        return {
          text: `<score_A> ${scoreA} </score_A>\n<score_B> ${scoreB} </score_B>`,
          tokens: ["<score_A>", ` ${scoreA}`, " </score_A>\n<score_B>", ` ${scoreB}`, " </score_B>"],
          positionLogprobs: [
            [["<score_A>", 0]],
            [[scoreA, 0]],
            [[" </score_A>\n<score_B>", 0]],
            [[scoreB, 0]],
            [[" </score_B>", 0]],
          ],
        };
      },
    } as unknown as VerifierClient;
    const tasks = {
      task: [
        { trialName: "a", reward: 0 as const, problem: "p", trace: "trace a" },
        { trialName: "b", reward: 1 as const, problem: "p", trace: "trace b" },
      ],
    };
    const scores = await scoreDirectedPairs(
      client,
      tasks,
      { task: [[0, 1]] },
      [{ id: "c1", name: "C1", description: "desc" }],
      "note",
      2,
      1,
      undefined,
      { progress: false },
    );
    // rep 0 puts (a,b) in the prompt; rep 1 swaps the slots.
    expect(captured).toHaveLength(2);
    expect(captured[0]!).toContain("**Trajectory A:**\ntrace a");
    expect(captured[0]!).toContain("**Trajectory B:**\ntrace b");
    expect(captured[1]!).toContain("**Trajectory A:**\ntrace b");
    expect(captured[1]!).toContain("**Trajectory B:**\ntrace a");
    // After mapping back to candidate identity, score_A is always candidate a.
    for (const entry of Object.values(scores)) {
      expect(entry.score_A).toBe(1);
      expect(entry.score_B).toBe(0);
    }
  });
});
