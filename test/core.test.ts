import { describe, expect, test } from "bun:test";
import { extractScore, GRANULARITY, SCALE, type VerifierReply } from "../src/scale.ts";
import { bradleyTerry, pivotRoundPairs, ringCycle, selectBest, selectPivots } from "../src/ppt.ts";
import { mulberry32 } from "../src/run.ts";

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
    const expectedA = (0.9 * 20 + 0.1 * 1 - 1) / (20 - 1); // (18.1-1)/19
    const expectedB = (0.95 * 1 + 0.05 * 20 - 1) / (19); // (1.95-1)/19
    expect(ra).toBeCloseTo(expectedA, 10);
    expect(rb).toBeCloseTo(expectedB, 10);
    expect(ra).toBeGreaterThan(rb);
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

  test("pivotRoundPairs: k(N-k) + C(k,2) pairs", () => {
    const pairs = pivotRoundPairs(5, [2, 3]);
    expect(pairs.length).toBe(2 * 3 + 1);
    // non-pivots take slot A, pivots slot B
    for (const [a, b] of pairs.slice(0, 6)) {
      expect([0, 1, 4]).toContain(a);
      expect([2, 3]).toContain(b);
    }
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
    const { bestIndex } = selectBest(5, ring, 2, score);
    expect(bestIndex).toBe(2);
  });

  test("selectPivots picks ring leaders", () => {
    const w = [0.2, 0.8, 0.5];
    const c = [2, 2, 2];
    expect(selectPivots(w, c, 2)).toEqual([1, 2]);
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
