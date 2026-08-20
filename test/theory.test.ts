/**
 * Theory-anchor pins against the LLM-as-a-Verifier paper
 * (arXiv:2607.05391) and its reference implementation in `_ref/`.
 *
 * These tests exist so a future edit that drifts from the paper's reward
 * formula, pairwise prompt, or PPT cost budget fails loudly instead of
 * silently changing what the plugin verifies.
 */

import { describe, expect, test } from "bun:test";
import { AUTO_SELECTION_DEFAULTS } from "../src/auto.ts";
import type { VerifierClient } from "../src/client.ts";
import { buildPrompt, CODING_AGENT_CRITERIA } from "../src/prompt.ts";
import { GRANULARITY, SCALE } from "../src/scale.ts";
import { select } from "../src/select.ts";
import { runBenchmark, validateVerifyOptions } from "../src/run.ts";

/**
 * Verbatim scale text from `_ref/llm_verifier/fine_grained_reward.py`
 * (SCALE["scale_description"], GRANULARITY=20). The plugin's `SCALE` must
 * live on this exact string so every score token distribution stays
 * comparable to the reference.
 */
const REFERENCE_SCALE_DESCRIPTION =
  "Rate how likely the agent correctly solved the task on a " +
  "20-point scale using letters A through T:\n" +
  "  A = clearly and completely succeeded with verified output (best)\n" +
  "  B-D = succeeded with only minor issues\n" +
  "  E-G = above average, mostly correct with some issues\n" +
  "  H-J = uncertain, leans toward success\n" +
  "  K-M = uncertain, leans toward failure\n" +
  "  N-P = below average, significant issues remain\n" +
  "  Q-S = failed with some partial progress\n" +
  "  T = clearly and completely failed (worst)";

describe("paper theory anchors", () => {
  test("granularity-20 scale description matches the reference verbatim", () => {
    expect(GRANULARITY).toBe(20);
    expect(SCALE.scaleDescription).toBe(REFERENCE_SCALE_DESCRIPTION);
    expect(SCALE.scoreFormat).toBe("LETTER_A_TO_T");
  });

  test("pairwise prompt keeps the shared prefix first and criterion at the tail", () => {
    const [c1, c2] = CODING_AGENT_CRITERIA;
    const p1 = buildPrompt("task", "A TRACE", "B TRACE", c1, "NOTE");
    const p2 = buildPrompt("task", "A TRACE", "B TRACE", c2, "NOTE");
    // Everything before the criterion block is identical, so a prefix-caching
    // backend serves the trace-heavy body from cache across criteria.
    const prefixLen = p1.indexOf("**Evaluation Guideline —");
    expect(p1.lastIndexOf(c1.name)).toBeGreaterThan(prefixLen);
    expect(p2.lastIndexOf(c2.name)).toBeGreaterThan(prefixLen);
    expect(p2.slice(0, prefixLen)).toBe(p1.slice(0, prefixLen));
    expect(p1).toContain("**Task:**\ntask");
    expect(p1).toContain("**Trajectory A:**\nA TRACE");
    expect(p1).toContain("**Trajectory B:**\nB TRACE");
    expect(p1).toContain("<score_A> LETTER_A_TO_T </score_A>");
    expect(p1.endsWith("Begin your analysis now.")).toBe(true);
    const withImages = buildPrompt("task", "A TRACE", "B TRACE", c1, "NOTE", 2);
    expect(withImages).toContain(
      "**Attached images:** 2 image(s) are attached to this message, in order; " +
      "they are part of the task context.\n\n**Trajectory A:**",
    );
    const withTrajectoryImages = buildPrompt(
      "task",
      "A TRACE",
      "B TRACE",
      c1,
      "NOTE",
      { shared: 1, trajectoryA: 2, trajectoryB: 1 },
    );
    expect(withTrajectoryImages).toContain(
      "**Attached images:** 4 image(s) are attached to this message. " +
      "Order: 1 shared task-context image(s), then 2 Trajectory A image(s), " +
      "then 1 Trajectory B image(s).",
    );
  });

  test("buildPrompt is verbatim the reference build_prompt for a fixed input", () => {
    const expected =
      "You are an expert evaluator of AI coding agents. You will see a task " +
      "description and two agent trajectories, then evaluate them on ONE " +
      "specific criterion, stated at the end.\n\n" +
      "NOTE\n\n" +
      "**Task:**\ntask\n\n" +
      "**Trajectory A:**\nA TRACE\n\n" +
      "**Trajectory B:**\nB TRACE\n\n" +
      "**Rating Scale:**\n" +
      SCALE.scaleDescription +
      "\n\n" +
      "**Evaluation Guideline — Task Correctness:**\n" +
      "Check whether the proposed response directly advances the user's " +
      "current request, respects the stated constraints, and chooses " +
      "technically sound actions or conclusions for the available context." +
      "\n\n" +
      "Score each trajectory ONLY on this specific criterion " +
      '("Task Correctness"). Ignore other aspects of the trajectory that are ' +
      "not relevant to it.\n\n" +
      "Reason it through first, then END your reply with exactly these two " +
      "lines and nothing after them. Replace each placeholder with a single " +
      "letter A-T, keeping the spaces around the letter exactly as shown:\n" +
      "<score_A> LETTER_A_TO_T </score_A>\n" +
      "<score_B> LETTER_A_TO_T </score_B>\n\n" +
      "Begin your analysis now.";
    const actual = buildPrompt("task", "A TRACE", "B TRACE", CODING_AGENT_CRITERIA[0], "NOTE");
    expect(actual).toBe(expected);
  });

  test("wrapper best-of-3 config follows the formal Algorithm 1 edge set", async () => {
    // Paper self-verification defaults (scripts/run_bo3.py): pivots=1, K=2.
    expect(AUTO_SELECTION_DEFAULTS).toEqual({ pivots: 1, nEvaluations: 2, seed: 0, maxWorkers: 8 });
    let calls = 0;
    const client = {
      scoreReply: async () => {
        calls += 1;
        return {
          text: "<score_A> A </score_A>\n<score_B> T </score_B>",
          tokens: ["<score_A>", " A", " </score_A>\n<score_B>", " T", " </score_B>"],
          positionLogprobs: [
            [["<score_A>", 0]],
            [["A", 0]],
            [[" </score_A>\n<score_B>", 0]],
            [["T", 0]],
            [[" </score_B>", 0]],
          ],
        };
      },
    } as unknown as VerifierClient;
    const result = await select("problem", [
      { name: "a", trace: "trace a" },
      { name: "b", trace: "trace b" },
      { name: "c", trace: "trace c" },
    ], {
      ...AUTO_SELECTION_DEFAULTS,
      criteria: CODING_AGENT_CRITERIA,
      client,
      taskName: "current_request",
      progress: false,
    });
    // For N=3, k=1, the pivot has one incoming ring edge. Algorithm 1 removes
    // that overlap: 3 ring comparisons + 1 new pivot comparison = 4.
    expect(result.nComparisons).toBe(4);
    // One verifier call per unique directed comparison, criterion and repeat.
    expect(calls).toBe(4 * 3 * 2);
    expect(result.paperEquivalent).toBe(true);
    expect(result.scoreSources).toEqual({
      logprobs: 4 * 3 * 2 * 2,
      textFallback: 0,
      neutralTie: 0,
      unknown: 0,
    });
    expect(result.scoreDistribution).toEqual({
      logprobScores: 4 * 3 * 2 * 2,
      minSupport: 1,
      meanSupport: 1,
      minProbabilityMass: 1,
      meanProbabilityMass: 1,
    });
    // Every candidate is scored three times (in each PPT comparison round),
    // so exactly one candidate wins and the rest trail on mean preference.
    expect(result.scores).toHaveLength(3);
  });

  test("benchmark reports provenance for the exact formal comparison set", async () => {
    let calls = 0;
    const client = {
      scoreReply: async () => {
        calls += 1;
        return {
          text: "<score_A> A </score_A>\n<score_B> T </score_B>",
          tokens: ["<score_A>", " A", " </score_A>\n<score_B>", " T", " </score_B>"],
          positionLogprobs: [
            [["<score_A>", 0]],
            [["A", 0]],
            [[" </score_A>\n<score_B>", 0]],
            [["T", 0]],
            [[" </score_B>", 0]],
          ],
        };
      },
    } as unknown as VerifierClient;
    const stats = await runBenchmark({ benchmark: [
      { trialName: "a", reward: 0, problem: "problem", trace: "trace a" },
      { trialName: "b", reward: 1, problem: "problem", trace: "trace b" },
      { trialName: "c", reward: 0, problem: "problem", trace: "trace c" },
    ] }, CODING_AGENT_CRITERIA, {
      ...AUTO_SELECTION_DEFAULTS,
      client,
      progress: false,
    });

    expect(stats.totalComparisons).toBe(4);
    expect(stats.avgComparisons).toBe(4);
    expect(calls).toBe(4 * 3 * 2);
    expect(stats.paperEquivalent).toBe(true);
    expect(stats.scoreSources).toEqual({
      logprobs: 4 * 3 * 2 * 2,
      textFallback: 0,
      neutralTie: 0,
      unknown: 0,
    });
    expect(stats.scoreDistribution.logprobScores).toBe(4 * 3 * 2 * 2);
  });

  test("all candidates share an identical task and ordered visual evidence", () => {
    const base = {
      trialName: "a",
      reward: 0 as const,
      problem: "same task",
      trace: "trace a",
      images: [{ type: "image" as const, data: "YQ==", mimeType: "image/png" }],
    };
    expect(() => validateVerifyOptions({ task: [
      base,
      { ...base, trialName: "b", trace: "trace b", problem: "different task" },
    ] }, CODING_AGENT_CRITERIA, {})).toThrow("must share one identical problem");

    expect(() => validateVerifyOptions({ task: [
      base,
      {
        ...base,
        trialName: "b",
        trace: "trace b",
        images: [{ type: "image", data: "Yg==", mimeType: "image/png" }],
      },
    ] }, CODING_AGENT_CRITERIA, {})).toThrow("must share one identical ordered image set");
  });
});
