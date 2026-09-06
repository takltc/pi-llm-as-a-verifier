import { expect, test } from "bun:test";
import { summarizeScoreDistribution, type CachedEntry } from "../src/cache.ts";

test("score distribution preserves mixed evidence and empty summaries", () => {
  const empty = {
    logprobScores: 0,
    minSupport: 0,
    meanSupport: 0,
    minProbabilityMass: 0,
    meanProbabilityMass: 0,
  };
  expect(summarizeScoreDistribution([])).toEqual(empty);
  expect(summarizeScoreDistribution([undefined, { score_A: 0.5, score_B: 0.5 }])).toEqual(empty);
  expect(summarizeScoreDistribution([
    {
      score_A: 0.7, score_B: 0.3,
      source_A: "logprobs", source_B: "logprobs",
      support_A: 2, support_B: 4,
      probability_mass_A: 0.25, probability_mass_B: 0.75,
    },
    {
      score_A: 0.5, score_B: 0.5,
      source_A: "text_fallback", source_B: "logprobs",
      support_A: 1, support_B: 0,
      probability_mass_A: 0.125, probability_mass_B: 0.125,
    },
    undefined,
  ])).toEqual({
    logprobScores: 2,
    minSupport: 2,
    meanSupport: 3,
    minProbabilityMass: 0.25,
    meanProbabilityMass: 0.5,
  });
});

test("score distribution streams large single-use iterables without argument overflow", () => {
  function* entries(): Generator<CachedEntry> {
    for (let i = 0; i < 500_000; i++) {
      yield {
        score_A: 0.7, score_B: 0.3,
        source_A: "logprobs", source_B: "logprobs",
        support_A: 2, support_B: 4,
        probability_mass_A: 0.25, probability_mass_B: 0.75,
      };
    }
  }
  expect(summarizeScoreDistribution(entries())).toEqual({
    logprobScores: 1_000_000,
    minSupport: 2,
    meanSupport: 3,
    minProbabilityMass: 0.25,
    meanProbabilityMass: 0.5,
  });
});
