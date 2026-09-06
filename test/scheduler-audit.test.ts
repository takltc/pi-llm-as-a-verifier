import { expect, test } from "bun:test";
import { scoreDirectedPairs } from "../src/run.ts";
import type { VerifierClient } from "../src/client.ts";

test("unsupported breaker drains large batches without recursion or provider calls", async () => {
  const result = await scoreDirectedPairs(
    { scoreReply: async () => { throw new Error("unexpected provider call"); } } as unknown as VerifierClient,
    { task: [
      { trialName: "a", reward: 0, problem: "task", trace: "a" },
      { trialName: "b", reward: 1, problem: "task", trace: "b" },
    ] },
    { task: [[0, 1]] },
    [{ id: "c", name: "Correctness", description: "Correctness" }],
    "", 50_000, 1, undefined,
    { progress: false, unsupportedBreaker: { failures: 2, skip: true } },
  );
  expect(Object.keys(result)).toHaveLength(50_000);
  expect(Object.values(result).every((entry) =>
    entry.score_A === 0.5 && entry.score_B === 0.5 &&
    entry.source_A === "neutral_tie" && entry.source_B === "neutral_tie"
  )).toBe(true);
});
