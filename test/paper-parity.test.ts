import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { cacheKey, directedReward, type ScoreCache } from "../src/cache.ts";
import type { VerifierClient } from "../src/client.ts";
import { formatTrace } from "../src/loader.ts";
import { accumulate, pivotRoundPairs, ringCycle, selectPivots } from "../src/ppt.ts";
import { buildPrompt } from "../src/prompt.ts";
import { extractScorePair, type PositionLogprobs, type VerifierReply } from "../src/scale.ts";
import { select } from "../src/select.ts";

const referencePath = fileURLToPath(new URL("../_ref/llm_verifier/fine_grained_reward.py", import.meta.url));
const python = Bun.which("python3");

test.skipIf(!python || !existsSync(referencePath))(
  "Python reference differential (requires python3 and the fixed _ref checkout)",
  () => {
    const fixtures: VerifierReply[] = [];
    for (const prefix of [["<score_A>"], ["<", "score", "_A", ">"], ["<score_A"]]) {
      for (const whitespace of ["", " ", "\u0085", "\u001c", "\u3000"]) {
        for (const probability of [0.05, 0.15, 0.35, 0.75]) {
          const tokens = [...prefix, whitespace, "A", "</score_A>", "<score_B>", "T"];
          const positionLogprobs: PositionLogprobs[] = tokens.map(() => []);
          positionLogprobs[prefix.length] = [
            ["A", Math.log(probability)], [" a", Math.log(probability / 2)],
            [">T", Math.log(0.05)], ["other", Math.log(0.1)],
          ];
          positionLogprobs[tokens.length - 1] = [["T", 0]];
          fixtures.push({ text: tokens.join(""), tokens, positionLogprobs });
        }
      }
    }
    // Execute the fixed source's actual functions and scale constants without
    // importing its optional model SDKs or making any provider requests.
    const result = spawnSync(python!, ["-c", `
import ast, json, math, pathlib, re, sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text())
nodes = [n for n in tree.body if
    (isinstance(n, ast.FunctionDef) and n.name in ("_find_tag_logprobs", "extract_score")) or
    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("GRANULARITY", "SCALE") for t in n.targets))]
namespace = {"math": math, "re": re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), sys.argv[1], "exec"), namespace)
fixtures = json.load(sys.stdin)
print(json.dumps([[namespace["extract_score"](f["text"], f["tokens"], f["positionLogprobs"], tag)
    for tag in ("<score_A>", "<score_B>")] for f in fixtures]))
`, referencePath], { input: JSON.stringify(fixtures), encoding: "utf8", timeout: 10_000 });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(0);
    const expected: number[][] = JSON.parse(result.stdout);
    expect(expected).toHaveLength(60);
    for (let i = 0; i < fixtures.length; i++) {
      const actual = extractScorePair(fixtures[i]);
      expect(actual.scoreA).toBeCloseTo(expected[i][0], 14);
      expect(actual.scoreB).toBeCloseTo(expected[i][1], 14);
    }
  },
);

// Eq. (3.1), and _ref extract_score: aliases share a raw value and retain
// their maximum probability; unrelated vocabulary does not enter the mean.
test("paper reward normalizes returned mass across score-tag token boundaries", () => {
  const distribution: PositionLogprobs = [
    ["A", Math.log(0.15)], [" a", Math.log(0.1)],
    [">T", Math.log(0.05)], ["unrelated", Math.log(0.7)],
  ];
  for (const prefix of [["<score_A>"], ["<", "score", "_A", ">"], ["<score_A"]]) {
    for (const whitespace of ["", " ", "\u0085", "\u001c", "\u3000"]) {
      const tokens = [...prefix, whitespace, "A", "</score_A>", "<score_B>", "T"];
      const positionLogprobs: PositionLogprobs[] = tokens.map(() => []);
      positionLogprobs[prefix.length] = distribution;
      positionLogprobs[tokens.length - 1] = [["T", 0]];
      const result = extractScorePair({ text: tokens.join(""), tokens, positionLogprobs });
      expect(result.sourceA).toBe("logprobs");
      expect(result.scoreA).toBeCloseTo(0.75, 14);
      expect(result.supportA).toBe(2);
      expect(result.probabilityMassA).toBeCloseTo(0.2, 14);
      expect(result.scoreB).toBe(0);
    }
  }
});

test("paper ring balances prompt slots and pivot rounds remove only directed overlaps", () => {
  for (let n = 2; n <= 12; n++) {
    let state = n;
    const ring = ringCycle(n, () => ((state = (1664525 * state + 1013904223) >>> 0) / 2 ** 32));
    expect(new Set(ring.map(([a]) => a)).size).toBe(n);
    expect(new Set(ring.map(([, b]) => b)).size).toBe(n);
    for (let i = 0; i < n; i++) expect(ring[i][1]).toBe(ring[(i + 1) % n][0]);
    const w = Array(n).fill(0);
    const c = Array(n).fill(0);
    accumulate(ring, () => [1, 0], w, c);
    expect(c).toEqual(Array(n).fill(2));
    for (const wins of w) expect(wins).toBeCloseTo(1, 14);
  }
  expect(selectPivots([1.2, 0.7, 1.4], [3, 1, 2], 2)).toEqual([1, 2]);
  const ring: Array<[number, number]> = [[0, 1], [1, 2], [2, 3], [3, 0]];
  // Algorithm 1 E_piv \\ E_ring; reverse edge [2,1] remains a distinct observation.
  expect(pivotRoundPairs(4, [1, 3], ring)).toEqual([[0, 3], [2, 1], [1, 3]]);
});

test("C and K average independent observations before Bradley-Terry", () => {
  const cache: ScoreCache = {};
  const observations = [[1, 0], [1, 0], [0, 1], [0.2, 0.8]];
  for (let i = 0; i < observations.length; i++) {
    cache[cacheKey(i < 2 ? "first" : "second", "task", 0, 1, i % 2)] = {
      score_A: observations[i][0], score_B: observations[i][1],
    };
  }
  const rewards = directedReward(cache, "task", 0, 1, ["first", "second"], 2);
  expect(rewards[0]).toBeCloseTo(0.55, 14);
  expect(rewards[1]).toBeCloseTo(0.45, 14);
  const wins = [0, 0];
  accumulate([[0, 1]], () => rewards, wins, [0, 0]);
  expect(wins[0]).toBeCloseTo(0.52497918747894, 14);
});

test("C times K performs distinct verifier calls for each directed comparison", async () => {
  const prompts: string[] = [];
  const client = {
    scoreReply: async (prompt: string) => {
      prompts.push(prompt);
      return {
        text: "<score_A>A</score_A><score_B>A</score_B>",
        tokens: ["<score_A>", "A", "</score_A><score_B>", "A"],
        positionLogprobs: [[], [["A", 0]], [], [["A", 0]]],
      };
    },
  } as unknown as VerifierClient;
  const result = await select("task", [{ trace: "FIRST" }, { trace: "SECOND" }], {
    client, progress: false, pivots: 1, nEvaluations: 3,
    criteria: [
      { id: "one", name: "One", description: "CRITERION_ONE" },
      { id: "two", name: "Two", description: "CRITERION_TWO" },
    ],
  });
  expect(prompts).toHaveLength(result.nComparisons * 2 * 3);
  expect(prompts.filter((prompt) => prompt.includes("CRITERION_ONE"))).toHaveLength(6);
  expect(prompts.filter((prompt) => prompt.includes("CRITERION_TWO"))).toHaveLength(6);
  expect(result.scores).toEqual([0.5, 0.5]);
  expect(result.paperEquivalent).toBe(true);
});

test("Terminal-Bench evidence reaches the prompt without length truncation", () => {
  const output = `${"early evidence\n".repeat(2000)}FINAL_VERIFICATION_FAILED`;
  const trace = formatTrace({ steps: [{
    source: "agent", step_id: 1, message: "I believe this succeeded",
    tool_calls: [{ arguments: { keystrokes: "run-final-verification\n" } }],
    observation: { results: [{ content: output }] },
  }] });
  expect(trace).toContain("[Command] run-final-verification");
  expect(trace).toContain(output);
  const prompt = buildPrompt("task", trace, "other trajectory", {
    id: "output", name: "Output", description: "Check final output",
  });
  expect(prompt).toContain(trace);
  expect(prompt).toContain("FINAL_VERIFICATION_FAILED");
});
