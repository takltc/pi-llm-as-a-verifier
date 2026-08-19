import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  cacheKey,
  loadCache,
  saveCache,
  type CacheContext,
} from "../src/cache.ts";
import { parseModelSelector, VerifierClient } from "../src/client.ts";
import verifierExtension, { parseArgs, tokenizeArgs } from "../src/index.ts";
import { detectInputLayout, loadCandidateDir, loadTerminalDir } from "../src/loader.ts";
import { TERMINAL_BENCH_CRITERIA } from "../src/prompt.ts";
import { runBenchmark, scoreDirectedPairs } from "../src/run.ts";
import type { VerifierReply } from "../src/scale.ts";
import { select } from "../src/select.ts";

const temporaryPaths: string[] = [];

afterEach(() => {
  for (const path of temporaryPaths.splice(0)) {
    rmSync(path, { recursive: true, force: true });
  }
});

function tempDir(): string {
  const path = join(tmpdir(), `omp-verifier-${crypto.randomUUID()}`);
  mkdirSync(path, { recursive: true });
  temporaryPaths.push(path);
  return path;
}

function context(overrides: Partial<CacheContext> = {}): CacheContext {
  return {
    criterionId: "criterion",
    criterionName: "Criterion",
    criterionDescription: "Evaluate the criterion.",
    problem: "problem",
    traceA: "trace A",
    traceB: "trace B",
    model: "deepseek-v4-flash",
    effort: "xhigh",
    maxTokens: 65536,
    baseUrl: "https://opencode.ai/zen/go/v1",
    groundTruthNote: "terminal output is ground truth",
    promptVersion: "terminal-bench-pairwise-v1",
    ...overrides,
  };
}

function reply(scoreA: number, scoreB: number): VerifierReply {
  const distribution = (score: number): Array<[string, number]> => {
    if (score <= 0) return [["T", 0]];
    if (score >= 1) return [["A", 0]];
    return [
      ["A", Math.log(score)],
      ["T", Math.log(1 - score)],
    ];
  };
  return {
    text: "<score_A> A </score_A>\n<score_B> A </score_B>",
    tokens: ["<score_A>", " A", " </score_A>\n<score_B>", " A", " </score_B>"],
    positionLogprobs: [
      [["<score_A>", 0]],
      distribution(scoreA),
      [[" </score_A>\n<score_B>", 0]],
      distribution(scoreB),
      [[" </score_B>", 0]],
    ],
  };
}

class MatrixClient extends VerifierClient {
  calls = 0;

  constructor() {
    super({ apiKey: "test-key", model: "opencode-go/deepseek-v4-flash:xhigh" });
  }

  override async scoreReply(prompt: string): Promise<VerifierReply> {
    this.calls += 1;
    const match = /\*\*Trajectory A:\*\*\n([\s\S]*?)\n\n\*\*Trajectory B:\*\*\n([\s\S]*?)\n\n\*\*Rating Scale/.exec(prompt);
    if (!match) throw new Error("test prompt did not contain trajectories");
    const a = Number(/candidate-(\d+)/.exec(match[1])?.[1]);
    const b = Number(/candidate-(\d+)/.exec(match[2])?.[1]);
    const directedDiff = new Map<string, number>([
      ["4,2", 1],
      ["2,3", 1],
      ["3,0", 0],
      ["0,1", -0.5],
      ["1,4", 0],
      ["0,4", 1],
      ["2,4", 1],
      ["3,4", 0.5],
    ]);
    const diff = directedDiff.get(`${a},${b}`) ?? 0;
    return reply((diff + 1) / 2, (1 - diff) / 2);
  }
}

class NoScoreClient extends VerifierClient {
  constructor() {
    super({ apiKey: "test-key" });
  }

  override async scoreReply(): Promise<VerifierReply> {
    return { text: "analysis without verdict", tokens: ["analysis"], positionLogprobs: [[["analysis", 0]]] };
  }
}

function trajectory(reward: 0 | 1, id: number): Record<string, unknown> {
  return {
    trial_name: `trial-${id}`,
    reward,
    trajectory: {
      steps: [
        { source: "user", message: "Solve the task exactly." },
        {
          source: "agent",
          step_id: 1,
          message: `candidate-${id}`,
          tool_calls: [{ arguments: { keystrokes: "npm test" } }],
          observation: { results: [{ content: "tests passed" }] },
        },
      ],
    },
  };
}

describe("cache identity and durability", () => {
  test("key changes for every prompt-affecting identity field", () => {
    const base = cacheKey("criterion", "task", 0, 1, 0, context());
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ traceA: "changed" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ problem: "changed" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ model: "other" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ effort: "high" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ criterionName: "Renamed" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ criterionDescription: "Changed rubric" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ promptVersion: "v2" }))).not.toBe(base);
  });

  test("corrupt cache is isolated and later writes use an atomic merged file", () => {
    const dir = tempDir();
    const file = join(dir, "nested", "cache.json");
    mkdirSync(join(dir, "nested"), { recursive: true });
    writeFileSync(file, "{corrupt");
    expect(loadCache(file)).toEqual({});
    saveCache(file, { first: { score_A: 0.2, score_B: 0.8 } });
    saveCache(file, { second: { score_A: 0.7, score_B: 0.3 } });
    expect(loadCache(file)).toEqual({
      first: { score_A: 0.2, score_B: 0.8 },
      second: { score_A: 0.7, score_B: 0.3 },
    });
    expect(readdirSync(join(dir, "nested")).some((name) => name.includes(".tmp-") || name.endsWith(".lock"))).toBe(false);
  });

  test("independent writer processes preserve each entry", async () => {
    const dir = tempDir();
    const file = join(dir, "scores.json");
    const cacheModule = resolve(import.meta.dir, "../src/cache.ts");
    const children = Array.from({ length: 8 }, (_, index) => {
      const source = `import { saveCache } from ${JSON.stringify(cacheModule)}; saveCache(${JSON.stringify(file)}, {${JSON.stringify(`k${index}`)}: {score_A: ${index / 10}, score_B: ${(9 - index) / 10}}});`;
      return Bun.spawn(["bun", "-e", source], { stdout: "ignore", stderr: "ignore" });
    });
    const codes = await Promise.all(children.map((child) => child.exited));
    expect(codes).toEqual(new Array(8).fill(0));
    expect(Object.keys(loadCache(file)).sort()).toEqual(
      Array.from({ length: 8 }, (_, index) => `k${index}`),
    );
  });
});

describe("benchmark pipeline", () => {
  test("keeps phase-A scores when phase B runs without a cache file", async () => {
    const client = new MatrixClient();
    const tasks = {
      task: Array.from({ length: 5 }, (_, index) => ({
        trialName: `candidate-${index}`,
        reward: index === 2 ? (1 as const) : (0 as const),
        problem: "Solve the task exactly.",
        trace: `candidate-${index}`,
      })),
    };
    const stats = await runBenchmark(tasks, TERMINAL_BENCH_CRITERIA, {
      client,
      pivots: 1,
      nEvaluations: 1,
      seed: 0,
      maxWorkers: 1,
      progress: false,
    });
    expect(stats.totalComparisons).toBe(9);
    expect(stats.bestPerTask.task.index).toBe(2);
    expect(stats.bestPerTask.task.reward).toBe(1);
    // Pair (1,4) appears in both phases and is reused from the in-memory cache.
    expect(client.calls).toBe(8 * 3);
  });

  test("failed replies return temporary ties and never persist", async () => {
    const dir = tempDir();
    const cacheFile = join(dir, "scores.json");
    const tasks = {
      task: [
        { trialName: "a", reward: 0 as const, problem: "p", trace: "a" },
        { trialName: "b", reward: 1 as const, problem: "p", trace: "b" },
      ],
    };
    const scores = await scoreDirectedPairs(
      new NoScoreClient(),
      tasks,
      { task: [[0, 1]] },
      [TERMINAL_BENCH_CRITERIA[0]],
      undefined,
      1,
      1,
      cacheFile,
      { progress: false },
    );
    expect(Object.values(scores)).toEqual([{ score_A: 0.5, score_B: 0.5 }]);
    expect(existsSync(cacheFile)).toBe(false);
  });

  test("select injects the Terminal-Bench terminal-output ground-truth note by default", async () => {
    let prompt = "";
    class CaptureClient extends VerifierClient {
      constructor() { super({ apiKey: "test-key" }); }
      override async scoreReply(value: string): Promise<VerifierReply> {
        prompt = value;
        return reply(1, 0);
      }
    }
    await select("do work", [{ name: "a", trace: "a" }, { name: "b", trace: "b" }], {
      client: new CaptureClient(),
      nEvaluations: 1,
      maxWorkers: 1,
      progress: false,
    });
    expect(prompt).toContain("Focus on TERMINAL OUTPUT as ground truth");
  });

  test("benchmark validation reports non-string task fields", async () => {
    await expect(
      runBenchmark(
        {
          task: [
            { trialName: "a", reward: 0, problem: 42 as unknown as string, trace: "a" },
            { trialName: "b", reward: 1, problem: "p", trace: "b" },
          ],
        },
        [TERMINAL_BENCH_CRITERIA[0]],
        { progress: false },
      ),
    ).rejects.toThrow("problem and trace must be strings");
  });
});

describe("loader, client selector, and extension interface", () => {
  test("recognizes Terminal-Bench layout by trajectory files and validates reward", () => {
    const dir = tempDir();
    mkdirSync(join(dir, "notes"));
    mkdirSync(join(dir, "task-a"));
    writeFileSync(join(dir, "task-a", "one_trajectory.json"), JSON.stringify(trajectory(1, 1)));
    writeFileSync(join(dir, "task-a", "two_trajectory.json"), JSON.stringify(trajectory(0, 2)));
    expect(detectInputLayout(dir)).toBe("terminal");
    expect(loadTerminalDir(dir).tasks["task-a"]).toHaveLength(2);

    const bad = tempDir();
    writeFileSync(join(bad, "candidate.json"), JSON.stringify({ ...trajectory(1, 1), reward: 2 }));
    expect(() => loadCandidateDir(bad, "task")).toThrow("reward must be numeric 0 or 1");

    const malformed = tempDir();
    writeFileSync(
      join(malformed, "candidate.json"),
      JSON.stringify({ ...trajectory(1, 3), trajectory: { steps: [null] } }),
    );
    expect(() => loadCandidateDir(malformed, "task")).toThrow("step 0 must be an object");
  });

  test("rejects mixed layouts and parses quoted extension command arguments", () => {
    const dir = tempDir();
    mkdirSync(join(dir, "task"));
    writeFileSync(join(dir, "task", "one_trajectory.json"), JSON.stringify(trajectory(1, 1)));
    writeFileSync(join(dir, "candidate.json"), JSON.stringify(trajectory(0, 2)));
    expect(() => detectInputLayout(dir)).toThrow("Mixed trajectory layouts");
    expect(tokenizeArgs("'folder with spaces' --note \"terminal output only\"")).toEqual(["folder with spaces", "--note", "terminal output only"]);
    const parsed = parseArgs("'folder with spaces' --k=2");
    expect(parsed.positionals).toEqual(["folder with spaces"]);
    expect(parsed.optInt("k")).toBe(2);
    expect(() => parseArgs("'unterminated")).toThrow("Unterminated quote");
  });

  test("splits the OMP selector before the API call", () => {
    expect(parseModelSelector("opencode-go/deepseek-v4-flash:xhigh")).toEqual({
      provider: "opencode-go",
      model: "deepseek-v4-flash",
      effort: "xhigh",
    });
  });

  test("client keeps token and logprob positions aligned for malformed entries", async () => {
    const originalFetch = globalThis.fetch;
    let request: Record<string, unknown> | undefined;
    globalThis.fetch = (async (_input, init) => {
      request = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          choices: [
            {
              message: { content: "<score_A> A </score_A>" },
              finish_reason: "stop",
              logprobs: {
                content: [
                  { token: "<score_A>", top_logprobs: [{ token: "<score_A>", logprob: 0 }] },
                  { token: " A", top_logprobs: [{ token: " A", logprob: 0 }] },
                  { token: "", top_logprobs: [null], logprob: "invalid" },
                ],
              },
            },
          ],
          usage: {},
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    try {
      const reply = await new VerifierClient({
        apiKey: "test-key",
        model: "opencode-go/deepseek-v4-flash:xhigh",
      }).scoreReply("short prompt", { maxTokens: 128 });
      expect(request?.model).toBe("deepseek-v4-flash");
      expect(request?.reasoning_effort).toBe("xhigh");
      expect(request?.logprobs).toBe(true);
      expect(request?.top_logprobs).toBe(20);
      expect(reply.tokens?.length).toBe(3);
      expect(reply.positionLogprobs?.length).toBe(3);
      expect(reply.positionLogprobs?.[2]).toEqual([]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("registers both slash commands and verifier_select with OMP's extension API", () => {
    const commands: string[] = [];
    const tools: string[] = [];
    const schema = () => {
      const chain = {
        min: () => chain,
        max: () => chain,
        int: () => chain,
        optional: () => chain,
        describe: () => chain,
      };
      return chain;
    };
    const pi = {
      zod: { string: schema, number: schema, array: schema, object: schema },
      setLabel: () => {},
      registerCommand: (name: string) => commands.push(name),
      registerTool: (definition: { name: string }) => tools.push(definition.name),
      on: () => {},
    };
    verifierExtension(pi as never);
    expect(commands.sort()).toEqual(["vcompare", "verify"]);
    expect(tools).toEqual(["verifier_select"]);
  });
});
