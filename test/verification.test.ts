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
import { createVerifierClient, VerifierClient, type VerifierConfig } from "../src/client.ts";
import verifierExtension, {
  createDefaultVerifierClient,
  parseArgs,
  tokenizeArgs,
} from "../src/index.ts";
import { detectInputLayout, loadCandidateDir, loadTerminalDir } from "../src/loader.ts";
import { TERMINAL_BENCH_CRITERIA } from "../src/prompt.ts";
import {
  runBenchmark,
  scoreDirectedPairs,
  SELF_VERIFICATION_DEFAULTS,
  validateVerifyOptions,
} from "../src/run.ts";
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
    provider: "opencode-go",
    api: "openai-responses",
    model: "deepseek-v4-flash",
    effort: "high",
    maxTokens: 32768,
    baseUrl: "https://opencode.ai/zen/go/v1",
    requestIdentity: "request-identity",
    groundTruthNote: "terminal output is ground truth",
    promptVersion: "terminal-bench-pairwise-v1",
    ...overrides,
  };
}

function clientConfig(overrides: Partial<VerifierConfig> = {}): VerifierConfig {
  return {
    baseUrl: "https://opencode.ai/zen/go/v1",
    apiKey: "test-key",
    provider: "opencode-go",
    api: "openai-completions",
    modelId: "deepseek-v4-flash",
    effort: "high",
    maxTokens: 32768,
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
    super(clientConfig());
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
    super(clientConfig());
  }

  override async scoreReply(): Promise<VerifierReply> {
    return { text: "analysis without verdict", tokens: ["analysis"], positionLogprobs: [[["analysis", 0]]] };
  }
}

class CancellingClient extends VerifierClient {
  started = 0;
  settled = 0;
  aborted = 0;
  constructor(
    private readonly failFirst: boolean,
    private readonly slowMs = 100,
    private readonly failMs = 5,
  ) {
    super(clientConfig());
  }

  override scoreReply(_prompt: string, opts: { signal?: AbortSignal } = {}): Promise<VerifierReply> {
    const call = this.started++;
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (callback: () => void): void => {
        if (settled) return;
        settled = true;
        this.settled += 1;
        callback();
      };
      const timer = setTimeout(() => {
        finish(() => {
          if (this.failFirst && call === 0) reject(new Error("first verifier failure"));
          else resolve(reply(1, 0));
        });
      }, call === 0 ? this.failMs : this.slowMs);
      opts.signal?.addEventListener("abort", () => {
        clearTimeout(timer);
        this.aborted += 1;
        finish(() => reject(opts.signal?.reason ?? new DOMException("aborted", "AbortError")));
      }, { once: true });
    });
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
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ provider: "other" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ api: "openai-completions" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ model: "other" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ effort: "xhigh" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ requestIdentity: "rotated" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ criterionName: "Renamed" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ criterionDescription: "Changed rubric" }))).not.toBe(base);
    expect(cacheKey("criterion", "task", 0, 1, 0, context({ promptVersion: "v2" }))).not.toBe(base);
  });

  test("request identity is hashed and never stores credential material", () => {
    const first = new VerifierClient(clientConfig({ apiKey: "secret-one", headers: { "X-Route": "a" } }));
    const second = new VerifierClient(clientConfig({ apiKey: "secret-two", headers: { "X-Route": "a" } }));
    const third = new VerifierClient(clientConfig({ apiKey: "secret-one", headers: { "X-Route": "b" } }));
    const fourth = new VerifierClient(clientConfig({
      compat: { reasoningEffortMap: { high: "xhigh" } },
    }));
    expect(first.requestIdentity).not.toBe(second.requestIdentity);
    expect(first.requestIdentity).not.toBe(third.requestIdentity);
    expect(first.requestIdentity).not.toBe(fourth.requestIdentity);
    expect(first.requestIdentity).not.toContain("secret-one");
    expect(first.requestIdentity).not.toContain("X-Route");
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

  test("raise cancels in-flight workers, waits for them, and preserves completed cache entries", async () => {
    const dir = tempDir();
    const cacheFile = join(dir, "scores.json");
    const client = new CancellingClient(true, 1, 20);
    const tasks = {
      task: [
        { trialName: "a", reward: 0 as const, problem: "p", trace: "a" },
        { trialName: "b", reward: 1 as const, problem: "p", trace: "b" },
        { trialName: "c", reward: 0 as const, problem: "p", trace: "c" },
      ],
    };
    await expect(scoreDirectedPairs(
      client,
      tasks,
      { task: [[0, 1], [0, 2]] },
      [TERMINAL_BENCH_CRITERIA[0]],
      undefined,
      1,
      2,
      cacheFile,
      { onError: "raise", progress: false },
    )).rejects.toThrow("first verifier failure");
    expect(client.started).toBe(client.settled);
    expect(client.aborted).toBeGreaterThan(0);
    expect(Object.keys(loadCache(cacheFile)).length).toBeGreaterThanOrEqual(1);
  });

  test("external cancellation propagates in tie mode", async () => {
    const controller = new AbortController();
    const client = new CancellingClient(false, 200);
    const promise = scoreDirectedPairs(
      client,
      {
        task: [
          { trialName: "a", reward: 0 as const, problem: "p", trace: "a" },
          { trialName: "b", reward: 1 as const, problem: "p", trace: "b" },
        ],
      },
      { task: [[0, 1]] },
      [TERMINAL_BENCH_CRITERIA[0]],
      undefined,
      1,
      1,
      undefined,
      { onError: "tie", progress: false, signal: controller.signal },
    );
    setTimeout(() => controller.abort(new Error("caller cancelled")), 5);
    await expect(promise).rejects.toThrow("caller cancelled");
    expect(client.started).toBe(client.settled);
  });

  test("select injects the Terminal-Bench terminal-output ground-truth note by default", async () => {
    let prompt = "";
    class CaptureClient extends VerifierClient {
      constructor() { super(clientConfig()); }
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

describe("loader, OMP model binding, and extension interface", () => {
  test("recognizes Terminal-Bench layout by trajectory files and validates reward", () => {
    const dir = tempDir();
    mkdirSync(join(dir, "notes"));
    mkdirSync(join(dir, "task-a"));
    writeFileSync(join(dir, "task-a", "one_trajectory.json"), JSON.stringify(trajectory(1, 1)));
    writeFileSync(join(dir, "task-a", "two_trajectory.json"), JSON.stringify(trajectory(0, 2)));
    expect(detectInputLayout(dir)).toBe("terminal");
    expect(loadTerminalDir(dir).tasks["task-a"]).toHaveLength(2);

    const bad = tempDir();
    mkdirSync(join(bad, "task-b"));
    writeFileSync(
      join(bad, "task-b", "bad_trajectory.json"),
      JSON.stringify({ ...trajectory(1, 1), reward: 2 }),
    );
    expect(() => loadTerminalDir(bad)).toThrow("Trajectory reward must be numeric 0 or 1");

    const malformed = tempDir();
    mkdirSync(join(malformed, "task-c"));
    writeFileSync(
      join(malformed, "task-c", "malformed_trajectory.json"),
      JSON.stringify({ ...trajectory(1, 3), trajectory: { steps: [null] } }),
    );
    expect(() => loadTerminalDir(malformed)).toThrow("Trajectory step 0 must be an object");
  });

  test("loads a flat candidate directory with a shared task", () => {
    const dir = tempDir();
    writeFileSync(
      join(dir, "alpha.json"),
      JSON.stringify({ task: "solve the issue", trace: "alpha trace" }),
    );
    writeFileSync(
      join(dir, "beta.json"),
      JSON.stringify({ task: "solve the issue", name: "Beta", trace: "beta trace" }),
    );
    expect(detectInputLayout(dir)).toBe("candidates");
    expect(loadCandidateDir(dir)).toEqual({
      task: "solve the issue",
      candidates: [
        { name: "alpha", trace: "alpha trace" },
        { name: "Beta", trace: "beta trace" },
      ],
    });
  });

  test("rejects invalid flat candidate sets", () => {
    const one = tempDir();
    writeFileSync(join(one, "only.json"), JSON.stringify({ task: "t", trace: "x" }));
    expect(() => loadCandidateDir(one)).toThrow("At least two candidate JSON files");

    const mismatch = tempDir();
    writeFileSync(join(mismatch, "a.json"), JSON.stringify({ task: "a", trace: "x" }));
    writeFileSync(join(mismatch, "b.json"), JSON.stringify({ task: "b", trace: "y" }));
    expect(() => loadCandidateDir(mismatch)).toThrow("Candidate task mismatch");

    const duplicate = tempDir();
    writeFileSync(join(duplicate, "a.json"), JSON.stringify({ task: "t", name: "same", trace: "x" }));
    writeFileSync(join(duplicate, "b.json"), JSON.stringify({ task: "t", name: "same", trace: "y" }));
    expect(() => loadCandidateDir(duplicate)).toThrow("Duplicate candidate name");
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
    expect(parsed.opts.k).toBe("2");
    expect(() => parseArgs("'unterminated")).toThrow("Unterminated quote");
  });

  test("uses the configured OMP default model, explicit effort, and authenticated request settings", async () => {
    const activeModel = {
      provider: "active-provider",
      id: "active-model",
      api: "openai-completions",
      baseUrl: "https://active.example/v1",
      maxTokens: 100_000,
      reasoning: true,
      thinking: { efforts: ["low", "high", "xhigh"] },
    };
    const defaultModel = {
      provider: "default-provider",
      id: "default-model",
      requestModelId: "default-wire-model",
      api: "openai-responses",
      baseUrl: "https://default.example/v1",
      headers: { "X-Model": "model-header" },
      maxTokens: 100_000,
      reasoning: true,
      thinking: { efforts: ["low", "high", "xhigh"] },
    };
    let authenticatedModel: unknown;
    const client = await createVerifierClient({
      model: activeModel,
      models: {
        current: () => activeModel,
        resolve: (spec: string) => spec === "@default" ? defaultModel : undefined,
      },
      modelRegistry: {
        getApiKeyAndHeaders: async (model: unknown) => {
          authenticatedModel = model;
          return {
            ok: true,
            apiKey: "session-key",
            headers: { "X-Auth": "auth-header" },
          };
        },
      },
      defaultThinkingLevel: "xhigh",
    } as never);
    expect(authenticatedModel).toBe(defaultModel);
    expect(client.provider).toBe("default-provider");
    expect(client.api).toBe("openai-responses");
    expect(client.model).toBe("default-wire-model");
    expect(client.baseUrl).toBe("https://default.example/v1");
    expect(client.apiKey).toBe("session-key");
    expect(client.effort).toBe("xhigh");
    expect(client.maxTokens).toBe(32768);
    expect(client.headers).toEqual({ "X-Model": "model-header", "X-Auth": "auth-header" });
  });

  test("reads the explicit thinking level from OMP modelRoles.default", async () => {
    const defaultModel = {
      provider: "opencode-go",
      id: "deepseek-v4-flash",
      api: "openai-responses",
      baseUrl: "https://opencode.ai/zen/go/v1",
      maxTokens: 100_000,
      reasoning: true,
      thinking: { efforts: ["low", "high", "xhigh"] },
    };
    const client = await createDefaultVerifierClient(
      {
        pi: {
          settings: {
            getModelRole: (role: string) =>
              role === "default"
                ? "opencode-go/deepseek-v4-flash:xhigh"
                : undefined,
          },
        },
      } as never,
      {
        models: {
          resolve: (spec: string) => spec === "@default" ? defaultModel : undefined,
        },
        modelRegistry: {
          getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "session-key" }),
        },
      } as never,
    );
    expect(client.effort).toBe("xhigh");
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
        ...clientConfig({ maxTokens: 128, headers: { "X-Test": "preserved" } }),
      }).scoreReply("short prompt");
      expect(request?.model).toBe("deepseek-v4-flash");
      expect(request?.reasoning_effort).toBe("high");
      expect(request?.max_tokens).toBe(128);
      expect(request?.temperature).toBeUndefined();
      expect(request?.logprobs).toBe(true);
      expect(request?.top_logprobs).toBe(20);
      expect(reply.tokens?.length).toBe(3);
      expect(reply.positionLogprobs?.length).toBe(3);
      expect(reply.positionLogprobs?.[2]).toEqual([]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("uses the Responses transport and parses output token logprobs", async () => {
    const originalFetch = globalThis.fetch;
    let requestUrl = "";
    let request: Record<string, unknown> | undefined;
    globalThis.fetch = (async (input, init) => {
      requestUrl = String(input);
      request = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          status: "completed",
          output: [
            {
              type: "message",
              content: [
                {
                  type: "output_text",
                  text: "<score_A> A </score_A>",
                  logprobs: [
                    { token: "<score_A>", logprob: 0, top_logprobs: [] },
                    {
                      token: " A",
                      logprob: -0.1,
                      top_logprobs: [
                        { token: " A", logprob: -0.1 },
                        { token: " B", logprob: -2.4 },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
          usage: {
            input_tokens: 30,
            output_tokens: 7,
            input_tokens_details: { cached_tokens: 20 },
            output_tokens_details: { reasoning_tokens: 4 },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    try {
      const result = await new VerifierClient({
        ...clientConfig({ api: "openai-responses", effort: "xhigh", maxTokens: 256 }),
      }).scoreReply("short prompt");
      expect(requestUrl).toBe("https://opencode.ai/zen/go/v1/responses");
      expect(request?.max_output_tokens).toBe(256);
      expect(request?.top_logprobs).toBe(20);
      expect(request?.include).toEqual(["message.output_text.logprobs"]);
      expect(request?.reasoning).toEqual({ effort: "xhigh" });
      expect(request?.temperature).toBeUndefined();
      expect(request?.messages).toBeUndefined();
      expect(result.text).toBe("<score_A> A </score_A>");
      expect(result.tokens).toEqual(["<score_A>", " A"]);
      expect(result.positionLogprobs?.[1]).toEqual([
        [" A", -0.1],
        [" B", -2.4],
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("keyless OMP auth preserves routing headers without Bearer N/A", async () => {
    const originalFetch = globalThis.fetch;
    let requestHeaders: Headers | undefined;
    globalThis.fetch = (async (_input, init) => {
      requestHeaders = new Headers(init?.headers);
      return new Response(
        JSON.stringify({
          choices: [{
            message: { content: "<score_A> A </score_A>" },
            logprobs: { content: [{ token: " A", logprob: 0 }] },
          }],
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    try {
      await new VerifierClient(clientConfig({
        apiKey: "N/A",
        headers: { "X-Route": "gateway-route" },
      })).scoreReply("short prompt");
      expect(requestHeaders?.get("authorization")).toBeNull();
      expect(requestHeaders?.get("x-route")).toBe("gateway-route");
      expect(requestHeaders?.get("content-type")).toBe("application/json");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("redacts API credentials and routing header values from API errors", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({ error: "Bearer test-key route-secret cookie-secret" }),
        { status: 401, statusText: "Unauthorized" },
      )) as unknown as typeof fetch;
    try {
      const client = new VerifierClient(clientConfig({
        headers: {
          "X-Route": "route-secret",
          Cookie: "session=cookie-secret",
        },
      }));
      let message = "";
      try {
        await client.scoreReply("short prompt");
      } catch (error) {
        message = error instanceof Error ? error.message : String(error);
      }
      expect(message).toContain("[redacted]");
      expect(message).not.toContain("test-key");
      expect(message).not.toContain("route-secret");
      expect(message).not.toContain("cookie-secret");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("registers /verify and verifier_select with OMP's extension API", () => {
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
    expect(commands).toEqual(["verify"]);
    expect(tools).toEqual(["verifier_select"]);
  });

  test("uses the paper Bo5 defaults without caller-supplied tuning parameters", () => {
    const tasks = {
      task: [
        { trialName: "a", reward: 0 as const, problem: "p", trace: "a" },
        { trialName: "b", reward: 1 as const, problem: "p", trace: "b" },
      ],
    };
    expect(validateVerifyOptions(tasks, TERMINAL_BENCH_CRITERIA, {})).toEqual({
      k: SELF_VERIFICATION_DEFAULTS.pivots,
      nReps: SELF_VERIFICATION_DEFAULTS.nEvaluations,
      seed: SELF_VERIFICATION_DEFAULTS.seed,
      maxWorkers: SELF_VERIFICATION_DEFAULTS.maxWorkers,
    });
  });
});
