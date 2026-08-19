import { describe, expect, test } from "bun:test";
import { afterEach } from "bun:test";
import { readFileSync } from "node:fs";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import type { AssistantMessage, Context, Model, SimpleStreamOptions } from "@oh-my-pi/pi-ai";
import { cacheKey, loadCache, saveCache, type CacheContext } from "../src/cache.ts";
import {
  AUTO_CANDIDATE_COUNT,
  createAutoVerifierStream,
  normalizeCandidateCount,
  serializeAssistantMessage,
  serializeContext,
} from "../src/auto.ts";
import { createDefaultVerifierClient, resolvePluginSettings } from "../src/index.ts";
import { CODING_AGENT_CRITERIA, CODING_AGENT_GROUND_TRUTH_NOTE } from "../src/prompt.ts";
import { SELF_VERIFICATION_DEFAULTS } from "../src/run.ts";
import { VerifierClient, type VerifierConfig } from "../src/client.ts";
import type { VerifierReply } from "../src/scale.ts";

const temporaryFiles: string[] = [];

afterEach(() => {
  for (const path of temporaryFiles.splice(0)) Bun.file(path).delete().catch(() => undefined);
});

function clientConfig(overrides: Partial<VerifierConfig> = {}): VerifierConfig {
  return {
    baseUrl: "https://example.test/v1",
    apiKey: "test-key",
    provider: "opencode-go",
    api: "openai-completions",
    modelId: "deepseek-v4-flash-0731",
    effort: "max",
    maxTokens: 32768,
    ...overrides,
  };
}

function usage() {
  return {
    input: 1,
    output: 1,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 2,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function message(index: number, withTool = false): AssistantMessage {
  const content: AssistantMessage["content"] = [
    { type: "thinking", thinking: "candidate thinking " + index },
    { type: "text", text: "candidate text " + index },
  ];
  if (withTool) {
    content.push({
      type: "toolCall",
      id: "call-" + index,
      name: "bash",
      arguments: { command: "printf candidate-" + index },
    });
  }
  return {
    role: "assistant",
    content,
    api: "openai-completions",
    provider: "opencode-go",
    model: "deepseek-v4-flash-0731",
    responseId: "response-" + index,
    usage: usage(),
    stopReason: withTool ? "toolUse" : "stop",
    stopDetails: { type: "test" },
    providerPayload: { type: "openaiResponsesHistory", items: [{ index }] },
    duration: 10 + index,
    ttft: 2,
    timestamp: 1000 + index,
  };
}

function fakeCandidateStreamFactory(
  calls: Array<{ context: Context; options: SimpleStreamOptions }>,
  failing = new Set<number>(),
) {
  let nextIndex = 0;
  return (_model: Model, context: Context, options: SimpleStreamOptions = {}) => {
    const index = nextIndex++;
    calls.push({ context, options });
    const stream = new AssistantMessageEventStream();
    queueMicrotask(() => {
      if (failing.has(index)) {
        stream.fail(new Error("candidate " + index + " failed"));
        return;
      }
      const result = message(index, index === 2);
      stream.push({ type: "start", partial: result });
      stream.push({
        type: "done",
        reason: result.stopReason === "toolUse" ? "toolUse" : "stop",
        message: result,
      });
    });
    return stream;
  };
}

class RankingVerifier extends VerifierClient {
  calls = 0;

  constructor(private readonly fail = false) {
    super(clientConfig());
  }

  override async scoreReply(prompt: string): Promise<VerifierReply> {
    this.calls += 1;
    if (this.fail) throw new Error("verifier unavailable");
    const a = /candidate text (\d+)/.exec(prompt)?.[1];
    const b = /candidate text (\d+)/.exec(prompt.slice(prompt.indexOf("Trajectory B")))?.[1];
    const aWins = a === "2" && b !== "2";
    const bWins = b === "2" && a !== "2";
    const scoreA = aWins ? "A" : bWins ? "T" : "A";
    const scoreB = bWins ? "A" : aWins ? "T" : "A";
    return { text: "<score_A> " + scoreA + " </score_A>\n<score_B> " + scoreB + " </score_B>" };
  }
}

function model(): Model {
  return {
    provider: "opencode-go",
    id: "deepseek-v4-flash-0731",
    name: "DeepSeek V4 Flash",
    api: "openai-completions",
    baseUrl: "https://example.test/v1",
    reasoning: true,
    thinking: { mode: "effort", efforts: ["low", "high", "max"] },
    input: ["text"],
    supportsTools: true,
    cost: { input: 1, output: 1, cacheRead: 1, cacheWrite: 1 },
    contextWindow: 128000,
    maxTokens: 64000,
  } as unknown as Model;
}

function context(): Context {
  return {
    systemPrompt: ["You are an OMP coding agent."],
    messages: [{ role: "user", content: "Fix the failing test and verify it.", timestamp: 1 }],
    tools: [{ name: "bash", description: "run a shell command", parameters: {} }],
  };
}

async function collect(stream: AssistantMessageEventStream) {
  const events: unknown[] = [];
  const reader = (async () => {
    for await (const event of stream) events.push(event);
  })();
  const result = await stream.result();
  await reader;
  return { events, result };
}

describe("automatic request-level provider", () => {
  test("generates three candidates and replays the verified winner", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const verifier = new RankingVerifier();
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
    }, context());
    const { events, result } = await collect(stream);

    expect(calls).toHaveLength(3);
    expect(verifier.calls).toBeGreaterThan(0);
    expect(verifier.calls % 6).toBe(0);
    expect(calls.every((call) => call.options.statefulResponses === false)).toBe(true);
    expect(calls.every((call) => call.options.sessionId === undefined)).toBe(true);
    expect(calls.every((call) => call.options.providerSessionState === undefined)).toBe(true);
    expect(calls.every((call) => call.options.promptCacheKey === undefined)).toBe(true);
    expect(result.responseId).toBe("response-2");
    expect(result.stopReason).toBe("toolUse");
    expect(result.content.some((block) => block.type === "toolCall")).toBe(true);
    expect((events as Array<{ type: string }>).map((event) => event.type)).toEqual([
      "start",
      "thinking_start",
      "thinking_delta",
      "thinking_end",
      "text_start",
      "text_delta",
      "text_end",
      "toolcall_start",
      "toolcall_delta",
      "toolcall_end",
      "done",
    ]);
    const replayEvents = events as Array<{ type: string; partial?: AssistantMessage }>;
    const textStart = replayEvents.find((event) => event.type === "text_start");
    const textDelta = replayEvents.find((event) => event.type === "text_delta");
    const toolStart = replayEvents.find((event) => event.type === "toolcall_start");
    const toolDelta = replayEvents.find((event) => event.type === "toolcall_delta");
    expect(textStart?.partial?.content[1]).toEqual({ type: "text", text: "" });
    expect(textDelta?.partial?.content[1]).toMatchObject({ type: "text", text: "candidate text 2" });
    expect(toolStart?.partial?.content[2]).toMatchObject({
      type: "toolCall",
      id: "call-2",
      name: "bash",
      arguments: {},
    });
    expect(toolDelta?.partial?.content[2]).toMatchObject({
      type: "toolCall",
      arguments: { command: "printf candidate-2" },
    });
  });

  test("uses the configured candidate count", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const stream = createAutoVerifierStream(
      {
        originalModel: model(),
        verifierClient: new RankingVerifier(),
        apiKeyResolver: () => "original-key",
        streamSimpleFn: fakeCandidateStreamFactory(calls),
      },
      context(),
      {},
      { candidateCount: 4 },
    );
    await collect(stream);
    expect(calls).toHaveLength(4);
  });

  test("falls back to the first successful candidate when verification fails", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const degraded: unknown[] = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(true),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
      onDegraded: (event) => degraded.push(event),
    }, context());
    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-0");
    expect(degraded).toEqual([{
      reason: "verification_error",
      candidateCount: 3,
      successfulCandidates: 3,
      error: "verifier unavailable",
    }]);
  });

  test("keeps successful candidates when one candidate request fails", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const degraded: unknown[] = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls, new Set([0])),
      onDegraded: (event) => degraded.push(event),
    }, context());
    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-2");
    expect(calls).toHaveLength(3);
    expect(degraded).toEqual([]);
  });

  test("reports degradation when only one candidate succeeds", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const degraded: unknown[] = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls, new Set([0, 1])),
      onDegraded: (event) => degraded.push(event),
    }, context());
    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-2");
    expect(degraded).toEqual([{
      reason: "insufficient_candidates",
      candidateCount: 3,
      successfulCandidates: 1,
    }]);
  });

  test("returns an OMP error event when every candidate fails", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls, new Set([0, 1, 2])),
    }, context());
    const { events, result } = await collect(stream);
    expect(result.stopReason).toBe("error");
    expect((events as Array<{ type: string }>).map((event) => event.type)).toEqual(["error"]);
    expect(calls).toHaveLength(3);
  });

  test("returns an OMP aborted event when the request is cancelled", async () => {
    const controller = new AbortController();
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_model, _context, options = {}) => {
        const candidate = new AssistantMessageEventStream();
        const abort = () => candidate.push({
          type: "error",
          reason: "aborted",
          error: { ...message(0), content: [], stopReason: "aborted", errorMessage: "cancelled" },
        });
        if (options.signal?.aborted) queueMicrotask(abort);
        else options.signal?.addEventListener("abort", abort, { once: true });
        return candidate;
      },
    }, context(), { signal: controller.signal });
    controller.abort();
    const { events, result } = await collect(stream);
    expect(result.stopReason).toBe("aborted");
    expect((events as Array<{ type: string }>).map((event) => event.type)).toEqual(["error"]);
  });

  test("serializes current context and candidate metadata", () => {
    const current = context();
    const contextText = serializeContext(current);
    const candidateText = serializeAssistantMessage(message(2, true));
    expect(contextText).toContain("Fix the failing test and verify it.");
    expect(contextText).toContain("Message 1 (user)");
    expect(candidateText).toContain("response-2");
    expect(candidateText).toContain("candidate thinking 2");
    expect(candidateText).toContain("call-2");
  });

  test("preserves image bytes and complete tool contracts for verification", () => {
    const rich: Context = {
      systemPrompt: ["You are an OMP coding agent."],
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "Inspect this screenshot." },
          { type: "image", data: "aGVsbG8=", mimeType: "image/png", detail: "high" },
        ],
        timestamp: 1,
      }],
      tools: [{
        name: "bash",
        description: "run a shell command",
        parameters: {
          type: "object",
          properties: { command: { type: "string" } },
          required: ["command"],
        },
        strict: true,
        customWireName: "bash",
      }],
    };
    const serialized = serializeContext(rich);
    expect(serialized).toContain("Available tools");
    expect(serialized).toContain("customWireName");
    expect(serialized).toContain("required");
    expect(serialized).toContain("aGVsbG8=");
    expect(serialized).toContain("image/png");
  });
});

describe("OMP default model inheritance", () => {
  test("uses modelRoles.default and preserves its explicit thinking selector", async () => {
    const defaultModel = {
      ...model(),
      provider: "inferx",
      id: "deepseek-v4-flash-0731",
      api: "openai-responses",
      thinking: { mode: "effort", efforts: ["low", "high", "max"] },
    } as unknown as Model;
    const registry = {
      getApiKeyAndHeaders: async () => ({ ok: true, apiKey: "default-key", headers: { "X-Auth": "yes" } }),
      resolver: () => () => "default-key",
    };
    const client = await createDefaultVerifierClient({
      pi: {
        settings: {
          getModelRole: (role: string) => role === "default" ? "inferx/deepseek-v4-flash-0731:max" : undefined,
        },
      },
    } as never, {
      models: { resolve: (spec: string) => spec === "@default" ? defaultModel : undefined },
      modelRegistry: registry,
    } as never);
    expect(client.provider).toBe("inferx");
    expect(client.model).toBe("deepseek-v4-flash-0731");
    expect(client.effort).toBe("max");
    expect(client.headers).toEqual({ "X-Auth": "yes" });
    expect(client.apiKeyResolver).toBeDefined();
  });
});

describe("cache and fixed paper defaults", () => {
  test("cache identity includes model, effort, prompt and request lineage", () => {
    const base: CacheContext = {
      criterionId: "criterion",
      criterionName: "Criterion",
      criterionDescription: "description",
      problem: "problem",
      traceA: "a",
      traceB: "b",
      provider: "inferx",
      api: "openai-responses",
      model: "deepseek-v4-flash-0731",
      effort: "max",
      maxTokens: 32768,
      baseUrl: "https://example.test/v1",
      requestIdentity: "identity",
      groundTruthNote: CODING_AGENT_GROUND_TRUTH_NOTE,
      promptVersion: "pairwise-granularity20-v2",
    };
    const key = cacheKey("criterion", "task", 0, 1, 0, base);
    expect(cacheKey("criterion", "task", 0, 1, 0, { ...base, effort: "high" })).not.toBe(key);
    expect(cacheKey("criterion", "task", 0, 1, 0, { ...base, traceA: "changed" })).not.toBe(key);
  });

  test("saveCache merges entries durably", () => {
    const path = "/tmp/omp-verifier-" + crypto.randomUUID() + ".json";
    temporaryFiles.push(path);
    saveCache(path, { first: { score_A: 0.2, score_B: 0.8 } });
    saveCache(path, { second: { score_A: 0.7, score_B: 0.3 } });
    expect(Object.keys(loadCache(path)).sort()).toEqual(["first", "second"]);
  });

  test("uses the paper's small self-verification defaults", () => {
    expect(SELF_VERIFICATION_DEFAULTS).toEqual({ pivots: 1, nEvaluations: 2, seed: 0, maxWorkers: 8 });
    expect(CODING_AGENT_CRITERIA).toHaveLength(3);
  });
});

describe("documentation and plugin surface", () => {
  test("documents transparent configuration-driven use", () => {
    const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
    const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    expect(readme).toContain("omp plugin install");
    expect(readme).toContain("omp plugin config set omp-llm-verifier enabled true");
    expect(readme).toContain("omp plugin config set omp-llm-verifier enabled false");
    expect(readme).toContain("omp plugin config set omp-llm-verifier candidateCount 3");
    expect(manifest.omp.settings.enabled.default).toBe(false);
    expect(manifest.omp.settings.candidateCount).toEqual({
      type: "number",
      default: 3,
      min: 2,
      max: 8,
      step: 1,
      description: "每次普通请求生成的候选数量（2-8，默认 3）",
    });
    expect(readme).not.toContain("/verify");
    expect(readme).not.toContain("verifier_select");
  });

  test("normalizes plugin enablement and candidate-count settings", () => {
    expect(resolvePluginSettings({ enabled: true, candidateCount: 5 })).toEqual({
      enabled: true,
      candidateCount: 5,
    });
    expect(resolvePluginSettings({ enabled: false, candidateCount: 99 })).toEqual({
      enabled: false,
      candidateCount: AUTO_CANDIDATE_COUNT,
    });
    expect(normalizeCandidateCount("4")).toBe(4);
  });
});
