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
import {
  createAutomaticVerificationRuntime,
  createDefaultVerifierClient,
  ensureAutomaticVerification,
  resolvePluginSettings,
} from "../src/index.ts";
import { CODING_AGENT_CRITERIA, CODING_AGENT_GROUND_TRUTH_NOTE } from "../src/prompt.ts";
import { SELF_VERIFICATION_DEFAULTS } from "../src/run.ts";
import {
  isVerifierLogprobsUnsupportedError,
  VerifierClient,
  VerifierLogprobsUnsupportedError,
  type VerifierConfig,
} from "../src/client.ts";
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
  test("retries a transient 429 before parsing verifier logprobs", async () => {
    const originalFetch = globalThis.fetch;
    let attempts = 0;
    globalThis.fetch = (async () => {
      attempts += 1;
      if (attempts === 1) return new Response("busy", { status: 429, headers: { "retry-after": "0" } });
      return new Response(JSON.stringify({
        choices: [{
          message: { content: "A" },
          finish_reason: "stop",
          logprobs: { content: [{ token: "A", logprob: -0.1, top_logprobs: [{ token: "A", logprob: -0.1 }] }] },
        }],
      }), { status: 200, headers: { "content-type": "application/json" } });
    }) as unknown as typeof fetch;
    try {
      const reply = await new VerifierClient(clientConfig()).scoreReply("Return A.");
      expect(reply.positionLogprobs).toHaveLength(1);
      expect(attempts).toBe(2);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("does not retry a non-transient verifier error", async () => {
    const originalFetch = globalThis.fetch;
    let attempts = 0;
    globalThis.fetch = (async () => {
      attempts += 1;
      return new Response("bad request", { status: 400 });
    }) as unknown as typeof fetch;
    try {
      await expect(new VerifierClient(clientConfig()).scoreReply("Return A.")).rejects.toThrow("Verifier API 400");
      expect(attempts).toBe(1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("treats a persistente no-logprobs response as unsupported after retries", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => new Response(JSON.stringify({
      choices: [{
        message: { content: "A" },
        finish_reason: "stop",
        logprobs: null,
      }],
    }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })) as unknown as typeof fetch;
    try {
      let error: unknown;
      try {
        await new VerifierClient(clientConfig({ effort: "off" })).scoreReply("Return A.");
      } catch (caught) {
        error = caught;
      }
      expect(isVerifierLogprobsUnsupportedError(error)).toBe(true);
      expect((error as Error).message).toContain("token logprobs");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

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
    // The wrapper's credentials are threaded onto candidate calls; OMP's
    // session/cache identity is preserved (asserted in the affinity test below).
    const resolver = calls[0]?.options.apiKey;
    expect(typeof resolver).toBe("function");
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

  test("replays a tool-use turn directly without expanding or verifying", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const verifier = new RankingVerifier();
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_m, _context, _o = {}) => {
        calls.push({ context: _context, options: _o });
        const out = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = message(0, true); // stopReason toolUse
          out.push({ type: "start", partial: result });
          out.push({ type: "done", reason: "toolUse", message: result });
        });
        return out;
      },
    }, context());
    const { events, result } = await collect(stream);
    // terminal gating: an intermediate tool-use turn is the agent's own next
    // action, so the wrapper replays it and never fans out or verifies.
    expect(calls).toHaveLength(1);
    expect(verifier.calls).toBe(0);
    expect(result.stopReason).toBe("toolUse");
    expect(result.responseId).toBe("response-0");
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
  });

  test("skips the verifier when an exact-action majority agrees", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const verifier = new RankingVerifier();
    let next = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_m, _context, _o = {}) => {
        calls.push({ context: _context, options: _o });
        const index = next++;
        const out = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const text = index === 2 ? "different plan B" : "same plan A";
          const result = {
            ...message(index),
            content: [{ type: "text" as const, text }],
            stopReason: "stop" as const,
          };
          out.push({ type: "start", partial: result });
          out.push({ type: "done", reason: "stop", message: result });
        });
        return out;
      },
    }, context());
    const { result } = await collect(stream);
    // TurboAgent-style shortcut: 2 of 3 candidates share the same normalized
    // action (> N/2), so selecting among them cannot change the answer and the
    // whole verifier tournament is skipped.
    expect(calls).toHaveLength(3);
    expect(verifier.calls).toBe(0);
    expect(result.responseId).toBe("response-0");
    expect(result.content).toEqual([{ type: "text", text: "same plan A" }]);
  });

  test("preserves the OMP cache/session identity on candidate calls", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const sessionState = new Map<string, { close(): void }>([
      ["opencode-go", { close: () => undefined }],
    ]);
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
    }, context(), {
      sessionId: "session-cache-id",
      promptCacheKey: "prompt-cache-key",
      providerSessionState: sessionState,
    });
    await collect(stream);
    expect(calls.length).toBeGreaterThan(0);
    // Only turn chaining must be off; every candidate keeps the OMP default-call
    // identity so the full-context prefix hits the provider's cache instead of
    // paying a fresh uncached write per candidate.
    for (const call of calls) {
      expect(call.options.statefulResponses).toBe(false);
      expect(call.options.sessionId).toBe("session-cache-id");
      expect(call.options.promptCacheKey).toBe("prompt-cache-key");
      expect(call.options.providerSessionState).toBe(sessionState);
    }
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
    // reference on_error="tie": a failed verifier call scores 0.5/0.5
    // instead of aborting the tournament or degrading the response.
    expect(degraded).toEqual([]);
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

  test("serializes the task and a compact candidate trace", () => {
    const current = context();
    const contextText = serializeContext(current);
    const candidateText = serializeAssistantMessage(message(2, true));
    expect(contextText).toContain("Fix the failing test and verify it.");
   expect(contextText).not.toContain("Message 1 (user)");
   expect(candidateText).toContain("candidate text 2");
   expect(candidateText).toContain("[proposed tool call] bash");
   expect(candidateText).not.toContain("response-2");
    // Reasoning is not part of the reference trace; only visible text and
    // tool calls are scored, so thinking must not leak into the prompt.
    expect(candidateText).not.toContain("candidate thinking 2");
 });

  test("reduces user context to the task and counts images", () => {
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
    expect(serialized).toContain("Inspect this screenshot.");
    expect(serialized).toContain("1 image(s)");
    expect(serialized).not.toContain("Available tools");
    expect(serialized).not.toContain("aGVsbG8=");
  });

  test("bounds a many-block candidate trace to the total budget", () => {
    const hugeArgs = { command: "build ".repeat(20_000) }; // ~130k chars
    const giant = "x".repeat(100_000);
    const content: AssistantMessage["content"] = [];
    for (let i = 0; i < 6; i++) {
      content.push({ type: "text", text: giant });
      content.push({ type: "toolCall", id: "c" + i, name: "bash", arguments: hugeArgs });
    }
    const candidate = {
      role: "assistant",
      content,
      api: "openai-completions",
      provider: "opencode-go",
      model: "deepseek-v4-flash-0731",
      responseId: "response-bloat",
      usage: usage(),
      stopReason: "stop",
      timestamp: 1,
    } as AssistantMessage;
    const serialized = serializeAssistantMessage(candidate);
    // Per-block (2k) and per-trace (8k, separators included) caps keep one
    // oversized candidate from blowing up every pairwise verifier prompt.
    expect(serialized.length).toBeLessThanOrEqual(8_000);
    expect(serialized).toContain("[truncated]");
  });

  test("keeps many near-cap text blocks within the 8k trace budget", () => {
    const cap = 1950; // just under the 2000 per-block cap
    const slabs: Array<string> = [];
    for (let i = 0; i < 20; i++) slabs.push("block-" + i + " " + "y".repeat(cap));
    const content: AssistantMessage["content"] = slabs.map((text) => ({ type: "text", text }));
    const candidate = {
      role: "assistant",
      content,
      api: "openai-completions",
      provider: "opencode-go",
      model: "deepseek-v4-flash-0731",
      responseId: "response-many",
      usage: usage(),
      stopReason: "stop",
      timestamp: 1,
    } as AssistantMessage;
    // 20 blocks at ~1950 chars each would join to ~39k without the total cap;
    // separators (every join) must count against the 8k budget.
    expect(serializeAssistantMessage(candidate).length).toBeLessThanOrEqual(8_000);
  });

  test("keeps the serialized problem within the 16k input budget", () => {
    const hugeArgs = { command: "build ".repeat(20_000) }; // ~130k chars
    const giant = "x".repeat(100_000);
    const messages: Context["messages"] = [
      {
        role: "user",
        content: "Overhaul " + "the project ".repeat(3_000) + " end-to-end.",
        timestamp: 1,
      },
    ];
    for (let i = 0; i < 8; i++) {
      messages.push({
        role: "assistant",
        content: [
          { type: "thinking", thinking: "let me reason " + "y".repeat(50_000) },
          { type: "text", text: giant },
          { type: "toolCall", id: "c" + i, name: "bash", arguments: hugeArgs },
        ],
        api: "openai-completions",
        provider: "opencode-go",
        model: "deepseek-v4-flash-0731",
        usage: usage(),
        stopReason: "toolUse",
        timestamp: 2 + i,
      } as unknown as AssistantMessage);
      messages.push({
        role: "toolResult",
        toolCallId: "c" + i,
        toolName: "bash",
        isError: i % 2 === 1,
        content: [{ type: "text", text: "result ".repeat(50_000) }],
        timestamp: 3 + i,
      });
    }
    const serialized = serializeContext({ ...context(), messages });
    // Task + trajectory evidence share a hard 16k char budget, so no pair
    // prompt can exceed it even for a very long session.
    expect(serialized.length).toBeLessThanOrEqual(16_000);
    expect(serialized).toContain("... [task truncated]");
  });
});

describe("OMP default model inheritance", () => {
  test("caches an unsupported-model capability warning for startup and later turns", async () => {
    let activeModel: Model = model();
    const configuredModel = activeModel;
    const wrappers = new Map<string, Model>();
    const fakePi = {
      pi: { settings: { getModelRole: () => "opencode-go/deepseek-v4-flash-0731:high" } },
      registerProvider: (provider: string) => {
        wrappers.set(provider + "/default", { ...configuredModel, provider, id: "default" } as unknown as Model);
      },
      getThinkingLevel: () => "high",
      setThinkingLevel: () => undefined,
      setModel: async (next: Model) => { activeModel = next; return true; },
    } as never;
    const ctx = {
      get model() { return activeModel; },
      models: { resolve: (spec: string) => spec === "@default" ? configuredModel : wrappers.get(spec) },
      modelRegistry: {
        getApiKey: async () => "test-key",
        resolver: () => () => "test-key",
        getProviderHeaders: () => undefined,
        refreshRuntimeProviders: async () => undefined,
      },
      sessionManager: { getSessionId: () => "unsupported-session" },
    } as never;
    const runtime = createAutomaticVerificationRuntime({
      probeVerifier: async () => {
        throw new VerifierLogprobsUnsupportedError("probe returned no token logprobs");
      },
    });
    await expect(ensureAutomaticVerification(fakePi, ctx, runtime)).rejects.toThrow("opencode-go/deepseek-v4-flash-0731");
    expect(runtime.capabilityErrors.size).toBe(1);
    await expect(ensureAutomaticVerification(fakePi, ctx, runtime)).rejects.toThrow("token logprobs");
  });

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

  test("rebinds the transparent wrapper after the default model changes", async () => {
    const modelA = model();
    const modelB = {
      ...model(),
      provider: "inferx",
      id: "deepseek-v4-flash-0731",
      name: "DeepSeek V4 Flash 0731",
    } as unknown as Model;
    let configuredModel = modelA;
    let activeModel: Model = modelA;
    let selector = "opencode-go/deepseek-v4-flash-0731:max";
    const wrappers = new Map<string, Model>();
    const registrations: string[] = [];
    const fakePi = {
      pi: { settings: { getModelRole: () => selector } },
      registerProvider: (provider: string) => {
        registrations.push(provider);
        const wrapper = { ...configuredModel, provider, id: "default", name: "wrapped" } as unknown as Model;
        wrappers.set(provider + "/default", wrapper);
      },
      getThinkingLevel: () => "high",
      setThinkingLevel: () => undefined,
      setModel: async (next: Model) => {
        activeModel = next;
        return true;
      },
    } as never;
    const registry = {
      getApiKey: async () => "test-key",
      resolver: () => () => "test-key",
      getProviderHeaders: () => undefined,
      refreshRuntimeProviders: async () => undefined,
    };
    const ctx = {
      get model() { return activeModel; },
      models: {
        resolve: (spec: string) => spec === "@default" ? configuredModel : wrappers.get(spec),
      },
      modelRegistry: registry,
      sessionManager: { getSessionId: () => "session-1" },
      ui: { notify: () => undefined },
    } as never;
    const runtime = createAutomaticVerificationRuntime();

    const first = await ensureAutomaticVerification(fakePi, ctx, runtime);
    expect(first.originalModel).toBe(modelA);
    expect(activeModel.provider).toBe(first.providerName);
    expect(registrations).toHaveLength(1);
    await ensureAutomaticVerification(fakePi, ctx, runtime);
    expect(registrations).toHaveLength(1);

    configuredModel = modelB;
    activeModel = modelB;
    selector = "inferx/deepseek-v4-flash-0731:max";
    const second = await ensureAutomaticVerification(fakePi, ctx, runtime);
    expect(second.originalModel).toBe(modelB);
    expect(second.providerName).not.toBe(first.providerName);
    expect(activeModel.provider).toBe(second.providerName);
    expect(registrations).toHaveLength(2);
  });

  test("does not let a stale rebind overwrite a newer active model", async () => {
    const modelA = model();
    const modelB = { ...model(), provider: "inferx", id: "deepseek-v4-flash-0731" } as unknown as Model;
    let configuredModel = modelA;
    let activeModel: Model = modelA;
    const wrappers = new Map<string, Model>();
    const pendingRegistrations: Array<{ provider: string; resolve: () => void }> = [];
    const fakePi = {
      pi: { settings: { getModelRole: () => "default" } },
      registerProvider: (provider: string) => {
        const wrapper = { ...configuredModel, provider, id: "default" } as unknown as Model;
        wrappers.set(provider + "/default", wrapper);
      },
      getThinkingLevel: () => "high",
      setThinkingLevel: () => undefined,
      setModel: async (next: Model) => { activeModel = next; return true; },
    } as never;
    const registry = {
      getApiKey: async () => "test-key",
      resolver: () => () => "test-key",
      getProviderHeaders: () => undefined,
      refreshRuntimeProviders: async () => {
        await new Promise<void>((resolve) => pendingRegistrations.push({ provider: "pending", resolve }));
      },
    };
    const ctx = {
      get model() { return activeModel; },
      models: { resolve: (spec: string) => spec === "@default" ? configuredModel : wrappers.get(spec) },
      modelRegistry: registry,
      sessionManager: { getSessionId: () => "session-2" },
    } as never;
    const runtime = createAutomaticVerificationRuntime();
    const first = ensureAutomaticVerification(fakePi, ctx, runtime);
    await Promise.resolve();
    configuredModel = modelB;
    activeModel = modelB;
    const second = ensureAutomaticVerification(fakePi, ctx, runtime);
    while (pendingRegistrations.length > 0) pendingRegistrations.shift()?.resolve();
    const [firstBinding, secondBinding] = await Promise.all([first, second]);
    expect(secondBinding.originalModel).toBe(modelB);
    expect(firstBinding.originalModel).toBe(modelB);
    expect(activeModel.provider).toBe(secondBinding.providerName);
  });

  test("keeps the active wrapper source while the persisted default is stale", async () => {
    const modelA = model();
    const modelB = { ...model(), provider: "inferx", id: "deepseek-v4-flash-0731" } as unknown as Model;
    let configuredModel = modelA;
    let activeModel: Model = modelA;
    const wrappers = new Map<string, Model>();
    const fakePi = {
      pi: { settings: { getModelRole: () => "opencode-go/deepseek-v4-flash:max" } },
      registerProvider: (provider: string) => {
        const wrapper = { ...configuredModel, provider, id: "default" } as unknown as Model;
        wrappers.set(provider + "/default", wrapper);
      },
      getThinkingLevel: () => "high",
      setThinkingLevel: () => undefined,
      setModel: async (next: Model) => { activeModel = next; return true; },
    } as never;
    const registry = {
      getApiKey: async () => "test-key",
      resolver: () => () => "test-key",
      getProviderHeaders: () => undefined,
      refreshRuntimeProviders: async () => undefined,
    };
    const ctx = {
      get model() { return activeModel; },
      models: { resolve: (spec: string) => spec === "@default" ? configuredModel : wrappers.get(spec) },
      modelRegistry: registry,
      sessionManager: { getSessionId: () => "session-3" },
    } as never;
    const runtime = createAutomaticVerificationRuntime();

    const first = await ensureAutomaticVerification(fakePi, ctx, runtime);
    configuredModel = modelB;
    activeModel = modelB;
    const second = await ensureAutomaticVerification(fakePi, ctx, runtime);
    expect(second.originalModel).toBe(modelB);

    configuredModel = modelA;
    activeModel = second.wrapperModel;
    const third = await ensureAutomaticVerification(fakePi, ctx, runtime);
    expect(third.originalModel).toBe(modelB);
    expect(third.providerName).toBe(second.providerName);
    expect(activeModel.provider).toBe(second.providerName);
    expect(first.providerName).not.toBe(second.providerName);
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
    expect(readme).toContain("README.zh-CN.md");
    expect(readFileSync(new URL("../README.zh-CN.md", import.meta.url), "utf8")).toContain("## OMP 安装与使用");
    expect(readme).toContain("omp plugin config set omp-llm-verifier enabled true");
    expect(readme).toContain("omp plugin config set omp-llm-verifier enabled false");
    expect(readme).toContain("omp plugin disable omp-llm-verifier");
    expect(readme).toContain("omp plugin enable omp-llm-verifier");
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
