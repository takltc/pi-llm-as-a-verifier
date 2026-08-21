import { describe, expect, test } from "bun:test";
import { afterEach } from "bun:test";
import { readFileSync } from "node:fs";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import type {
  AssistantMessage,
  Context,
  ImageContent,
  Model,
  SimpleStreamOptions,
} from "@oh-my-pi/pi-ai";
import { cacheKey, loadCache, saveCache, type CacheContext } from "../src/cache.ts";
import {
  AUTO_CANDIDATE_COUNT,
  AUTO_SELECTION_DEFAULTS,
  createAutoVerifierStream,
  normalizeCandidateCount,
  normalizeEvaluations,
  normalizePivots,
  serializeAssistantMessage,
  serializeContext,
} from "../src/auto.ts";
import {
  createAutomaticVerificationRuntime,
  createDefaultVerifierClient,
  ensureAutomaticVerification,
  resolvePluginSettings,
} from "../src/index.ts";
import {
  CODING_AGENT_ACTION_CRITERIA,
  CODING_AGENT_CRITERIA,
  CODING_AGENT_GROUND_TRUTH_NOTE,
} from "../src/prompt.ts";
import { SELF_VERIFICATION_DEFAULTS } from "../src/run.ts";
import { select } from "../src/select.ts";
import {
  CAPABILITY_PROBE_TIMEOUT_MS,
  createVerifierClient,
  isVerifierLogprobsUnsupportedError,
  VerifierClient,
  VerifierLogprobsUnsupportedError,
  type VerifierConfig,
} from "../src/client.ts";
import { extractScorePair, type VerifierReply } from "../src/scale.ts";
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
      const result = message(index);
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
  readonly prompts: string[] = [];
  readonly imageBatches: Array<readonly ImageContent[]> = [];

  constructor(
    private readonly fail = false,
    supportsImages = false,
    private readonly scoreMode: "logprobs" | "text_fallback" = "logprobs",
  ) {
    super(clientConfig({ supportsImages }));
  }

  override async scoreReply(
    prompt: string,
    opts: Parameters<VerifierClient["scoreReply"]>[1] = {},
  ): Promise<VerifierReply> {
    this.calls += 1;
    this.prompts.push(prompt);
    this.imageBatches.push(opts.images ?? []);
    if (this.fail) throw new Error("verifier unavailable");
    const a = /candidate text (\d+)/.exec(prompt)?.[1];
    const b = /candidate text (\d+)/.exec(prompt.slice(prompt.indexOf("Trajectory B")))?.[1];
    const aWins = a === "2" && b !== "2";
    const bWins = b === "2" && a !== "2";
    const scoreA = aWins ? "A" : bWins ? "T" : "A";
    const scoreB = bWins ? "A" : aWins ? "T" : "A";
    const text = "<score_A> " + scoreA + " </score_A>\n<score_B> " + scoreB + " </score_B>";
    if (this.scoreMode === "text_fallback") return { text };
    return {
      text,
      tokens: ["<score_A>", " " + scoreA, " </score_A>\n<score_B>", " " + scoreB, " </score_B>"],
      positionLogprobs: [
        [["<score_A>", 0]],
        [[scoreA, 0]],
        [[" </score_A>\n<score_B>", 0]],
        [[scoreB, 0]],
        [[" </score_B>", 0]],
      ],
    };
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

describe("automatic request/action-level provider", () => {
  test("keeps the startup capability probe inside OMP's handler deadline", async () => {
    class ProbeBudgetVerifier extends VerifierClient {
      timeoutMs: number | undefined;

      override async scoreReply(
        _prompt: string,
        opts: Parameters<VerifierClient["scoreReply"]>[1] = {},
      ): Promise<VerifierReply> {
        this.timeoutMs = opts.timeoutMs;
        return { text: "A" };
      }
    }

    const verifier = new ProbeBudgetVerifier(clientConfig());
    await verifier.probeLogprobs();
    expect(verifier.timeoutMs).toBe(CAPABILITY_PROBE_TIMEOUT_MS);
    expect(CAPABILITY_PROBE_TIMEOUT_MS).toBeLessThan(30_000);
  });

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

  test("treats a persistent no-logprobs response as unsupported after retries", async () => {
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

  test("classifies a provider logprobs rejection as unsupported without retrying", async () => {
    const originalFetch = globalThis.fetch;
    let attempts = 0;
    globalThis.fetch = (async () => {
      attempts += 1;
      // Kimi for Coding's exact rejection shape for logprobs requests.
      return new Response(
        JSON.stringify({ error: { message: "Your request body contains invalid value for param logprobs", type: "invalid_request_error" } }),
        { status: 400, headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;
    try {
      let error: unknown;
      try {
        await new VerifierClient(clientConfig()).scoreReply("Return A.");
      } catch (caught) {
        error = caught;
      }
      expect(isVerifierLogprobsUnsupportedError(error)).toBe(true);
      expect((error as VerifierLogprobsUnsupportedError).retryable).toBe(false);
      expect((error as Error).message).toContain("provider rejects token logprobs");
      expect(attempts).toBe(1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("pins the reference verifier request shape: temperature 1, no thinking for generic models", async () => {
    const originalFetch = globalThis.fetch;
    const bodies: Array<{ body: Record<string, unknown> }> = [];
    const okReply = () => new Response(JSON.stringify({
      choices: [{ message: { content: "A" }, finish_reason: "stop",
        logprobs: { content: [{ token: "A", logprob: -0.1, top_logprobs: [{ token: "A", logprob: -0.1 }] }] } }],
    }), { status: 200, headers: { "content-type": "application/json" } });
    globalThis.fetch = (async (_url: unknown, init: unknown) => {
      bodies.push({ body: JSON.parse(String((init as { body?: string }).body)) });
      return okReply();
    }) as unknown as typeof fetch;
    try {
      // createVerifierClient mirrors the reference defaults per provider:
      // the generic OpenAI-compatible path gets 4096 output + no reasoning,
      // the DeepSeek path gets 32768 + reasoning enabled.
      const genericModel = { ...model(), provider: "inferx", id: "deepseek-v4-flash-0731" } as unknown as Model;
      const genericClient = await createVerifierClient({
        model: genericModel,
        models: { resolve: () => genericModel },
        modelRegistry: {
          resolver: () => () => "test-key",
          getApiKeyAndHeaders: async () => ({ ok: true as const, apiKey: "test-key", headers: {} }),
        },
        sessionId: "shape-session",
      } as never);
      expect(genericClient.effort).toBe("off");
      expect(genericClient.maxTokens).toBe(4096);
      await genericClient.scoreReply("Return A.");
      const generic = bodies[0]!.body;
      expect(generic.temperature).toBe(1);
      expect(generic.logprobs).toBe(true);
      expect(generic.max_tokens).toBe(4096);
      expect(generic.reasoning_effort).toBeUndefined();
      expect(generic.thinking).toBeUndefined();

      const deepseekModel = {
        ...model(),
        provider: "deepseek",
        id: "deepseek-v4-flash",
        thinking: { mode: "effort", efforts: ["low", "high", "max"] },
      } as unknown as Model;
      const deepseekClient = await createVerifierClient({
        model: deepseekModel,
        models: { resolve: () => deepseekModel },
        modelRegistry: {
          resolver: () => () => "test-key",
          getApiKeyAndHeaders: async () => ({ ok: true as const, apiKey: "test-key", headers: {} }),
        },
        sessionId: "shape-session",
      } as never);
      expect(deepseekClient.effort).toBe("high");
      expect(deepseekClient.maxTokens).toBe(32768);
      await deepseekClient.scoreReply("Return A.");
      const deepseek = bodies[1]!.body;
      expect(deepseek.temperature).toBe(1);
      expect(deepseek.max_tokens).toBe(32768);
      expect(deepseek.reasoning_effort).toBe("high");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("encodes task images for Chat Completions and Responses verifier requests", async () => {
    const originalFetch = globalThis.fetch;
    const bodies: Record<string, unknown>[] = [];
    globalThis.fetch = (async (url: unknown, init: unknown) => {
      bodies.push(JSON.parse(String((init as { body?: string }).body)));
      const response = String(url).endsWith("/responses")
        ? {
            status: "completed",
            output: [{
              type: "message",
              content: [{
                type: "output_text",
                text: "A",
                logprobs: [{
                  token: "A",
                  logprob: -0.1,
                  top_logprobs: [{ token: "A", logprob: -0.1 }],
                }],
              }],
            }],
          }
        : {
            choices: [{
              message: { content: "A" },
              finish_reason: "stop",
              logprobs: {
                content: [{
                  token: "A",
                  logprob: -0.1,
                  top_logprobs: [{ token: "A", logprob: -0.1 }],
                }],
              },
            }],
          };
      return new Response(JSON.stringify(response), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;
    const image: ImageContent = {
      type: "image",
      data: "aGVsbG8=",
      mimeType: "image/png",
      detail: "high",
    };
    try {
      const chat = new VerifierClient(clientConfig({ supportsImages: true, effort: "off" }));
      await chat.scoreReply("Inspect the image.", { images: [image] });
      const chatMessage = (bodies[0]!.messages as Array<{ content: unknown }>)[0]!;
      expect(chatMessage.content).toEqual([
        { type: "text", text: "Inspect the image." },
        {
          type: "image_url",
          image_url: { url: "data:image/png;base64,aGVsbG8=", detail: "high" },
        },
      ]);

      const responses = new VerifierClient(clientConfig({
        api: "openai-responses",
        supportsImages: true,
        effort: "off",
      }));
      await responses.scoreReply("Inspect the image.", { images: [image] });
      const responseInput = (bodies[1]!.input as Array<{ content: unknown }>)[0]!;
      expect(responseInput.content).toEqual([
        { type: "input_text", text: "Inspect the image." },
        {
          type: "input_image",
          image_url: "data:image/png;base64,aGVsbG8=",
          detail: "high",
        },
      ]);

      const textOnly = new VerifierClient(clientConfig({ effort: "off" }));
      await expect(textOnly.scoreReply("Inspect.", { images: [image] })).rejects.toThrow(
        "cannot inspect the context images",
      );
      expect(bodies).toHaveLength(2);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("parses only Responses output_text logprobs", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => new Response(JSON.stringify({
      status: "completed",
      output: [
        {
          type: "message",
          content: [{
            type: "output_text",
            text: "<score_A> A </score_A>\n<score_B> T </score_B>",
            logprobs: [
              { token: "<score_A>", logprob: 0, top_logprobs: [{ token: "<score_A>", logprob: 0 }] },
              { token: " A", logprob: 0, top_logprobs: [{ token: "A", logprob: 0 }] },
              { token: " </score_A>\n<score_B>", logprob: 0, top_logprobs: [{ token: " </score_A>\n<score_B>", logprob: 0 }] },
              { token: " T", logprob: 0, top_logprobs: [{ token: "T", logprob: 0 }] },
              { token: " </score_B>", logprob: 0, top_logprobs: [{ token: " </score_B>", logprob: 0 }] },
            ],
          }],
        },
        {
          type: "reasoning",
          content: [{
            type: "reasoning_text",
            text: "<score_A> T </score_A>\n<score_B> A </score_B>",
            logprobs: [
              { token: "<score_A>", logprob: 0, top_logprobs: [{ token: "<score_A>", logprob: 0 }] },
              { token: " T", logprob: 0, top_logprobs: [{ token: "T", logprob: 0 }] },
              { token: " </score_A>\n<score_B>", logprob: 0, top_logprobs: [{ token: " </score_A>\n<score_B>", logprob: 0 }] },
              { token: " A", logprob: 0, top_logprobs: [{ token: "A", logprob: 0 }] },
              { token: " </score_B>", logprob: 0, top_logprobs: [{ token: " </score_B>", logprob: 0 }] },
            ],
          }],
        },
      ],
    }), { status: 200, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
    try {
      const reply = await new VerifierClient(clientConfig({ api: "openai-responses" }))
        .scoreReply("Compare.");
      const pair = extractScorePair(reply);
      expect(reply.text).toBe("<score_A> A </score_A>\n<score_B> T </score_B>");
      expect(pair.scoreA).toBe(1);
      expect(pair.scoreB).toBe(0);
      expect(pair.sourceA).toBe("logprobs");
      expect(pair.sourceB).toBe("logprobs");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test("generates the default three candidates and replays the verified winner", async () => {
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
    expect(verifier.calls).toBeLessThanOrEqual(6);
    expect(calls.every((call) => call.options.statefulResponses === false)).toBe(true);
    expect(calls.every((call) => call.options.temperature === 1)).toBe(true);
    expect(calls.every((call) => call.options.anthropicCacheRefresh === false)).toBe(true);
    expect(calls.every((call) => call.options.anthropicCacheRefreshRequest === false)).toBe(true);
    // The wrapper's credentials are threaded onto candidate calls; OMP's
    // session/cache identity is preserved (asserted in the affinity test below).
    const resolver = calls[0]?.options.apiKey;
    expect(typeof resolver).toBe("function");
    expect(result.responseId).toBe("response-2");
    expect(result.stopReason).toBe("stop");
    expect(result.content.some((block) => block.type === "toolCall")).toBe(false);
    expect((events as Array<{ type: string }>).map((event) => event.type)).toEqual([
      "start",
      "thinking_start",
      "thinking_delta",
      "thinking_end",
      "text_start",
      "text_delta",
      "text_end",
      "done",
    ]);
    const replayEvents = events as Array<{ type: string; partial?: AssistantMessage }>;
    const textStart = replayEvents.find((event) => event.type === "text_start");
    const textDelta = replayEvents.find((event) => event.type === "text_delta");
    expect(textStart?.partial?.content[1]).toEqual({ type: "text", text: "" });
    expect(textDelta?.partial?.content[1]).toMatchObject({ type: "text", text: "candidate text 2" });
  });

  test("starts all request/action candidates concurrently", async () => {
    let started = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: () => {
        const index = started++;
        const candidate = new AssistantMessageEventStream();
        void gate.then(() => {
          const result = message(index);
          candidate.push({ type: "start", partial: result });
          candidate.push({ type: "done", reason: "stop", message: result });
        });
        return candidate;
      },
    }, context());

    const pending = collect(stream);
    await Bun.sleep(0);
    expect(started).toBe(3);
    release();
    const { result } = await pending;
    expect(result.responseId).toBe("response-2");
  });

  test("rejects provider-native execution before candidate fan-out", async () => {
    let candidateCalls = 0;
    const decisions: unknown[] = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: () => {
        candidateCalls += 1;
        return new AssistantMessageEventStream();
      },
      onDecision: (decision) => decisions.push(decision),
    }, context(), {
      execHandlers: {} as NonNullable<SimpleStreamOptions["execHandlers"]>,
    });
    const { result } = await collect(stream);

    expect(candidateCalls).toBe(0);
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toContain("declarative tool calls");
    expect(decisions[0]).toMatchObject({
      path: "error",
      granularity: "request_action",
      successfulCandidates: 0,
    });
  });

  test("uses the configured candidate count through the maximum of eight", async () => {
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
      { candidateCount: 8 },
    );
    await collect(stream);
    expect(calls).toHaveLength(8);
  });

  test("uses the configured verifier repeats K through PPT", async () => {
    // Paper §4.2 quality/cost axis: raising K raises the verifier call count by
    // exactly the same factor from TurboAgent's online K=1 default.
    const defaultCalls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const defaultVerifier = new RankingVerifier();
    await collect(createAutoVerifierStream({
      originalModel: model(),
      verifierClient: defaultVerifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(defaultCalls),
    }, context(), {}, {}));
    // This deterministic ranking produces 5 PPT comparisons x C=1 x K=1.
    expect(defaultVerifier.calls).toBe(5);

    const qualityCalls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const qualityVerifier = new RankingVerifier();
    await collect(createAutoVerifierStream({
      originalModel: model(),
      verifierClient: qualityVerifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(qualityCalls),
    }, context(), {}, { candidateCount: 3, nEvaluations: 3 }));
    expect(qualityVerifier.calls).toBe(15);
  });

  test("isolates mutable message content across candidate contexts", async () => {
    const rich: Context = {
      ...context(),
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Original task evidence." }],
          timestamp: 1,
        },
        message(99, true),
      ],
    };
    const observed: Array<{ text: string; command: string }> = [];
    let nextIndex = 0;

    await collect(createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_model, candidateContext) => {
        const index = nextIndex++;
        const userContent = candidateContext.messages[0]!.content as Array<{
          type: string;
          text?: string;
        }>;
        const priorAssistant = candidateContext.messages[1] as AssistantMessage;
        const toolCall = priorAssistant.content.find((block) => block.type === "toolCall");
        if (!toolCall) throw new Error("Expected prior tool call");
        observed.push({
          text: userContent[0]?.text ?? "",
          command: String(toolCall.arguments.command),
        });

        userContent[0]!.text = "mutated task " + index;
        toolCall.arguments.command = "mutated command " + index;

        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = message(index);
          candidate.push({ type: "start", partial: result });
          candidate.push({ type: "done", reason: "stop", message: result });
        });
        return candidate;
      },
    }, rich));

    expect(observed).toEqual(Array.from({ length: 3 }, () => ({
      text: "Original task evidence.",
      command: "printf candidate-99",
    })));
    const sourceUserContent = rich.messages[0]!.content as Array<{ text?: string }>;
    const sourceToolCall = (rich.messages[1] as AssistantMessage).content.find(
      (block) => block.type === "toolCall",
    );
    expect(sourceUserContent[0]?.text).toBe("Original task evidence.");
    expect(sourceToolCall?.type === "toolCall" && sourceToolCall.arguments.command)
      .toBe("printf candidate-99");
  });

  test("fans out and verifies every tool-use action before replay", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const verifier = new RankingVerifier();
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_m, _context, _o = {}) => {
        const index = calls.length;
        calls.push({ context: _context, options: _o });
        const out = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = message(index, true);
          out.push({ type: "start", partial: result });
          out.push({ type: "done", reason: "toolUse", message: result });
        });
        return out;
      },
    }, context());
    const { events, result } = await collect(stream);
    expect(calls).toHaveLength(3);
    expect(verifier.calls).toBe(5);
    expect(result.stopReason).toBe("toolUse");
    expect(result.responseId).toBe("response-2");
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

  test("compares terminal and tool-use actions in one request/action-level PPT", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const degraded: unknown[] = [];
    const decisions: unknown[] = [];
    const verifier = new RankingVerifier();
    let nextIndex = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_model, candidateContext, options = {}) => {
        const index = nextIndex++;
        calls.push({ context: candidateContext, options });
        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = message(index, index > 0);
          candidate.push({ type: "start", partial: result });
          candidate.push({
            type: "done",
            reason: result.stopReason === "toolUse" ? "toolUse" : "stop",
            message: result,
          });
        });
        return candidate;
      },
      onDegraded: (event) => degraded.push(event),
      onDecision: (decision) => decisions.push(decision),
    }, context());
    const { result } = await collect(stream);

    expect(calls).toHaveLength(3);
    expect(verifier.calls).toBe(5);
    expect(result.responseId).toBe("response-2");
    expect(result.stopReason).toBe("toolUse");
    expect(degraded).toEqual([]);
    expect(decisions[0]).toMatchObject({
      path: "verifier",
      granularity: "request_action",
      winnerIndex: 2,
      winnerStopReason: "toolUse",
      successfulCandidates: 3,
      terminalCandidates: 1,
      toolUseCandidates: 2,
      nonterminalCandidates: 2,
    });
  });

  test("uses exact-action majority before PPT and ignores tool-call ids", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const verifier = new RankingVerifier();
    const decisions: unknown[] = [];
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
          const command = index === 2 ? "printf plan-b" : "printf plan-a";
          const result = {
            ...message(index),
            content: [
              { type: "text" as const, text: index === 2 ? "different plan B" : "same plan A" },
              {
                type: "toolCall" as const,
                id: "provider-call-" + index,
                name: "bash",
                arguments: { command },
              },
            ],
            stopReason: "toolUse" as const,
          };
          out.push({ type: "start", partial: result });
          out.push({ type: "done", reason: "toolUse", message: result });
        });
        return out;
      },
      onDecision: (decision) => decisions.push(decision),
    }, context());
    const { result } = await collect(stream);
    expect(calls).toHaveLength(3);
    expect(verifier.calls).toBe(0);
    expect(result.responseId).toBe("response-0");
    expect(result.stopReason).toBe("toolUse");
    expect(decisions).toHaveLength(1);
    expect(decisions[0]).toMatchObject({
      path: "majority",
      granularity: "request_action",
      winnerIndex: 0,
      winnerStopReason: "toolUse",
      nComparisons: 0,
      successfulCandidates: 2,
      discardedCandidates: 1,
      scores: [1, 1],
    });
  });

  test("requires full untruncated action equality for majority", async () => {
    const sharedPrefix = "shared ".repeat(1_500);
    const verifier = new RankingVerifier();
    const decisions: unknown[] = [];
    let next = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: () => {
        const index = next++;
        const out = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = {
            ...message(index),
            content: [{ type: "text" as const, text: sharedPrefix + "candidate text " + index }],
          };
          out.push({ type: "start", partial: result });
          out.push({ type: "done", reason: "stop", message: result });
        });
        return out;
      },
      onDecision: (decision) => decisions.push(decision),
    }, context());

    await collect(stream);
    expect(verifier.calls).toBeGreaterThan(0);
    expect(decisions[0]).toMatchObject({ path: "verifier" });
  });

  test("reports the PPT decision: winner, scores, comparison count, usage", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const verifier = new RankingVerifier();
    const decisions: unknown[] = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
      onDecision: (decision) => decisions.push(decision),
    }, context());
    await collect(stream);
    expect(decisions).toHaveLength(1);
    const decision = decisions[0] as {
      path: string;
      winnerIndex?: number;
      candidateCount?: number;
      successfulCandidates?: number;
      nComparisons?: number;
      criteria?: string[];
      scores?: number[];
      winnerScore?: number;
      usage?: unknown;
      durationMs?: number;
      model?: string;
      promptVersion?: string;
      paperEquivalent?: boolean;
      scoreSources?: unknown;
      scoreDistribution?: unknown;
      granularity?: string;
      toolUseCandidates?: number;
      terminalCandidates?: number;
    };
    expect(decision.path).toBe("verifier");
    expect(decision.winnerIndex).toBe(2);
    expect(decision.candidateCount).toBe(3);
    expect(decision.successfulCandidates).toBe(3);
    expect(decision.granularity).toBe("request_action");
    expect(decision.toolUseCandidates).toBe(0);
    expect(decision.terminalCandidates).toBe(3);
    expect(decision.nComparisons).toBe(5);
    expect(decision.paperEquivalent).toBe(true);
    expect(decision.scoreSources).toEqual({
      logprobs: 10,
      textFallback: 0,
      neutralTie: 0,
      unknown: 0,
    });
    expect(decision.scoreDistribution).toEqual({
      logprobScores: 10,
      minSupport: 1,
      meanSupport: 1,
      minProbabilityMass: 1,
      meanProbabilityMass: 1,
    });
    expect(decision.criteria).toEqual(["task_success"]);
    expect(Array.isArray(decision.scores) && decision.scores.length).toBe(3);
    expect(decision.winnerScore).toBeGreaterThanOrEqual(0);
    expect(decision.winnerScore).toBeLessThanOrEqual(1);
    expect(decision.durationMs).toBeGreaterThanOrEqual(0);
    expect(decision.usage).toBeTruthy();
    // The decision is self-describing: which verifier model and prompt
    // contract produced it, so drift is traceable per request.
    expect(decision.model).toBe("opencode-go/deepseek-v4-flash-0731");
    expect(decision.promptVersion).toBe("pairwise-granularity20-v5");
  });

  test("reports an error decision when every candidate fails", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const decisions: unknown[] = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls, new Set([0, 1, 2])),
      onDecision: (decision) => decisions.push(decision),
    }, context());
    const { result } = await collect(stream);
    expect(result.stopReason).toBe("error");
    expect(decisions).toHaveLength(1);
    const decision = decisions[0] as {
      path: string;
      candidateCount?: number;
      successfulCandidates?: number;
      error?: string;
      durationMs?: number;
    };
    expect(decision.path).toBe("error");
    expect(decision.candidateCount).toBe(3);
    expect(decision.successfulCandidates).toBe(0);
    expect(typeof decision.error).toBe("string");
    expect(decision.durationMs).toBeGreaterThanOrEqual(0);
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
      temperature: 0.35,
      anthropicCacheRefresh: true,
      anthropicCacheRefreshRequest: true,
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
      expect(call.options.temperature).toBe(0.35);
      expect(call.options.anthropicCacheRefresh).toBe(false);
      expect(call.options.anthropicCacheRefreshRequest).toBe(false);
    }
  });

  test("reports neutral runtime ties when verification calls fail", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const degraded: unknown[] = [];
    const decisions: unknown[] = [];
    const cacheFile = "/tmp/omp-verifier-ties-" + crypto.randomUUID() + ".json";
    temporaryFiles.push(cacheFile);
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(true),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
      cacheFile,
      onDegraded: (event) => degraded.push(event),
      onDecision: (decision) => decisions.push(decision),
    }, context());
    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-0");
    expect(degraded).toHaveLength(1);
    expect(degraded[0]).toMatchObject({
      reason: "non_probabilistic_scores",
      granularity: "request_action",
      scoreSources: { logprobs: 0, textFallback: 0, neutralTie: 8, unknown: 0 },
    });
    expect(decisions[0]).toMatchObject({
      path: "verifier",
      paperEquivalent: false,
      scoreSources: { logprobs: 0, textFallback: 0, neutralTie: 8, unknown: 0 },
    });
    expect(loadCache(cacheFile)).toEqual({});
  });

  test("opens a no-logprobs circuit breaker instead of retrying every job", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const decisions: unknown[] = [];
    let verifierCalls = 0;
    const cacheFile = "/tmp/omp-verifier-breaker-" + crypto.randomUUID() + ".json";
    temporaryFiles.push(cacheFile);
    class Unsupported extends VerifierClient {
      override async scoreReply(): Promise<VerifierReply> {
        verifierCalls += 1;
        // A deterministic provider rejection of the logprobs parameters: every
        // scoring job sends an identical request shape, so after the first two
        // confirmations the remaining jobs become ties without a call.
        throw new VerifierLogprobsUnsupportedError("provider rejects logprobs", { retryable: false });
      }
    }
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new Unsupported(clientConfig()),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
      cacheFile,
      onDecision: (decision) => decisions.push(decision),
    }, context());
    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-0");
    // The online profile has one criterion and one repetition. The three
    // concurrent ring jobs begin before the breaker opens; later jobs tie
    // locally, which bounds provider traffic to three calls.
    expect(verifierCalls).toBe(3);
    expect(decisions[0]).toMatchObject({
      path: "verifier",
      paperEquivalent: false,
      nComparisons: 4,
      scoreSources: { logprobs: 0, textFallback: 0, neutralTie: 8, unknown: 0 },
    });
    // Neutral ties never reach the durable cache.
    expect(loadCache(cacheFile)).toEqual({});
  });

  test("persists verified scores durably after the throttled checkpoint saves", async () => {
    const cacheFile = "/tmp/omp-verifier-durable-" + crypto.randomUUID() + ".json";
    temporaryFiles.push(cacheFile);
    const verifier = new RankingVerifier();
    await select("problem", [
      { name: "a", trace: "trace a" },
      { name: "b", trace: "trace b" },
    ], {
      ...SELF_VERIFICATION_DEFAULTS,
      criteria: CODING_AGENT_CRITERIA,
      client: verifier,
      cacheFile,
      progress: false,
    });
    const persisted = loadCache(cacheFile);
    expect(Object.keys(persisted).length).toBeGreaterThan(0);
    for (const entry of Object.values(persisted)) {
      expect(entry.source_A).toBe("logprobs");
      expect(entry.source_B).toBe("logprobs");
    }
  });

  test("marks literal score fallback outside the paper-equivalent metric", async () => {
    const degraded: unknown[] = [];
    const decisions: unknown[] = [];
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    await collect(createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(false, false, "text_fallback"),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
      onDegraded: (event) => degraded.push(event),
      onDecision: (decision) => decisions.push(decision),
    }, context()));

    expect(degraded[0]).toMatchObject({
      reason: "non_probabilistic_scores",
      scoreSources: { logprobs: 0, textFallback: 10, neutralTie: 0, unknown: 0 },
    });
    expect(decisions[0]).toMatchObject({
      path: "verifier",
      paperEquivalent: false,
      scoreSources: { logprobs: 0, textFallback: 10, neutralTie: 0, unknown: 0 },
    });
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

  test("retries transient candidate failures once and restores the configured candidate count", async () => {
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const degraded: unknown[] = [];
    const decisions: unknown[] = [];
    let nextAttempt = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      candidateRetryDelayMs: 0,
      streamSimpleFn: (_model, candidateContext, options = {}) => {
        const attempt = nextAttempt++;
        calls.push({ context: candidateContext, options });
        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => {
          if (attempt < 2) {
            const error = new Error("429 Too Many Requests: all replicas at capacity");
            candidate.fail(attempt === 0 ? Object.assign(error, { status: 429 }) : error);
            return;
          }
          const result = message(attempt);
          candidate.push({ type: "start", partial: result });
          candidate.push({ type: "done", reason: "stop", message: result });
        });
        return candidate;
      },
      onDegraded: (event) => degraded.push(event),
      onDecision: (decision) => decisions.push(decision),
    }, context());

    await collect(stream);
    expect(calls).toHaveLength(5);
    expect(degraded).toEqual([]);
    expect(decisions[0]).toMatchObject({
      candidateCount: 3,
      successfulCandidates: 3,
    });
  });

  test("cancels a pending candidate retry without issuing replacement requests", async () => {
    const controller = new AbortController();
    const decisions: unknown[] = [];
    let calls = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      candidateRetryDelayMs: 1_000,
      streamSimpleFn: () => {
        calls += 1;
        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => candidate.fail(new Error("429 Too Many Requests")));
        return candidate;
      },
      onDecision: (decision) => decisions.push(decision),
    }, context(), { signal: controller.signal });

    setTimeout(() => controller.abort(), 10);
    const { result } = await collect(stream);
    expect(result.stopReason).toBe("aborted");
    expect(calls).toBe(3);
    expect(decisions[0]).toMatchObject({ path: "aborted" });
  });
  test("short-circuits candidate fan-out once a strict majority is guaranteed", async () => {
    const decisions: unknown[] = [];
    const abortedAfterMajority: string[] = [];
    let nextIndex = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_model, _candidateContext, options = {}) => {
        const index = nextIndex++;
        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => {
          if (index < 2) {
            // Candidates 0 and 1 finish with the exact same action text.
            const result = message(0);
            candidate.push({ type: "start", partial: result });
            candidate.push({ type: "done", reason: "stop", message: result });
            return;
          }
          // Candidate 2 stalls until the majority short circuit cancels it.
          options.signal?.addEventListener("abort", () => {
            abortedAfterMajority.push(String(index));
            candidate.fail(new DOMException("aborted", "AbortError"));
          });
        });
        return candidate;
      },
      onDecision: (decision) => decisions.push(decision),
    }, context());

    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-0");
    expect(abortedAfterMajority).toEqual(["2"]);
    expect(decisions[0]).toMatchObject({
      path: "majority",
      candidateCount: 3,
      successfulCandidates: 2,
      discardedCandidates: 1,
      winnerIndex: 0,
    });
  });

  test("replays a strict majority when a discarded provider ignores abort", async () => {
    const decisions: unknown[] = [];
    let nextIndex = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: () => {
        const index = nextIndex++;
        const candidate = new AssistantMessageEventStream();
        if (index < 2) {
          queueMicrotask(() => {
            const result = message(0);
            candidate.push({ type: "start", partial: result });
            candidate.push({ type: "done", reason: "stop", message: result });
          });
        }
        return candidate;
      },
      onDecision: (decision) => decisions.push(decision),
    }, context());

    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-0");
    expect(decisions[0]).toMatchObject({
      path: "majority",
      candidateCount: 3,
      successfulCandidates: 2,
      discardedCandidates: 1,
      winnerIndex: 0,
    });
  });

  test("does not short-circuit when no action reaches a strict majority early", async () => {
    const decisions: unknown[] = [];
    let nextIndex = 0;
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: new RankingVerifier(),
      apiKeyResolver: () => "original-key",
      streamSimpleFn: (_model, _candidateContext) => {
        const index = nextIndex++;
        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = message(index);
          candidate.push({ type: "start", partial: result });
          candidate.push({ type: "done", reason: "stop", message: result });
        });
        return candidate;
      },
      onDecision: (decision) => decisions.push(decision),
    }, context());

    const { result } = await collect(stream);
    // Three distinct actions: no majority, so the full PPT selects the winner.
    expect(decisions[0]).toMatchObject({
      path: "verifier",
      candidateCount: 3,
      successfulCandidates: 3,
      discardedCandidates: 0,
    });
    expect(result.stopReason).toBe("stop");
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
      granularity: "request_action",
      candidateCount: 3,
      successfulCandidates: 1,
      toolUseCandidates: 0,
      terminalCandidates: 1,
      nonterminalCandidates: 0,
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
    const decisions: unknown[] = [];
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
      onDecision: (decision) => decisions.push(decision),
    }, context(), { signal: controller.signal });
    controller.abort();
    const { events, result } = await collect(stream);
    expect(result.stopReason).toBe("aborted");
    expect((events as Array<{ type: string }>).map((event) => event.type)).toEqual(["error"]);
    expect(decisions).toHaveLength(1);
    expect(decisions[0]).toMatchObject({ path: "aborted" });
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

  test("keeps selected trajectory evidence in chronological order", () => {
    const first = message(0);
    first.content = [{ type: "text", text: "first assistant action" }];
    const second = message(1);
    second.content = [{ type: "text", text: "second assistant action" }];
    const history: Context = {
      ...context(),
      messages: [
        { role: "user", content: "Complete the task.", timestamp: 1 },
        first,
        {
          role: "toolResult",
          toolCallId: "first",
          toolName: "bash",
          content: [{ type: "text", text: "first tool output" }],
          isError: false,
          timestamp: 2,
        },
        second,
        {
          role: "toolResult",
          toolCallId: "second",
          toolName: "bash",
          content: [{ type: "text", text: "second tool output" }],
          isError: false,
          timestamp: 3,
        },
      ],
    };
    const serialized = serializeContext(history);
    const positions = [
      serialized.indexOf("first assistant action"),
      serialized.indexOf("first tool output"),
      serialized.indexOf("second assistant action"),
      serialized.indexOf("second tool output"),
    ];
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
  });

  test("backfills prior conversation context for request/action verification", () => {
    const priorAssistant = message(0);
    priorAssistant.content = [{ type: "text", text: "Earlier repository finding." }];
    const history: Context = {
      ...context(),
      messages: [
        { role: "user", content: "Keep Java 8 compatibility.", timestamp: 1 },
        priorAssistant,
        { role: "user", content: "Implement the current fix.", timestamp: 2 },
        {
          role: "toolResult",
          toolCallId: "current",
          toolName: "bash",
          content: [{ type: "text", text: "Current test output." }],
          isError: false,
          timestamp: 3,
        },
      ],
    };

    const serialized = serializeContext(history);
    expect(serialized).toContain("Keep Java 8 compatibility.");
    expect(serialized).toContain("Earlier repository finding.");
    expect(serialized).toContain("Implement the current fix.");
    expect(serialized.indexOf("Keep Java 8 compatibility."))
      .toBeLessThan(serialized.indexOf("Implement the current fix."));
    expect(serialized.indexOf("Implement the current fix."))
      .toBeLessThan(serialized.indexOf("Current test output."));
  });

  test("forwards user task images through every PPT verifier comparison", async () => {
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
    const verifier = new RankingVerifier(false, true);
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
    }, rich);
    await collect(stream);
    const serialized = serializeContext(rich);
    expect(serialized).toContain("Inspect this screenshot.");
    expect(serialized).not.toContain("Available tools");
    expect(serialized).not.toContain("aGVsbG8=");
    expect(verifier.calls).toBeGreaterThan(0);
    expect(verifier.prompts.every((prompt) => prompt.includes(
      "**Attached images:** 1 image(s) are attached to this message, in order;",
    ))).toBe(true);
    for (const images of verifier.imageBatches) {
      expect(images).toEqual([
        { type: "image", data: "aGVsbG8=", mimeType: "image/png", detail: "high" },
      ]);
    }
  });

  test("forwards shared trajectory images in chronological order", async () => {
    const taskImage: ImageContent = {
      type: "image",
      data: "dGFzaw==",
      mimeType: "image/png",
    };
    const assistantImage: ImageContent = {
      type: "image",
      data: "YXNzaXN0YW50",
      mimeType: "image/webp",
      detail: "high",
    };
    const followupImage: ImageContent = {
      type: "image",
      data: "Zm9sbG93dXA=",
      mimeType: "image/png",
    };
    const toolImage: ImageContent = {
      type: "image",
      data: "dG9vbA==",
      mimeType: "image/jpeg",
    };
    const priorAssistant = message(9, true);
    priorAssistant.content.push(assistantImage);
    const rich: Context = {
      ...context(),
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "Inspect the visual evidence." }, taskImage],
          timestamp: 1,
        },
        priorAssistant,
        {
          role: "user",
          content: [followupImage],
          timestamp: 2,
        },
        {
          role: "toolResult",
          toolCallId: "call-9",
          toolName: "view_image",
          content: [
            { type: "text", text: "Rendered the current UI." },
            toolImage,
          ],
          isError: false,
          timestamp: 3,
        },
      ],
    };
    const verifier = new RankingVerifier(false, true);
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    await collect(createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
    }, rich));

    expect(verifier.calls).toBeGreaterThan(0);
    expect(verifier.prompts.every((prompt) => prompt.includes(
      "**Attached images:** 4 image(s) are attached to this message, in order;",
    ))).toBe(true);
    expect(verifier.imageBatches.every((images) =>
      JSON.stringify(images) === JSON.stringify([
        taskImage,
        assistantImage,
        followupImage,
        toolImage,
      ])
    )).toBe(true);
    const serialized = serializeContext(rich);
    expect(serialized).toContain("Rendered the current UI.");
    expect(serialized).toContain("[image attached]");
    expect(serialized).not.toContain(toolImage.data);
  });

  test("labels and aligns candidate-specific images with trajectory slots", async () => {
    const sharedImage: ImageContent = {
      type: "image",
      data: "c2hhcmVk",
      mimeType: "image/png",
    };
    const candidateImages: ImageContent[] = [0, 1, 2].map((index) => ({
      type: "image",
      data: Buffer.from("candidate-" + index).toString("base64"),
      mimeType: "image/png",
    }));
    const imageContext: Context = {
      ...context(),
      messages: [{
        role: "user",
        content: [{ type: "text", text: "Compare the rendered candidates." }, sharedImage],
        timestamp: 1,
      }],
    };
    const verifier = new RankingVerifier(false, true);
    let nextIndex = 0;
    await collect(createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: () => {
        const index = nextIndex++;
        const candidate = new AssistantMessageEventStream();
        queueMicrotask(() => {
          const result = message(index);
          result.content.push(candidateImages[index]!);
          candidate.push({ type: "start", partial: result });
          candidate.push({ type: "done", reason: "stop", message: result });
        });
        return candidate;
      },
    }, imageContext));

    expect(verifier.prompts.length).toBeGreaterThan(0);
    for (let index = 0; index < verifier.prompts.length; index += 1) {
      const prompt = verifier.prompts[index]!;
      const trajectoryA = Number(/\*\*Trajectory A:\*\*[\s\S]*?candidate text (\d+)/.exec(prompt)?.[1]);
      const trajectoryB = Number(/\*\*Trajectory B:\*\*[\s\S]*?candidate text (\d+)/.exec(prompt)?.[1]);
      expect(prompt).toContain(
        "Order: 1 shared task-context image(s), then 1 Trajectory A image(s), " +
        "then 1 Trajectory B image(s).",
      );
      expect(verifier.imageBatches[index]).toEqual([
        sharedImage,
        candidateImages[trajectoryA],
        candidateImages[trajectoryB],
      ]);
    }
  });

  test("reports a visible fallback when the verifier cannot inspect task images", async () => {
    const verifier = new RankingVerifier();
    const degraded: unknown[] = [];
    const decisions: unknown[] = [];
    const calls: Array<{ context: Context; options: SimpleStreamOptions }> = [];
    const imageContext: Context = {
      ...context(),
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "Inspect this screenshot." },
          { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
        ],
        timestamp: 1,
      }],
    };
    const stream = createAutoVerifierStream({
      originalModel: model(),
      verifierClient: verifier,
      apiKeyResolver: () => "original-key",
      streamSimpleFn: fakeCandidateStreamFactory(calls),
      onDegraded: (event) => degraded.push(event),
      onDecision: (decision) => decisions.push(decision),
    }, imageContext);
    const { result } = await collect(stream);
    expect(result.responseId).toBe("response-0");
    expect(verifier.calls).toBe(0);
    expect(degraded).toHaveLength(1);
    expect(degraded[0]).toMatchObject({ reason: "verification_error" });
    expect(decisions).toHaveLength(1);
    expect(decisions[0]).toMatchObject({ path: "fallback" });
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

  test("suggests a logprobs-capable fallback model in the capability warning", async () => {
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
      models: {
        resolve: (spec: string) => spec === "@default" ? configuredModel : wrappers.get(spec),
        list: () => [
          configuredModel,
          { ...model(), provider: "deepseek", id: "deepseek-v4-flash" },
          { ...model(), provider: "kimi-code", id: "k3-256k", api: "openai-completions" },
        ],
      },
      modelRegistry: {
        getApiKey: async () => "test-key",
        resolver: () => () => "test-key",
        getProviderHeaders: () => undefined,
        refreshRuntimeProviders: async () => undefined,
      },
      sessionManager: { getSessionId: () => "suggestion-session" },
    } as never;
    const runtime = createAutomaticVerificationRuntime({
      probeVerifier: async () => {
        throw new VerifierLogprobsUnsupportedError("probe returned no token logprobs");
      },
    });
    let message = "";
    try {
      await ensureAutomaticVerification(fakePi, ctx, runtime);
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    expect(message).toContain("verifierModel");
    expect(message).toContain("deepseek/deepseek-v4-flash");
    // kimi-code is declared openai-completions but rejects logprobs, so the
    // suggestion must never recommend it.
    expect(message).not.toContain("kimi-code");
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


  test("verifies candidates with the configured verifierModel instead of the session model", async () => {
    const source = model();
    const verifier = {
      ...model(),
      provider: "deepseek",
      id: "deepseek-v4-flash",
      name: "DeepSeek V4 Flash",
    } as unknown as Model;
    let activeModel: Model = source;
    const wrappers = new Map<string, Model>();
    const fakePi = {
      pi: { settings: { getModelRole: () => "opencode-go/deepseek-v4-flash-0731:max" } },
      registerProvider: (provider: string) => {
        wrappers.set(provider + "/default", { ...source, provider, id: "default" } as unknown as Model);
      },
      getThinkingLevel: () => "high",
      setThinkingLevel: () => undefined,
      setModel: async (next: Model) => { activeModel = next; return true; },
    } as never;
    const ctx = {
      get model() { return activeModel; },
      models: {
        resolve: (spec: string) =>
          spec === "@default" ? source :
          spec.startsWith("deepseek/deepseek-v4-flash") ? verifier :
          wrappers.get(spec),
      },
      modelRegistry: {
        getApiKey: async () => "test-key",
        resolver: () => () => "test-key",
        getProviderHeaders: () => undefined,
        refreshRuntimeProviders: async () => undefined,
      },
      sessionManager: { getSessionId: () => "verifier-override-session" },
      ui: { notify: () => undefined },
    } as never;
    let probed: VerifierClient | undefined;
    const runtime = createAutomaticVerificationRuntime({
      probeVerifier: async (client) => { probed = client; },
    });

    const binding = await ensureAutomaticVerification(fakePi, ctx, runtime, 3, "deepseek/deepseek-v4-flash:high");
    expect(binding.originalModel).toBe(source);
    expect(probed?.provider).toBe("deepseek");
    expect(probed?.model).toBe("deepseek-v4-flash");
    expect(probed?.effort).toBe("high");
    expect(activeModel.provider).toBe(binding.providerName);
  });

  test("rejects an unresolvable verifierModel selector with a clear error", async () => {
    const source = model();
    let activeModel: Model = source;
    const fakePi = {
      pi: { settings: { getModelRole: () => "opencode-go/deepseek-v4-flash-0731:max" } },
      registerProvider: () => undefined,
      getThinkingLevel: () => "high",
      setThinkingLevel: () => undefined,
      setModel: async (next: Model) => { activeModel = next; return true; },
    } as never;
    const ctx = {
      get model() { return activeModel; },
      models: { resolve: (spec: string) => spec === "@default" ? source : undefined },
      modelRegistry: {
        getApiKey: async () => "test-key",
        resolver: () => () => "test-key",
        getProviderHeaders: () => undefined,
        refreshRuntimeProviders: async () => undefined,
      },
      sessionManager: { getSessionId: () => "verifier-missing-session" },
      ui: { notify: () => undefined },
    } as never;
    const runtime = createAutomaticVerificationRuntime();
    await expect(
      ensureAutomaticVerification(fakePi, ctx, runtime, 3, "missing/provider-model"),
    ).rejects.toThrow("verifierModel");
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

describe("cache and fixed self-verification defaults", () => {
  test("cache identity includes model, effort, prompt and request lineage", () => {
    const base: CacheContext = {
      criterionId: "criterion",
      criterionName: "Criterion",
      criterionDescription: "description",
      problem: "problem",
      traceA: "a",
      traceB: "b",
      imagesFingerprint: "",
      trajectoryImagesAFingerprint: "",
      trajectoryImagesBFingerprint: "",
      provider: "inferx",
      api: "openai-responses",
      model: "deepseek-v4-flash-0731",
      effort: "max",
      maxTokens: 32768,
      baseUrl: "https://example.test/v1",
      requestIdentity: "identity",
      groundTruthNote: CODING_AGENT_GROUND_TRUTH_NOTE,
      promptVersion: "pairwise-granularity20-v5",
    };
    const key = cacheKey("criterion", "task", 0, 1, 0, base);
    expect(cacheKey("criterion", "task", 0, 1, 0, { ...base, effort: "high" })).not.toBe(key);
    expect(cacheKey("criterion", "task", 0, 1, 0, { ...base, traceA: "changed" })).not.toBe(key);
    expect(cacheKey("criterion", "task", 0, 1, 0, {
      ...base,
      imagesFingerprint: "different-image",
    })).not.toBe(key);
    expect(cacheKey("criterion", "task", 0, 1, 0, {
      ...base,
      trajectoryImagesAFingerprint: "different-candidate-image",
    })).not.toBe(key);
  });

  test("saveCache merges entries durably", () => {
    const path = "/tmp/omp-verifier-" + crypto.randomUUID() + ".json";
    temporaryFiles.push(path);
    saveCache(path, {
      first: {
        score_A: 0.2,
        score_B: 0.8,
        source_A: "logprobs",
        source_B: "logprobs",
        support_A: 3,
        support_B: 2,
        probability_mass_A: 0.9,
        probability_mass_B: 0.8,
      },
    });
    saveCache(path, { second: { score_A: 0.7, score_B: 0.3 } });
    const loaded = loadCache(path);
    expect(Object.keys(loaded).sort()).toEqual(["first", "second"]);
    expect(loaded.first).toMatchObject({
      source_A: "logprobs",
      support_A: 3,
      probability_mass_A: 0.9,
    });
  });

  test("keeps neutral failure ties outside the durable cache", () => {
    const path = "/tmp/omp-verifier-neutral-" + crypto.randomUUID() + ".json";
    temporaryFiles.push(path);
    saveCache(path, {
      durable: { score_A: 0.8, score_B: 0.2, source_A: "text_fallback", source_B: "text_fallback" },
      retryable: { score_A: 0.5, score_B: 0.5, source_A: "neutral_tie", source_B: "neutral_tie" },
    });
    expect(loadCache(path)).toEqual({
      durable: { score_A: 0.8, score_B: 0.2, source_A: "text_fallback", source_B: "text_fallback" },
    });
  });

  test("uses the author's small Bo3 self-verification defaults", () => {
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
      description: "每次模型动作请求并行生成的候选数量（2-8，默认 3）",
    });
    expect(manifest.omp.settings.nEvaluations.default).toBe(1);
    expect(manifest.omp.settings.pivots.default).toBe(2);
    expect(readme).not.toContain("/verify");
    expect(readme).not.toContain("verifier_select");
  });

  test("normalizes plugin enablement and candidate-count settings", () => {
    expect(resolvePluginSettings({ enabled: true, candidateCount: 5 })).toEqual({
      enabled: true,
      candidateCount: 5,
      nEvaluations: 1,
      pivots: 2,
      verifierModel: undefined,
    });
    expect(resolvePluginSettings({ enabled: false, candidateCount: 99 })).toEqual({
      enabled: false,
      candidateCount: AUTO_CANDIDATE_COUNT,
      nEvaluations: 1,
      pivots: 2,
      verifierModel: undefined,
    });
    expect(normalizeCandidateCount("4")).toBe(4);
  });

  test("normalizes the evaluator K and pivots settings with TurboAgent online defaults", () => {
    const quality = resolvePluginSettings({ nEvaluations: "8", pivots: 3 });
    expect(quality.nEvaluations).toBe(8);
    expect(quality.pivots).toBe(3);
    expect(AUTO_SELECTION_DEFAULTS).toEqual({
      pivots: 2,
      nEvaluations: 1,
      seed: 0,
      maxWorkers: 8,
    });
    expect(resolvePluginSettings({ nEvaluations: 0 }).nEvaluations).toBe(1);
    expect(resolvePluginSettings({ nEvaluations: 99 }).nEvaluations).toBe(1);
    expect(resolvePluginSettings({ pivots: 0 }).pivots).toBe(2);
    expect(resolvePluginSettings({ pivots: 99 }).pivots).toBe(2);
    expect(resolvePluginSettings({}).nEvaluations).toBe(1);
    expect(resolvePluginSettings({}).pivots).toBe(2);
    // String coercion matches the normalizers used by the settings layer.
    expect(normalizeEvaluations(undefined)).toBe(1);
    expect(normalizePivots(undefined)).toBe(2);
    expect(normalizeEvaluations("16")).toBe(16);
    expect(normalizePivots("8")).toBe(8);
  });

  test("normalizes the verifierModel selector setting", () => {
    expect(resolvePluginSettings({ verifierModel: " deepseek/deepseek-v4-flash:high " }).verifierModel)
      .toBe("deepseek/deepseek-v4-flash:high");
    expect(resolvePluginSettings({ verifierModel: "   " }).verifierModel).toBeUndefined();
    expect(resolvePluginSettings({}).verifierModel).toBeUndefined();
    expect(resolvePluginSettings({ verifierModel: 42 }).verifierModel).toBeUndefined();
    expect(JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"))
      .omp.settings.verifierModel.type).toBe("string");
  });
});
