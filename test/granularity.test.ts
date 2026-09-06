import { expect, test } from "bun:test";
import type { AssistantMessage, Context, Model } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { createAutoVerifierStream, type AutoVerifierDecision } from "../src/auto.ts";
import { VerifierClient } from "../src/client.ts";

type Action = "read" | "write";

async function step(actions: Action[]) {
  let generatorCalls = 0;
  let verifierCalls = 0;
  let decision: AutoVerifierDecision | undefined;
  const generated: AssistantMessage[] = [];
  const model = {
    provider: "fixture", id: "fixture", api: "openai-completions", name: "fixture",
    baseUrl: "https://example.test", reasoning: false, input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 100000, maxTokens: 1000,
  } as Model;
  const verifier = new VerifierClient({
    baseUrl: "https://example.test", apiKey: "fixture", provider: "fixture",
    api: "openai-completions", modelId: "fixture", effort: "off", maxTokens: 1000,
  });
  verifier.scoreReply = async (prompt) => {
    verifierCalls++;
    const a = /CANDIDATE_(\d)/.exec(prompt)?.[1];
    const b = /CANDIDATE_(\d)/.exec(prompt.slice(prompt.indexOf("**Trajectory B:**")))?.[1];
    expect(a).toBeDefined();
    expect(b).toBeDefined();
    const scoreA = a === "2" ? "A" : "T";
    const scoreB = b === "2" ? "A" : "T";
    return {
      text: `<score_A>${scoreA}</score_A><score_B>${scoreB}</score_B>`,
      tokens: ["<score_A>", scoreA, "</score_A><score_B>", scoreB],
      positionLogprobs: [[], [[scoreA, 0]], [], [[scoreB, 0]]],
    };
  };
  const output = createAutoVerifierStream({
    originalModel: model, verifierClient: verifier, apiKeyResolver: async () => "fixture",
    onDecision: (value) => { decision = value; },
    streamSimpleFn: () => {
      const candidate = generatorCalls++;
      const message: AssistantMessage = {
        role: "assistant", api: "openai-completions", provider: "fixture", model: "fixture",
        timestamp: 0, stopReason: actions.length ? "toolUse" : "stop",
        content: [
          { type: "text", text: `CANDIDATE_${candidate}` },
          ...actions.map((name, index) => ({
            type: "toolCall" as const, id: `candidate-${candidate}-tool-${index}`, name,
            arguments: { path: `file-${index}.ts`, content: `candidate ${candidate} change ${index}` },
          })),
        ],
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      };
      generated.push(message);
      const stream = new AssistantMessageEventStream();
      stream.push({ type: "done", reason: actions.length ? "toolUse" : "stop", message });
      return stream;
    },
  }, {
    messages: [{ role: "user", content: "Apply the complete consistent change.", timestamp: 0 }],
    tools: ["read", "write"].map((name) => ({ name, description: name, parameters: {}, approval: name })) as Context["tools"],
  }, {}, { candidateCount: 3, nEvaluations: 1, pivots: 2 });
  const replayed = await output.result();
  expect(decision).toBeDefined();
  expect(replayed.content).toEqual(generated[decision!.winnerIndex!].content);
  expect(replayed.content.filter((block) => block.type === "toolCall")).toHaveLength(actions.length);
  if (actions.length && actions.every((name) => name === "read")) {
    expect(generatorCalls).toBe(1);
    expect(verifierCalls).toBe(0);
    expect(decision!.path).toBe("single");
  } else {
    expect(generatorCalls).toBe(3);
    expect(verifierCalls).toBeGreaterThanOrEqual(4);
    expect(verifierCalls).toBeLessThanOrEqual(5);
    expect(decision!.path).toBe("verifier");
    expect(decision!.paperEquivalent).toBe(true);
    expect(decision!.winnerIndex).toBe(2);
  }
  return { generatorCalls, verifierCalls, decision: decision! };
}

test("one consequential action batch is selected once and replays one complete winner", async () => {
  const separate: Awaited<ReturnType<typeof step>>[] = [];
  const batched: Awaited<ReturnType<typeof step>>[] = [];
  for (let i = 0; i < 6; i++) {
    separate.push(await step(["read"]));
    batched.push(await step(["read"]));
  }
  for (let i = 0; i < 3; i++) separate.push(await step(["write"]));
  batched.push(await step(["write", "write", "write"]));
  separate.push(await step([]));
  batched.push(await step([]));
  const count = (results: typeof separate) => ({
    generator: results.reduce((sum, result) => sum + result.generatorCalls, 0),
    verifier: results.reduce((sum, result) => sum + result.verifierCalls, 0),
  });
  // Fixed reward fixture chooses candidate 2 and produces five PPT edges.
  // These are replay call counts, not measured latency or coding quality.
  expect(count(separate)).toEqual({ generator: 18, verifier: 20 });
  expect(count(batched)).toEqual({ generator: 12, verifier: 10 });
});

test("a mixed read/write action batch never takes the observation shortcut", async () => {
  const result = await step(["read", "write", "read"]);
  expect(result.decision.checkpointReason).toBe("write_tool");
});
