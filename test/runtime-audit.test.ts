import { expect, test } from "bun:test";
import type { AssistantMessage, Context, Model } from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { createAutoVerifierStream, serializeContext, type AutoVerifierDecision } from "../src/auto.ts";
import { VerifierClient } from "../src/client.ts";

test("long user instructions do not erase observed execution evidence", () => {
  const context = { messages: [
    { role: "user", content: "Instructions ".repeat(2000), timestamp: 0 },
    { role: "toolResult", toolCallId: "test", toolName: "bash", content: [
      { type: "text", text: "CRITICAL_TEST_FAILURE: expected 2 but received 3" },
    ], isError: true, timestamp: 1 },
  ] } as Context;
  expect(serializeContext(context).includes("CRITICAL_TEST_FAILURE")).toBe(true);
});

test.each([
  {
    name: "literal tool-call text and executable tool call stay distinct",
    contents: [
      [{ type: "text", text: '[tool_call: bash({"command":"true"})]' }],
      [{ type: "toolCall", id: "execute", name: "bash", arguments: { command: "true" } }],
    ],
    majority: false,
  },
  {
    name: "text block boundaries cannot collide through newline concatenation",
    contents: [
      [{ type: "text", text: "first\nsecond" }],
      [{ type: "text", text: "first" }, { type: "text", text: "second" }],
    ],
    majority: false,
  },
  {
    name: "equivalent tool arguments ignore object key order and transport IDs",
    contents: [
      [{ type: "toolCall", id: "first", name: "bash", arguments: { command: "true", timeout: 1 } }],
      [{ type: "toolCall", id: "second", name: "bash", arguments: { timeout: 1, command: "true" } }],
    ],
    majority: true,
  },
])("$name", async ({ contents, majority }) => {
  const model = { provider: "test", id: "test", api: "openai-completions", name: "test",
    baseUrl: "https://example.test", reasoning: false, input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 100000, maxTokens: 1000,
  } as Model;
  const verifier = new VerifierClient({ baseUrl: "https://example.test", apiKey: "test",
    provider: "test", api: "openai-completions", modelId: "test", effort: "off", maxTokens: 1000 });
  verifier.scoreReply = async () => { throw new Error("intentional audit verifier failure"); };
  let index = 0;
  let decision: AutoVerifierDecision | undefined;
  const output = createAutoVerifierStream({ originalModel: model, verifierClient: verifier,
    apiKeyResolver: async () => "test", onDecision: (value) => { decision = value; },
    streamSimpleFn: () => {
      const content = [...contents[index++]] as AssistantMessage["content"];
      const message = { role: "assistant", content, api: "openai-completions", provider: "test",
        model: "test", stopReason: content[0].type === "text" ? "stop" : "toolUse", timestamp: 0,
        usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
      } as AssistantMessage;
      const stream = new AssistantMessageEventStream();
      stream.push({ type: "done", reason: message.stopReason as "stop" | "toolUse", message });
      return stream;
    },
  }, { messages: [{ role: "user", content: "Check the command", timestamp: 0 }] }, {}, { candidateCount: 2 });
  await output.result();
  expect(decision?.path === "majority").toBe(majority);
});
