import { expect, test } from "bun:test";
import { VerifierClient, VerifierLogprobsUnsupportedError } from "../src/client.ts";
import { extractScorePair } from "../src/scale.ts";
import { select } from "../src/select.ts";

function reply(inputTokens = 10) {
  return {
    choices: [{
      message: { content: "<score_A>A</score_A><score_B>T</score_B>" },
      logprobs: { content: [
        { token: "<score_A>", logprob: -1 },
        { token: "A", top_logprobs: [{ token: "A", logprob: 0 }] },
        { token: "</score_A><score_B>", logprob: -1 },
        { token: "T", top_logprobs: [{ token: "T", logprob: 0 }] },
      ] },
      finish_reason: "stop",
    }],
    usage: { prompt_tokens: inputTokens, completion_tokens: 4 },
  };
}

function client(port: number, api = "openai-completions") {
  return new VerifierClient({
    baseUrl: `http://127.0.0.1:${port}/v1`,
    apiKey: "test", provider: "test", api,
    modelId: "test", effort: "off", maxTokens: 100,
  });
}

test("invalid logprob values cannot fabricate probability evidence", async () => {
  const server = Bun.serve({
    port: 0,
    fetch() {
      const data = reply();
      const position = data.choices[0]!.logprobs.content[1]!;
      position.top_logprobs = [
        { token: "A", logprob: null as unknown as number },
        { token: "B", logprob: true as unknown as number },
        { token: "C", logprob: "0" as unknown as number },
        { token: "D", logprob: 1 },
        { token: "T", logprob: Math.log(0.5) },
      ];
      return Response.json(data);
    },
  });
  try {
    const scores = extractScorePair(await client(server.port!).scoreReply("Score both."));
    expect(scores.scoreA).toBe(0);
    expect(scores.supportA).toBe(1);
    expect(scores.probabilityMassA).toBeCloseTo(0.5);
  } finally {
    await server.stop(true);
  }
});

test("non-scale tokens cannot enter the reward through inherited object properties", () => {
  const scores = extractScorePair({
    text: "<score_A>A</score_A><score_B>T</score_B>",
    tokens: ["<score_A>", "A", "</score_A><score_B>", "T"],
    positionLogprobs: [[], [["A", Math.log(0.5)], ["constructor", Math.log(0.5)]], [], [["T", 0]]],
  });
  expect(scores.scoreA).toBe(1);
  expect(scores.supportA).toBe(1);
  expect(scores.probabilityMassA).toBeCloseTo(0.5);
  expect(extractScorePair({ text: "<score_A>constructor</score_A><score_B>T</score_B>" }).sourceA).toBe("missing");
});

for (const api of ["openai-completions", "openai-responses"]) {
  test(`${api} restores omitted score openers only with whole-answer agreement`, async () => {
    let mismatched = false;
    const text = "<score_A> A </score_A>\n<score_B> T </score_B>";
    const positions = [
      { token: "sc", logprob: -1 }, { token: "ore_A", logprob: -1 },
      { token: ">", logprob: -1 },
      { token: " A", top_logprobs: [{ token: "A", logprob: Math.log(0.75) }, { token: "T", logprob: Math.log(0.25) }] },
      { token: " </score_A>\nscore_B>", logprob: -1 },
      { token: " T", top_logprobs: [{ token: "A", logprob: Math.log(0.25) }, { token: "T", logprob: Math.log(0.75) }] },
      { token: " </score_B>", logprob: -1 },
    ];
    const server = Bun.serve({
      port: 0,
      fetch() {
        const content = mismatched ? "Unaccounted text. " + text : text;
        return Response.json(api === "openai-completions"
          ? { choices: [{ message: { content }, logprobs: { content: positions } }] }
          : { output: [{ type: "message", content: [{ type: "output_text", text: content, logprobs: positions }] }] });
      },
    });
    try {
      const verifier = client(server.port!, api);
      const restored = await verifier.scoreReply("Score both.");
      expect(restored.tokens?.join("")).toBe(text);
      expect(restored.tokens).toHaveLength(positions.length);
      const scores = extractScorePair(restored);
      expect(scores.sourceA).toBe("logprobs");
      expect(scores.sourceB).toBe("logprobs");
      expect(scores.scoreA).toBeCloseTo(0.75);
      expect(scores.scoreB).toBeCloseTo(0.25);
      mismatched = true;
      const rejected = await verifier.scoreReply("Score both.");
      expect(rejected.tokens?.join("")).toBe(positions.map((position) => position.token).join(""));
      expect(extractScorePair(rejected).sourceA).toBe("text_fallback");
    } finally {
      await server.stop(true);
    }
  });
}

for (const api of ["openai-completions", "openai-responses"]) {
  test(`${api} capability probes reject entirely empty probability evidence`, async () => {
    const positions = [{ token: "A", logprob: null, top_logprobs: [] }];
    const server = Bun.serve({
      port: 0,
      fetch() {
        return Response.json(api === "openai-completions"
          ? { choices: [{ message: { content: "A" }, logprobs: { content: positions } }] }
          : { output: [{ type: "message", content: [{ type: "output_text", text: "A", logprobs: positions }] }] });
      },
    });
    try {
      await expect(client(server.port!, api).probeLogprobs()).rejects.toBeInstanceOf(VerifierLogprobsUnsupportedError);
    } finally {
      await server.stop(true);
    }
  });
}

test("concurrent selections count only their own provider attempts and tokens", async () => {
  const attempts = new Map<string, number>();
  const server = Bun.serve({
    port: 0,
    async fetch(request) {
      const body = await request.text();
      const task = body.includes("TASK_ONE") ? "one" : "two";
      const count = (attempts.get(task) ?? 0) + 1;
      attempts.set(task, count);
      await Bun.sleep(10);
      if (task === "one" && count === 1) {
        return new Response("retry", { status: 503, headers: { "Retry-After": "0" } });
      }
      const data = reply(task === "one" ? 10 : 100);
      // An accepted response may omit usage; it is still a provider call.
      if (task === "two" && count === 1) delete (data as { usage?: unknown }).usage;
      return Response.json(data);
    },
  });
  try {
    const verifier = client(server.port!);
    const options = {
      client: verifier, progress: false, nEvaluations: 1,
      criteria: [{ id: "task", name: "Task", description: "Correctness" }],
    };
    const candidates = [{ trace: "candidate A" }, { trace: "candidate B" }];
    const [one, two] = await Promise.all([
      select("TASK_ONE", candidates, options),
      select("TASK_TWO", candidates, options),
    ]);
    expect(one.usage.calls).toBe(attempts.get("one")!);
    expect(two.usage.calls).toBe(attempts.get("two")!);
    expect(one.usage.inputTokens).toBe(20);
    expect(two.usage.inputTokens).toBe(100);
    expect(one.usage.outputTokens).toBe(8);
    expect(two.usage.outputTokens).toBe(4);
  } finally {
    await server.stop(true);
  }
});

test("body-read timeouts retain abort identity and retry with a fresh budget", async () => {
  let calls = 0;
  const server = Bun.serve({
    port: 0,
    fetch() {
      calls += 1;
      if (calls === 1) {
        return new Response(new ReadableStream({
          start(controller) { controller.enqueue(new TextEncoder().encode("{")); },
        }));
      }
      return Response.json(reply());
    },
  });
  try {
    const result = await client(server.port!).scoreReply("Score both.", { timeoutMs: 50 });
    expect(extractScorePair(result).sourceA).toBe("logprobs");
    expect(calls).toBe(2);
  } finally {
    await server.stop(true);
  }
});
