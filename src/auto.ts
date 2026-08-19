/** Request-level automatic self-verification provider for OMP. */

import {
  streamSimple,
  type AssistantMessage,
  type Context,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
} from "@oh-my-pi/pi-ai";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import type { ApiKeyResolver } from "@oh-my-pi/pi-ai/auth-retry";
import { select } from "./select.ts";
import {
  CODING_AGENT_CRITERIA,
  CODING_AGENT_GROUND_TRUTH_NOTE,
} from "./prompt.ts";
import type { VerifierClient } from "./client.ts";

export const AUTO_CANDIDATE_COUNT = 3;
export const AUTO_SELECTION_DEFAULTS = {
  pivots: 1,
  nEvaluations: 2,
  seed: 0,
  maxWorkers: 8,
} as const;

export interface AutoVerifierState {
  originalModel: Model;
  verifierClient: VerifierClient;
  apiKeyResolver: ApiKeyResolver;
  streamSimpleFn?: typeof streamSimple;
}

export interface AutoVerifierOptions {
  candidateCount?: number;
}

interface CandidateResult {
  index: number;
  message: AssistantMessage;
}

export function createWrappedProvider(
  state: AutoVerifierState,
  options: AutoVerifierOptions = {},
): (model: Model, context: Context, streamOptions?: SimpleStreamOptions) => AssistantMessageEventStream {
  return (_model, context, streamOptions = {}) =>
    createAutoVerifierStream(state, context, streamOptions, options);
}

export function createAutoVerifierStream(
  state: AutoVerifierState,
  context: Context,
  streamOptions: SimpleStreamOptions = {},
  options: AutoVerifierOptions = {},
): AssistantMessageEventStream {
  const output = new AssistantMessageEventStream();
  const candidateCount = options.candidateCount ?? AUTO_CANDIDATE_COUNT;
  if (!Number.isInteger(candidateCount) || candidateCount < 2) {
    output.push({
      type: "error",
      reason: "error",
      error: terminalMessage(
        state.originalModel,
        "error",
        new Error("Automatic verifier candidateCount must be at least 2"),
      ),
    });
    return output;
  }
  output.trackLocalWork(runAutomaticVerification(
    state,
    context,
    streamOptions,
    candidateCount,
    output,
  )).catch(() => undefined);
  return output;
}

async function runAutomaticVerification(
  state: AutoVerifierState,
  context: Context,
  streamOptions: SimpleStreamOptions,
  candidateCount: number,
  output: AssistantMessageEventStream,
): Promise<void> {
  const controller = new AbortController();
  const onAbort = () => controller.abort(streamOptions.signal?.reason ?? abortReason());
  if (streamOptions.signal?.aborted) onAbort();
  else streamOptions.signal?.addEventListener("abort", onAbort, { once: true });
  try {
    const settled = await Promise.allSettled(
      Array.from({ length: candidateCount }, (_, index) =>
        generateCandidate(state, context, streamOptions, index, controller.signal),
      ),
    );
    if (controller.signal.aborted) throw controller.signal.reason ?? abortReason();

    const successful: CandidateResult[] = [];
    let firstError: unknown;
    for (const result of settled) {
      if (result.status === "fulfilled") successful.push(result.value);
      else firstError ??= result.reason;
    }
    successful.sort((a, b) => a.index - b.index);
    if (successful.length === 0) {
      throw firstError instanceof Error
        ? firstError
        : new Error(String(firstError ?? "All automatic verifier candidates failed"));
    }

    let winner = successful[0];
    if (successful.length >= 2) {
      try {
        const selection = await select(
          serializeContext(context),
          successful.map((candidate) => ({
            name: "candidate_" + candidate.index,
            trace: serializeAssistantMessage(candidate.message),
          })),
          {
            ...AUTO_SELECTION_DEFAULTS,
            criteria: CODING_AGENT_CRITERIA,
            groundTruthNote: CODING_AGENT_GROUND_TRUTH_NOTE,
            onError: "raise",
            progress: false,
            signal: controller.signal,
            client: state.verifierClient,
            taskName: "current_request",
          },
        );
        winner = successful[selection.index] ?? winner;
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) throw error;
        // Verifier failures leave the first complete candidate available.
      }
    }
    replayAssistantMessage(output, winner.message);
  } catch (error) {
    const reason = controller.signal.aborted || isAbortError(error) ? "aborted" : "error";
    output.push({
      type: "error",
      reason,
      error: terminalMessage(state.originalModel, reason, error),
    });
  } finally {
    streamOptions.signal?.removeEventListener("abort", onAbort);
  }
}

async function generateCandidate(
  state: AutoVerifierState,
  context: Context,
  streamOptions: SimpleStreamOptions,
  index: number,
  signal: AbortSignal,
): Promise<CandidateResult> {
  const candidateOptions: SimpleStreamOptions = {
    ...streamOptions,
    apiKey: state.apiKeyResolver,
    signal,
    statefulResponses: false,
    sessionId: undefined,
    providerSessionState: undefined,
    promptCacheKey: undefined,
  };
  const stream = (state.streamSimpleFn ?? streamSimple)(
    state.originalModel,
    cloneContext(context),
    candidateOptions,
  );
  const message = await stream.result();
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    throw new Error(message.errorMessage ?? "Candidate " + index + " failed");
  }
  return { index, message };
}

function cloneContext(context: Context): Context {
  return {
    systemPrompt: context.systemPrompt ? [...context.systemPrompt] : undefined,
    messages: context.messages.map((message) => ({ ...message })),
    tools: context.tools ? [...context.tools] : undefined,
  };
}

export function serializeContext(context: Context): string {
  const sections: string[] = [];
  if (context.systemPrompt?.length) {
    sections.push("System prompt:\n" + context.systemPrompt.join("\n\n"));
  }
  for (const [index, message] of context.messages.entries()) {
    sections.push(
      "Message " + (index + 1) + " (" + message.role + "):\n" +
      serializeMessageContent(message.content),
    );
  }
  return sections.join("\n\n") || "The current coding-agent request has no prior context.";
}

export function serializeAssistantMessage(message: AssistantMessage): string {
  const metadata = {
    api: message.api,
    provider: message.provider,
    model: message.model,
    responseId: message.responseId,
    stopReason: message.stopReason,
    stopDetails: message.stopDetails,
    usage: message.usage,
  };
  return "Assistant metadata:\n" + JSON.stringify(metadata) +
    "\n\nAssistant content:\n" + serializeMessageContent(message.content);
}

function serializeMessageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return JSON.stringify(content);
  return content.map((block) => {
    if (!block || typeof block !== "object") return String(block);
    const value = block as Record<string, unknown>;
    if (value.type === "text") return "[text]\n" + String(value.text ?? "");
    if (value.type === "thinking") return "[thinking]\n" + String(value.thinking ?? "");
    if (value.type === "toolCall") {
      return "[toolCall " + String(value.name ?? "") + "]\n" + JSON.stringify({
        id: value.id,
        arguments: value.arguments,
      });
    }
    if (value.type === "image") return "[image " + String(value.mimeType ?? "unknown") + "]";
    return JSON.stringify(value);
  }).join("\n\n");
}

function replayAssistantMessage(
  output: AssistantMessageEventStream,
  winner: AssistantMessage,
): void {
  const partial: AssistantMessage = { ...winner, content: [] };
  output.push({ type: "start", partial });
  let contentIndex = 0;
  for (const block of winner.content) {
    if (block.type === "text") replayText(output, partial, block, contentIndex++);
    else if (block.type === "thinking") replayThinking(output, partial, block, contentIndex++);
    else if (block.type === "image") {
      partial.content.push(block);
      output.push({
        type: "image_end",
        contentIndex: contentIndex++,
        content: block,
        partial: { ...partial, content: [...partial.content] },
      });
    } else if (block.type === "toolCall") {
      replayToolCall(output, partial, block, contentIndex++);
    } else {
      partial.content.push(block);
      contentIndex += 1;
    }
  }
  if (winner.stopReason === "error" || winner.stopReason === "aborted") {
    output.push({ type: "error", reason: winner.stopReason, error: winner });
    return;
  }
  output.push({ type: "done", reason: winner.stopReason, message: winner });
}

function replayText(
  output: AssistantMessageEventStream,
  partial: AssistantMessage,
  block: TextContent,
  contentIndex: number,
): void {
  const partialBlock: TextContent = { ...block, text: "" };
  partial.content.push(partialBlock);
  output.push({ type: "text_start", contentIndex, partial: snapshot(partial) });
  partialBlock.text = block.text;
  output.push({ type: "text_delta", contentIndex, delta: block.text, partial: snapshot(partial) });
  output.push({ type: "text_end", contentIndex, content: block.text, partial: snapshot(partial) });
}

function replayThinking(
  output: AssistantMessageEventStream,
  partial: AssistantMessage,
  block: ThinkingContent,
  contentIndex: number,
): void {
  const partialBlock: ThinkingContent = { ...block, thinking: "" };
  partial.content.push(partialBlock);
  output.push({ type: "thinking_start", contentIndex, partial: snapshot(partial) });
  partialBlock.thinking = block.thinking;
  output.push({ type: "thinking_delta", contentIndex, delta: block.thinking, partial: snapshot(partial) });
  output.push({ type: "thinking_end", contentIndex, content: block.thinking, partial: snapshot(partial) });
}

function replayToolCall(
  output: AssistantMessageEventStream,
  partial: AssistantMessage,
  block: ToolCall,
  contentIndex: number,
): void {
  const partialBlock: ToolCall = { ...block, arguments: {} };
  partial.content.push(partialBlock);
  output.push({ type: "toolcall_start", contentIndex, partial: snapshot(partial) });
  partialBlock.arguments = { ...block.arguments };
  output.push({
    type: "toolcall_delta",
    contentIndex,
    delta: JSON.stringify(block.arguments),
    partial: snapshot(partial),
  });
  output.push({ type: "toolcall_end", contentIndex, toolCall: block, partial: snapshot(partial) });
}

function snapshot(message: AssistantMessage): AssistantMessage {
  return {
    ...message,
    content: message.content.map((block) =>
      block.type === "toolCall" ? { ...block, arguments: { ...block.arguments } } : { ...block },
    ),
  };
}

function abortReason(): DOMException {
  return new DOMException("The operation was aborted", "AbortError");
}

function isAbortError(error: unknown): boolean {
  return (error instanceof DOMException && error.name === "AbortError") ||
    (error instanceof Error && error.name === "AbortError");
}

function terminalMessage(
  model: Model,
  reason: "error" | "aborted",
  error: unknown,
): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: reason,
    errorMessage: error instanceof Error ? error.message : String(error),
    timestamp: Date.now(),
  };
}
