/** Request-level automatic self-verification provider for OMP. */

import {
  streamSimple,
  type AssistantMessage,
  type Context,
  type Message,
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
export const AUTO_CANDIDATE_COUNT_MIN = 2;
export const AUTO_CANDIDATE_COUNT_MAX = 8;
// Reference trace compaction (SWE-bench `_sb_format_trace`, MedAgentBench
// `_med_format_trace`) truncates each block at a fixed character cap rather
// than trimming the whole trajectory to an arbitrary total. Match the
// reference SWE-bench cap so cached prefixes stay stable.
const TRACE_BLOCK_MAX_CHARS = 2000;
// Hard total budget for one candidate trace. The paper's best-of-3 pair prompt
// holds problem + both traces; with problem capped at PROBLEM_MAX_CHARS we
// bound the dynamic section of any pair prompt to ~32k chars so the unknown
// per-call input cannot explode even for a many-block candidate.
const TRACE_TOTAL_MAX_CHARS = 8000;
// Long reasoning traces are kept out of the verifier prompt: the reference
// formatters only keep the agent message and tool outputs, and the paper's
// reward is read off the FINAL score block, not the reasoning.
const THINKING_MAX_CHARS = 800;
// Cap the task description (reference `problem`) so a very long user request
// cannot dominate every pairwise prompt. 16k chars is far past any task
// statement while staying far below context limits.
const PROBLEM_MAX_CHARS = 16000;
// The paper's best-of-3 self-verification config (scripts/run_bo3.py):
// pivots=1, K=2 repeated verifications per criterion. The wrapper keeps these
// paper defaults; cost control lives in terminal gating and the majority
// shortcut, not in weakening the verifier's evaluation.
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
  /** Optional JSON score cache reused across requests in the same working tree. */
  cacheFile?: string;
  onDegraded?: (event: AutoVerifierDegradedEvent) => void;
}

export interface AutoVerifierOptions {
  candidateCount?: number;
}

export function isValidCandidateCount(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= AUTO_CANDIDATE_COUNT_MIN &&
    value <= AUTO_CANDIDATE_COUNT_MAX
  );
}

/** Resolve the persisted plugin setting without allowing unbounded verifier fan-out. */
export function normalizeCandidateCount(value: unknown): number {
  const parsed = typeof value === "string" && value.trim() ? Number(value) : value;
  return isValidCandidateCount(parsed) ? parsed : AUTO_CANDIDATE_COUNT;
}

export type AutoVerifierDegradedReason =
  | "insufficient_candidates"
  | "verification_error";

export interface AutoVerifierDegradedEvent {
  reason: AutoVerifierDegradedReason;
  candidateCount: number;
  successfulCandidates: number;
  error?: string;
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
  if (!isValidCandidateCount(candidateCount)) {
    output.push({
      type: "error",
      reason: "error",
      error: terminalMessage(
        state.originalModel,
        "error",
        new Error(
          "Automatic verifier candidateCount must be an integer between " +
            AUTO_CANDIDATE_COUNT_MIN + " and " + AUTO_CANDIDATE_COUNT_MAX,
        ),
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
    // The wrapper is invoked for every model call the agent loop makes,
    // including each intermediate tool-call turn. Candidate 0 is the natural
    // single-stream response; when it is a tool-use turn it is the agent's own
    // next action, so the wrapper replays it directly and skips the whole
    // tournament. Only a TERMINAL answer expands into candidates and gets
    // verified — this turns a per-tool-step BoN/PPT cost into a
    // one-per-final-answer cost.
    let first: CandidateResult | undefined;
    let firstError: unknown;
    try {
      first = await generateCandidate(state, context, streamOptions, 0, controller.signal);
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) throw error;
      firstError = error;
    }
    if (controller.signal.aborted) throw controller.signal.reason ?? abortReason();
    if (first && isToolUseMessage(first.message)) {
      replayAssistantMessage(output, first.message);
      return;
    }

    const settled = await Promise.allSettled(
      Array.from({ length: candidateCount - 1 }, (_, index) =>
        generateCandidate(state, context, streamOptions, index + 1, controller.signal),
      ),
    );
    if (controller.signal.aborted) throw controller.signal.reason ?? abortReason();

    const successful: CandidateResult[] = [];
    if (first) successful.push(first);
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
    const majority = successful.length >= 2 ? exactActionMajority(successful) : undefined;
    if (majority !== undefined) {
      // Self-consistency gate (cost optimization, not part of the paper's
      // tournament): when a strict majority (> N/2) of candidates produced
      // the same normalized action, majority voting already decides the
      // answer, so the verifier is never called. Every request without a
      // majority runs the paper's full PPT selection below unchanged. Normal
      // success path, no degradation signal.
      winner = majority;
    } else if (successful.length < 2) {
      reportDegraded(state, {
        reason: "insufficient_candidates",
        candidateCount,
        successfulCandidates: successful.length,
      });
    } else {
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
            onError: "tie",
            cacheFile: state.cacheFile,
            progress: false,
            signal: controller.signal,
            client: state.verifierClient,
            taskName: "current_request",
          },
        );
        winner = successful[selection.index] ?? winner;
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) throw error;
        reportDegraded(state, {
          reason: "verification_error",
          candidateCount,
          successfulCandidates: successful.length,
          error: errorMessage(error),
        });
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

/**
 * Tool-use turns are agent actions, not answers to verify. Candidate 0 is the
 * natural single-stream response; when it contains a tool call the wrapper
 * replays it directly so the agent loop continues without a tournament.
 */
function isToolUseMessage(message: AssistantMessage): boolean {
  return message.stopReason === "toolUse" ||
    (Array.isArray(message.content) &&
      message.content.some(
        (block) =>
          !!block && typeof block === "object" &&
          (block as unknown as Record<string, unknown>).type === "toolCall",
      ));
}

/**
 * A normalized fingerprint of the candidate's observable action: visible text,
 * tool name, and tool arguments. Call id and transport metadata are excluded
 * so identical actions agree even when ids/usage differ.
 */
function actionFingerprint(message: AssistantMessage): string {
  const parts: string[] = [];
  for (const block of message.content ?? []) {
    if (!block || typeof block !== "object") continue;
    const value = block as unknown as Record<string, unknown>;
    if (value.type === "text" && typeof value.text === "string") {
      const text = value.text.trim();
      if (text) parts.push(text.replace(/\s+/g, " "));
    } else if (value.type === "toolCall") {
      const toolCall = value as unknown as ToolCall;
      parts.push("[tool] " + toolCall.name + " " + compactJson(toolCall.arguments));
    }
  }
  return parts.join("\n");
}

/** First candidate among the strict > N/2 exact-action majority, if any. */
function exactActionMajority(candidates: CandidateResult[]): CandidateResult | undefined {
  const counts = new Map<string, number>();
  const first = new Map<string, CandidateResult>();
  for (const candidate of candidates) {
    const fingerprint = actionFingerprint(candidate.message);
    counts.set(fingerprint, (counts.get(fingerprint) ?? 0) + 1);
    if (!first.has(fingerprint)) first.set(fingerprint, candidate);
  }
  const threshold = candidates.length / 2;
  for (const [fingerprint, count] of counts) {
    if (fingerprint && count > threshold) return first.get(fingerprint);
  }
  return undefined;
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
    // Candidates must be independent, not chained to the OMP conversation, so
    // server-side turn chaining is disabled. But sessionId / promptCacheKey /
    // providerSessionState are OMP's default-call identity: keep them so the
    // three candidates share the full-context prefix cache instead of paying
    // for a full uncached write three times.
    statefulResponses: false,
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

/**
 * Extract the task description shown to the verifier as `problem`.
 *
 * Mirrors the reference loaders: `_tb_extract_problem` and `_sb_extract_problem`
 * both derive `problem` from the USER's request, never from the system prompt,
 * tool schemas, or transport metadata. For the transparent wrapper the task is
 * the most recent user message (the request being answered); images are reduced
 * to a count and very long requests are capped so the prompt prefix stays
 * stable and cheap across the tournament's repeated comparisons.
 *
 * The verifier's criteria instruct it to treat OBSERVED tool results as ground
 * truth, so the problem also carries a bounded, latest-first slice of the
 * trajectory since that request: visible assistant actions and tool outputs
 * (per-block truncated), still excluding system/developer prompts, reasoning,
 * tool schemas, and image payloads. The task, separator, and evidence share
 * the 16k problem budget — the hard cap on every pairwise prompt.
 */
export function serializeContext(context: Context): string {
  const messages = context.messages ?? [];
  let task = "";
  let taskIndex = -1;
  let imageCount = 0;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || message.role !== "user") continue;
    const parts: string[] = [];
    const content = message.content;
    if (typeof content === "string") {
      parts.push(content);
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!block || typeof block !== "object") continue;
        const value = block as unknown as Record<string, unknown>;
        if (value.type === "text" && typeof value.text === "string") parts.push(value.text);
        if (value.type === "image") imageCount += 1;
      }
    }
    if (parts.length) {
      task = parts.join("\n").trim();
      taskIndex = index;
      break;
    }
  }
  if (!task) return "(no user request captured)";
  const imageNote = imageCount > 0
    ? `\n\n[Note: the user attached ${imageCount} image(s) as part of the request.]`
    : "";
  let problem = truncateWithMarker(task + imageNote, PROBLEM_MAX_CHARS, "\n... [task truncated]");

  const separator = "\n\n";
  const evidenceBudget = PROBLEM_MAX_CHARS - problem.length - separator.length;
  if (evidenceBudget > 0) {
    const evidence = recentTrajectoryEvidence(messages, taskIndex, evidenceBudget);
    if (evidence) problem += separator + evidence;
  }
  return problem;
}

/** Truncate `text` to at most `max` chars, folding in `marker` when cut. */
function truncateWithMarker(text: string, max: number, marker: string): string {
  const cap = Math.max(0, max);
  if (text.length <= cap) return text;
  if (marker.length >= cap) return marker.slice(0, cap);
  return text.slice(0, cap - marker.length) + marker;
}

const TRACE_SEPARATOR_LEN = "\n\n".length;

/**
 * Latest-first slice of observable agent actions and tool outputs after the
 * current request (the verifier's ground-truth signal), accumulated strictly
 * within `budget` chars and stopping the moment the budget is spent. Reasoning,
 * system/developer messages, tool schemas, and image payloads never enter.
 */
function recentTrajectoryEvidence(
  messages: Message[],
  afterIndex: number,
  budget: number,
): string {
  if (budget <= 0) return "";
  const parts: string[] = [];
  let used = 0;
  const append = (piece: string): void => {
    if (used >= budget) return;
    const sepCost = parts.length ? TRACE_SEPARATOR_LEN : 0;
    const room = budget - used - sepCost;
    if (room <= 0) return;
    parts.push(truncateWithMarker(piece, room, "\n... [truncated]"));
    used += sepCost + parts[parts.length - 1]!.length;
  };

  for (let index = messages.length - 1; index > afterIndex; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") continue;
    if (message.role === "assistant") {
      for (const block of (message as unknown as AssistantMessage).content ?? []) {
        if (used >= budget) break;
        if (!block) continue;
        const value = block as unknown as Record<string, unknown>;
        if (value.type === "text" && typeof value.text === "string") {
          const text = String(value.text).trim();
          if (text) append(truncateBlock(text));
        } else if (value.type === "toolCall") {
          const toolCall = value as unknown as ToolCall;
          append(
            "[tool call] " + toolCall.name + " " + truncateBlock(compactJson(toolCall.arguments)),
          );
        }
      }
    } else if (message.role === "toolResult") {
      const toolName = (message as unknown as { toolName?: string }).toolName;
      const isError = (message as unknown as { isError?: boolean }).isError === true;
      const textBlocks: string[] = [];
      const content = (message as unknown as { content?: unknown }).content;
      if (typeof content === "string") {
        const trimmed = content.trim();
        if (trimmed) textBlocks.push(trimmed);
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!block || typeof block !== "object") continue;
          const value = block as unknown as Record<string, unknown>;
          if (value.type === "text" && typeof value.text === "string") {
            const text = String(value.text).trim();
            if (text) textBlocks.push(text);
          } else if (value.type === "image") {
            textBlocks.push("[image attached]");
          }
        }
      }
      if (textBlocks.length) {
        const kind = isError ? "[tool error]" : "[tool output]";
        append(
          kind + (toolName ? " " + toolName : "") + "\n" +
          textBlocks.map(truncateBlock).join("\n"),
        );
      }
    }
    if (used >= budget) break;
  }
  return parts.join("\n\n");
}

export function serializeAssistantMessage(message: AssistantMessage): string {
  if (!message || !Array.isArray(message.content)) return "(no candidate content)";
  const { trace, fallbackReasoning } = serializeTrace(message);
  if (trace) return trace;
  if (fallbackReasoning) return fallbackReasoning;
  return "(no candidate content)";
}

/**
 * Build the compact candidate trace, per-block capped then bounded to a total
 * budget. The per-block cap matches the reference SWE-bench truncation; the
 * total cap keeps a many-block candidate from dominating every pairwise prompt
 * while the paper's high-effort / 32k verifier budget is untouched.
 */
function serializeTrace(
  message: AssistantMessage,
): { trace: string; fallbackReasoning: string } {
  const chunks: string[] = [];
  let used = 0;
  let fallbackReasoning = "";
  const appendWithinBudget = (chunk: string): void => {
    if (used >= TRACE_TOTAL_MAX_CHARS) return;
    const trimmed = chunk.trim();
    if (!trimmed) return;
    const sepCost = chunks.length ? TRACE_SEPARATOR_LEN : 0;
    const room = TRACE_TOTAL_MAX_CHARS - used - sepCost;
    if (room <= 0) return;
    chunks.push(truncateWithMarker(trimmed, room, "\n... [truncated]"));
    used += sepCost + chunks[chunks.length - 1]!.length;
  };
  for (const block of message.content) {
    if (used >= TRACE_TOTAL_MAX_CHARS) break;
    if (!block || typeof block !== "object") continue;
    const value = block as unknown as Record<string, unknown>;
    if (value.type === "text" && typeof value.text === "string") {
      appendWithinBudget(truncateBlock(value.text));
    } else if (value.type === "thinking" && typeof value.thinking === "string" && !fallbackReasoning) {
      const thinking = value.thinking;
      fallbackReasoning = thinking.length <= THINKING_MAX_CHARS
        ? thinking
        : thinking.slice(0, THINKING_MAX_CHARS) + "\n... [reasoning truncated]";
    } else if (value.type === "toolCall") {
      const toolCall = value as unknown as ToolCall;
      appendWithinBudget(
        "[proposed tool call] " + toolCall.name + " " + truncateBlock(compactJson(toolCall.arguments)),
      );
    } else if (value.type === "image") {
      appendWithinBudget("[image attached]");
    }
  }
  return { trace: chunks.join("\n\n"), fallbackReasoning };
}

/** Truncate one content block at the reference per-block cap. */
function truncateBlock(text: string): string {
  return text.length <= TRACE_BLOCK_MAX_CHARS
    ? text
    : text.slice(0, TRACE_BLOCK_MAX_CHARS) + "\n... [truncated]";
}

/** Compact JSON for tool-call arguments, rendered inline without indentation. */
function compactJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function reportDegraded(state: AutoVerifierState, event: AutoVerifierDegradedEvent): void {
  try {
    state.onDegraded?.(event);
  } catch {
    // A diagnostic callback cannot change the provider result.
  }
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
