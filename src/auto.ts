/** Request-level automatic self-verification provider for OMP. */

import {
  streamSimple,
  type AssistantMessage,
  type Context,
  type ImageContent,
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
import type { UsageSnapshot, VerifierClient } from "./client.ts";
import { PROMPT_VERSION } from "./prompt.ts";
import type { ScoreDistributionQuality, ScoreSourceCounts } from "./cache.ts";

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
// Bo3 compatibility defaults. Cost control lives in terminal-answer gating;
// every eligible best-of-N selection still runs the paper's PPT evaluator.
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
  /** Diagnose what decided each verified answer (PPT, fallback, abort, or error). */
  onDecision?: (decision: AutoVerifierDecision) => void;
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
  | "verification_error"
  | "non_probabilistic_scores";

export interface AutoVerifierDegradedEvent {
  reason: AutoVerifierDegradedReason;
  candidateCount: number;
  successfulCandidates: number;
  nonterminalCandidates?: number;
  error?: string;
  scoreSources?: ScoreSourceCounts;
  scoreDistribution?: ScoreDistributionQuality;
}

export interface AutoVerifierDecision {
  /** How the winner was chosen for this final answer.
   *
   *  - "verifier": the paper's PPT tournament selected the winner
   *  - "fallback": PPT failed or too few candidates succeeded; the earliest
   *    successful terminal candidate was replayed
   *  - "aborted": the caller cancelled the request
   *  - "error": no usable candidate; the request errored
   */
  path: "verifier" | "fallback" | "aborted" | "error";
  candidateCount: number;
  successfulCandidates: number;
  /** Generated alternatives that proposed another tool action. */
  nonterminalCandidates?: number;
  /** Raw candidate index (0..candidateCount-1) whose response was replayed. */
  winnerIndex?: number;
  /** Wall-clock time the whole selection took, in milliseconds. */
  durationMs: number;
  /** Verifier token usage for this request (verifier path only). */
  usage?: UsageSnapshot;
  /** Mean preference w/c per candidate, index-aligned with the candidate array
   *  PPT scored (successful candidates in index order). */
  scores?: number[];
  /** Mean preference of the selected winner (verifier path only). */
  winnerScore?: number;
  /** Unique directed comparisons PPT aggregated after removing ring/pivot overlap. */
  nComparisons?: number;
  /** Criterion ids the verifier averaged over. */
  criteria?: string[];
  /** Score-tag provenance across all candidate observations. */
  scoreSources?: ScoreSourceCounts;
  /** Effective returned A-T support and probability mass. */
  scoreDistribution?: ScoreDistributionQuality;
  /** True when every score came from the Eq. (3.1) token distribution. */
  paperEquivalent?: boolean;
  /** Verifier model id (provider/model) that produced the scores. */
  model?: string;
  /** Prompt contract version whose criteria scored the candidates. */
  promptVersion?: string;
  error?: string;
}

/** Verifier model + prompt contract that produced a PPT decision. */
function decisionIdentity(state: AutoVerifierState): { model: string; promptVersion: string } {
  return {
    model: state.verifierClient.provider + "/" + state.verifierClient.model,
    promptVersion: PROMPT_VERSION,
  };
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
  const startedAt = Date.now();
  let successfulCandidates = 0;
  let nonterminalCandidates = 0;
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
      if (result.status === "fulfilled") {
        if (isToolUseMessage(result.value.message)) nonterminalCandidates += 1;
        else successful.push(result.value);
      } else firstError ??= result.reason;
    }
    successful.sort((a, b) => a.index - b.index);
    successfulCandidates = successful.length;
    if (successful.length === 0) {
      throw firstError instanceof Error
        ? firstError
        : new Error(String(firstError ?? "All automatic verifier candidates failed"));
    }
    let winner = successful[0];
    let decision: AutoVerifierDecision = {
      path: "fallback",
      candidateCount,
      successfulCandidates,
      nonterminalCandidates,
      winnerIndex: winner.index,
      durationMs: Date.now() - startedAt,
    };
    if (successful.length < 2) {
      reportDegraded(state, {
        reason: "insufficient_candidates",
        candidateCount,
        successfulCandidates,
        nonterminalCandidates,
      });
    } else {
      try {
        const verificationContext = serializeVerificationContext(context);
        const selection = await select(
          verificationContext.problem,
          successful.map((candidate) => ({
            name: "candidate_" + candidate.index,
            trace: serializeAssistantMessage(candidate.message),
            images: imagesFromContent(candidate.message.content),
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
            images: verificationContext.images,
          },
        );
        if (!selection.paperEquivalent) {
          reportDegraded(state, {
            reason: "non_probabilistic_scores",
            candidateCount,
            successfulCandidates,
            nonterminalCandidates,
            scoreSources: selection.scoreSources,
            scoreDistribution: selection.scoreDistribution,
          });
        }
        winner = successful[selection.index] ?? winner;
        decision = {
          ...decision,
          path: "verifier",
          winnerIndex: winner.index,
          scores: selection.scores,
          winnerScore: selection.scores[selection.index],
          nComparisons: selection.nComparisons,
          criteria: selection.criteria,
          scoreSources: selection.scoreSources,
          scoreDistribution: selection.scoreDistribution,
          paperEquivalent: selection.paperEquivalent,
          usage: selection.usage,
          ...decisionIdentity(state),
        };
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) throw error;
        reportDegraded(state, {
          reason: "verification_error",
          candidateCount,
          successfulCandidates,
          nonterminalCandidates,
          error: errorMessage(error),
        });
        decision = {
          ...decision,
          path: "fallback",
          error: errorMessage(error),
        };
      }
    }
    reportDecision(state, {
      ...decision,
      durationMs: Date.now() - startedAt,
    });
    replayAssistantMessage(output, winner.message);
  } catch (error) {
    const reason = controller.signal.aborted || isAbortError(error) ? "aborted" : "error";
    reportDecision(state, {
      path: reason,
      candidateCount,
      successfulCandidates,
      nonterminalCandidates,
      durationMs: Date.now() - startedAt,
      error: errorMessage(error),
    });
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
    messages: context.messages.map((message) => {
      const source = message as unknown as Record<string, unknown>;
      const cloned = { ...source };
      const content = source.content;
      if (content !== undefined) cloned.content = cloneMessageData(content);
      return cloned as unknown as Message;
    }),
    tools: context.tools ? [...context.tools] : undefined,
  };
}

/** Clone mutable message content while preserving provider payload identity. */
function cloneMessageData(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(cloneMessageData);
  if (!value || typeof value !== "object") return value;
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(
      ([key, item]) => [key, cloneMessageData(item)],
    ),
  );
}

/**
 * Extract the task description shown to the verifier as `problem`.
 *
 * Mirrors the reference loaders: `_tb_extract_problem` and `_sb_extract_problem`
 * both derive `problem` from the USER's request, never from the system prompt,
 * tool schemas, or transport metadata. For the transparent wrapper the task is
 * the most recent user message (the request being answered); shared images are
 * carried separately to the multimodal verifier and very long requests are
 * capped so the prompt prefix stays stable across repeated comparisons.
 *
 * The verifier's criteria instruct it to treat OBSERVED tool results as ground
 * truth, so the problem also carries a recency-bounded chronological slice of
 * the trajectory since that request: visible assistant actions and tool outputs
 * (per-block truncated), still excluding system/developer prompts, reasoning,
 * tool schemas, and image payloads. Task images and shared trajectory images
 * travel separately in chronological order. The task, separator, and evidence
 * share the 16k problem budget — the hard cap on every pairwise prompt.
 */
function serializeVerificationContext(
  context: Context,
): { problem: string; images: ImageContent[] } {
  const messages = context.messages ?? [];
  let task = "";
  let taskIndex = -1;
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
      }
    }
    if (parts.length) {
      task = parts.join("\n").trim();
      taskIndex = index;
      break;
    }
  }
  const images = sharedContextImages(messages, taskIndex);
  if (!task) return { problem: "(no user request captured)", images };
  let problem = truncateWithMarker(task, PROBLEM_MAX_CHARS, "\n... [task truncated]");

  const separator = "\n\n";
  const evidenceBudget = PROBLEM_MAX_CHARS - problem.length - separator.length;
  if (evidenceBudget > 0) {
    const evidence = recentTrajectoryEvidence(messages, taskIndex, evidenceBudget);
    if (evidence) problem += separator + evidence;
  }
  return { problem, images };
}

function imagesFromContent(content: unknown): ImageContent[] {
  if (!Array.isArray(content)) return [];
  const images: ImageContent[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const value = block as Record<string, unknown>;
    if (
      value.type === "image" && typeof value.data === "string" &&
      typeof value.mimeType === "string"
    ) {
      images.push(block as ImageContent);
    }
  }
  return images;
}

/** Images observed by every candidate, preserving message and block chronology. */
function sharedContextImages(messages: Message[], taskIndex: number): ImageContent[] {
  const images: ImageContent[] = [];
  for (let index = Math.max(0, taskIndex); index < messages.length; index += 1) {
    const message = messages[index];
    if (!message) continue;
    images.push(...imagesFromContent((message as { content?: unknown }).content));
  }
  return images;
}

export function serializeContext(context: Context): string {
  return serializeVerificationContext(context).problem;
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
 * Recency-bounded slice of observable agent actions and tool outputs after the
 * current request (the verifier's ground-truth signal). Selection walks from
 * newest to oldest, then restores chronological order for the verifier.
 * Reasoning, system/developer messages, tool schemas, and image payloads stay outside
 * the text budget; image markers remain here while payloads travel separately.
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
      const messageParts: string[] = [];
      const messageBudget = Math.max(
        0,
        budget - used - (parts.length ? TRACE_SEPARATOR_LEN : 0),
      );
      let messageUsed = 0;
      const appendMessagePart = (piece: string): void => {
        const separatorCost = messageParts.length ? 1 : 0;
        const room = messageBudget - messageUsed - separatorCost;
        if (room <= 0) return;
        const bounded = truncateWithMarker(piece, room, "\n... [truncated]");
        messageParts.push(bounded);
        messageUsed += separatorCost + bounded.length;
      };
      for (const block of (message as unknown as AssistantMessage).content ?? []) {
        if (messageUsed >= messageBudget) break;
        if (!block) continue;
        const value = block as unknown as Record<string, unknown>;
        if (value.type === "text" && typeof value.text === "string") {
          const text = String(value.text).trim();
          if (text) appendMessagePart(truncateBlock(text));
        } else if (value.type === "toolCall") {
          const toolCall = value as unknown as ToolCall;
          appendMessagePart(
            "[tool call] " + toolCall.name + " " + truncateBlock(compactJson(toolCall.arguments)),
          );
        } else if (value.type === "image") {
          appendMessagePart("[image attached]");
        }
      }
      if (messageParts.length) append(messageParts.join("\n"));
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
  return parts.reverse().join("\n\n");
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

function reportDecision(state: AutoVerifierState, decision: AutoVerifierDecision): void {
  try {
    state.onDecision?.(decision);
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
