/** Process-reward checkpoint verification provider for OMP. */

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
import { isProviderRetryableError } from "@oh-my-pi/pi-ai/error";
import {
  resolveToolTier,
  type ToolApproval,
  type ToolTier,
} from "@oh-my-pi/pi-coding-agent/tools/approval";
import { select } from "./select.ts";
import {
  CODING_AGENT_ACTION_CRITERIA,
  CODING_AGENT_ACTION_GROUND_TRUTH_NOTE,
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
const CANDIDATE_TRANSIENT_RETRIES = 1;
const CANDIDATE_RETRY_BASE_DELAY_MS = 500;
const CANDIDATE_RETRY_STAGGER_MS = 100;
const OMP_UTILITY_REQUEST_MAX_TOKENS = 2048;
const MANDATORY_REASONING_REJECTION =
  /(?:reasoning|thinking).{0,120}(?:mandatory|required|must be enabled|cannot be disabled)/i;
// Appendix B.3 evaluates LLM-as-a-Verifier as a process reward model (PRM)
// while varying the sampled actions per step. Audited OMP read-tier observation
// calls use one sample. Consequential checkpoints expand to configurable N
// actions (default 3, range 2-8), then
// use TurboAgent's exact majority shortcut or PPT with k=2, K=1 and C=1.
// The first sample completes before expansion, warming the shared provider
// prompt prefix for the remaining N-1 calls. The offline self-verification API
// keeps its separate Bo3 profile.
export const AUTO_SELECTION_DEFAULTS = {
  pivots: 2,
  nEvaluations: 1,
  seed: 0,
  maxWorkers: 8,
} as const;

export const AUTO_VERIFICATION_GRANULARITY = "prm" as const;
export type AutoVerificationGranularity = typeof AUTO_VERIFICATION_GRANULARITY;

export type ProcessCheckpointReason =
  | "read_only_tools"
  | "terminal_response"
  | "write_tool"
  | "exec_tool"
  | "stateful_read_tool"
  | "unclassified_read_tool"
  | "unknown_tool"
  | "approval_error"
  | "missing_tool_call"
  | "proposal_failed";

export interface ProcessActionClassification {
  scope: "observation" | "checkpoint";
  reason: Exclude<ProcessCheckpointReason, "proposal_failed">;
  toolTiers: ToolTier[];
}

export interface AutoVerifierState {
  originalModel: Model;
  verifierClient: VerifierClient;
  apiKeyResolver: ApiKeyResolver;
  streamSimpleFn?: typeof streamSimple;
  /** Override the transient candidate retry delay for deterministic tests. */
  candidateRetryDelayMs?: number;
  /** Optional JSON score cache reused across requests in the same working tree. */
  cacheFile?: string;
  onDegraded?: (event: AutoVerifierDegradedEvent) => void;
  /** Surface long buffered selection phases without exposing losing candidate content. */
  onPhase?: (event: AutoVerifierPhaseEvent) => void;
  /** Diagnose what decided each verified action (majority, PPT, fallback, abort, or error). */
  onDecision?: (decision: AutoVerifierDecision) => void;
  /** Learned endpoint capability after a 400 rejects disabled reasoning. */
  mandatoryReasoningRequired?: boolean;
}

export interface AutoVerifierOptions {
  candidateCount?: number;
  /** Paper §4.2 quality/cost axis: independent repeated verifications per criterion (online default 1). */
  nEvaluations?: number;
  /** PPT pivot count k per paper §3.2; clamped to the candidate count at run time (online default 2). */
  pivots?: number;
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

// Repeated verifications per criterion (K) and PPT pivot count (k) are the
// paper's documented quality/cost knobs (Eq. (3.1), §4.2 for K; §3.2 for k).
// K bounds stay well within the paper's experimental range; pivots are capped
// at the maximum candidate count so the O(Nk) PPT never degenerates to an
// uncontrolled round-robin.
export const AUTO_EVALUATIONS_MIN = 1;
export const AUTO_EVALUATIONS_MAX = 16;
export const AUTO_PIVOTS_MIN = 1;
export const AUTO_PIVOTS_MAX = AUTO_CANDIDATE_COUNT_MAX;

export function isValidEvaluations(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= AUTO_EVALUATIONS_MIN &&
    value <= AUTO_EVALUATIONS_MAX
  );
}

export function isValidPivots(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= AUTO_PIVOTS_MIN &&
    value <= AUTO_PIVOTS_MAX
  );
}

export function normalizeEvaluations(value: unknown): number {
  const parsed = typeof value === "string" && value.trim() ? Number(value) : value;
  return isValidEvaluations(parsed) ? parsed : AUTO_SELECTION_DEFAULTS.nEvaluations;
}

export function normalizePivots(value: unknown): number {
  const parsed = typeof value === "string" && value.trim() ? Number(value) : value;
  return isValidPivots(parsed) ? parsed : AUTO_SELECTION_DEFAULTS.pivots;
}

export type AutoVerifierDegradedReason =
  | "insufficient_candidates"
  | "verification_error"
  | "non_probabilistic_scores";

export interface AutoVerifierDegradedEvent {
  reason: AutoVerifierDegradedReason;
  granularity: AutoVerificationGranularity;
  candidateCount: number;
  successfulCandidates: number;
  toolUseCandidates?: number;
  terminalCandidates?: number;
  /** @deprecated Use toolUseCandidates. */
  nonterminalCandidates?: number;
  error?: string;
  scoreSources?: ScoreSourceCounts;
  scoreDistribution?: ScoreDistributionQuality;
}

export interface AutoVerifierPhaseEvent {
  phase:
    | "sampling_action"
    | "generating_candidates"
    | "verifying_candidates"
    | "replaying_winner";
  candidateCount: number;
  successfulCandidates?: number;
}

export interface AutoVerifierDecision {
  /** How the winner was chosen for this agent action.
   *
   *  - "single": a read-only PRM observation used k=1 and skipped selection
   *  - "majority": an exact action majority selected the winner without PPT
   *  - "verifier": the paper's PPT tournament selected the winner
   *  - "fallback": PPT failed or too few candidates succeeded; the earliest
   *    successful candidate was replayed
   *  - "aborted": the caller cancelled the request
   *  - "error": no usable candidate; the request errored
   */
  path: "single" | "majority" | "verifier" | "fallback" | "aborted" | "error";
  granularity: AutoVerificationGranularity;
  /** Configured sample budget N for consequential PRM checkpoints. */
  candidateCount: number;
  /** Candidate indices actually dispatched for this process step. */
  sampledCandidates: number;
  successfulCandidates: number;
  /** Why this process step used k=1 or expanded to the configured N. */
  checkpointReason?: ProcessCheckpointReason;
  /** Candidates still in flight when a strict majority (count > N/2) became
   *  guaranteed and the remaining fan-out was cancelled (majority shortcut). */
  discardedCandidates?: number;
  /** Successful candidates that proposed a tool action. */
  toolUseCandidates?: number;
  /** Successful candidates that proposed a terminal response. */
  terminalCandidates?: number;
  /** @deprecated Use toolUseCandidates. */
  nonterminalCandidates?: number;
  /** Raw candidate index (0..candidateCount-1) whose response was replayed. */
  winnerIndex?: number;
  /** Stop reason of the replayed action, including toolUse. */
  winnerStopReason?: AssistantMessage["stopReason"];
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
    {
      candidateCount,
      nEvaluations: normalizeEvaluations(options.nEvaluations),
      pivots: normalizePivots(options.pivots),
    },
    output,
  )).catch(() => undefined);
  return output;
}

async function runAutomaticVerification(
  state: AutoVerifierState,
  context: Context,
  streamOptions: SimpleStreamOptions,
  options: AutoVerifierOptions,
  output: AssistantMessageEventStream,
): Promise<void> {
  // OMP utility calls such as automatic title generation are outside the
  // coding-agent action trajectory. Keep them single-track so a title request
  // cannot trigger Best-of-N/PPT work or selection-phase UI.
  if (isOmpUtilityRequest(context, streamOptions)) {
    await runUtilityRequest(state, context, streamOptions, output);
    return;
  }
  const candidateCount = options.candidateCount ?? AUTO_CANDIDATE_COUNT;
  const controller = new AbortController();
  const onAbort = () => controller.abort(streamOptions.signal?.reason ?? abortReason());
  if (streamOptions.signal?.aborted) onAbort();
  else streamOptions.signal?.addEventListener("abort", onAbort, { once: true });
  const startedAt = Date.now();
  let sampledCandidates = 0;
  let successfulCandidates = 0;
  let toolUseCandidates = 0;
  let terminalCandidates = 0;
  let checkpointReason: ProcessCheckpointReason | undefined;
  try {
    if (streamOptions.execHandlers) {
      throw new Error(
        "Automatic process-reward verification requires declarative tool calls; " +
          "provider-native execHandlers can execute during candidate generation.",
      );
    }

    // Appendix B.3 treats the number of sampled actions at a process step as a
    // quality/compute axis. Sample one proposal first. Read-tier observations
    // use k=1; consequential steps expand to N total samples and enter the same
    // exact-majority/PPT selector. Completing the proposal before fan-out also
    // warms the shared prompt prefix for all N-1 additional calls.
    reportPhase(state, { phase: "sampling_action", candidateCount: 1 });
    sampledCandidates = 1;
    let proposal: PromiseSettledResult<CandidateResult>;
    try {
      proposal = {
        status: "fulfilled",
        value: await generateCandidate(state, context, streamOptions, 0, controller.signal),
      };
    } catch (reason) {
      proposal = { status: "rejected", reason };
    }
    if (controller.signal.aborted) throw controller.signal.reason ?? abortReason();

    if (proposal.status === "fulfilled") {
      const classification = classifyProcessAction(proposal.value.message, context.tools);
      checkpointReason = classification.reason;
      if (classification.scope === "observation") {
        successfulCandidates = 1;
        toolUseCandidates = 1;
        reportPhase(state, {
          phase: "replaying_winner",
          candidateCount: 1,
          successfulCandidates,
        });
        reportDecision(state, {
          path: "single",
          granularity: AUTO_VERIFICATION_GRANULARITY,
          candidateCount,
          sampledCandidates,
          successfulCandidates,
          checkpointReason,
          toolUseCandidates,
          terminalCandidates,
          nonterminalCandidates: toolUseCandidates,
          winnerIndex: proposal.value.index,
          winnerStopReason: proposal.value.message.stopReason,
          durationMs: Date.now() - startedAt,
        });
        replayAssistantMessage(output, proposal.value.message);
        return;
      }
    } else {
      checkpointReason = "proposal_failed";
    }

    reportPhase(state, {
      phase: "generating_candidates",
      candidateCount,
      successfulCandidates: proposal.status === "fulfilled" ? 1 : 0,
    });
    sampledCandidates = candidateCount;
    const gathered = await gatherCandidates(
      (index, signal) => generateCandidate(state, context, streamOptions, index, signal),
      candidateCount,
      controller.signal,
      [proposal],
    );
    if (controller.signal.aborted) throw controller.signal.reason ?? abortReason();

    let firstError: unknown;
    const successful: CandidateResult[] = [];
    for (const result of gathered.results) {
      if (result.status === "fulfilled") {
        successful.push(result.value);
        if (isToolUseMessage(result.value.message)) toolUseCandidates += 1;
        else terminalCandidates += 1;
      } else if (gathered.discardedCandidates === 0 || !isAbortError(result.reason)) {
        firstError ??= result.reason;
      }
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
      granularity: AUTO_VERIFICATION_GRANULARITY,
      candidateCount,
      sampledCandidates,
      successfulCandidates,
      checkpointReason,
      discardedCandidates: gathered.discardedCandidates,
      toolUseCandidates,
      terminalCandidates,
      nonterminalCandidates: toolUseCandidates,
      winnerIndex: winner.index,
      winnerStopReason: winner.message.stopReason,
      durationMs: Date.now() - startedAt,
    };
    if (successful.length < 2) {
      reportDegraded(state, {
        reason: "insufficient_candidates",
        granularity: AUTO_VERIFICATION_GRANULARITY,
        candidateCount,
        successfulCandidates,
        toolUseCandidates,
        terminalCandidates,
        nonterminalCandidates: toolUseCandidates,
      });
    } else {
      const actionIdentities = successful.map((candidate) =>
        serializeActionIdentity(candidate.message)
      );
      const majority = exactActionMajority(actionIdentities);
      if (majority) {
        winner = successful[majority.index] ?? winner;
        decision = {
          ...decision,
          path: "majority",
          winnerIndex: winner.index,
          winnerStopReason: winner.message.stopReason,
          scores: majority.scores,
          winnerScore: majority.scores[majority.index],
          nComparisons: 0,
        };
      } else {
        try {
          reportPhase(state, {
            phase: "verifying_candidates",
            candidateCount,
            successfulCandidates,
          });
          const verificationContext = serializeVerificationContext(context);
          const selection = await select(
            verificationContext.problem,
            successful.map((candidate) => ({
              name: "candidate_" + candidate.index,
              trace: serializeAssistantMessage(candidate.message),
              images: imagesFromContent(candidate.message.content),
            })),
            {
              pivots: options.pivots ?? AUTO_SELECTION_DEFAULTS.pivots,
              nEvaluations: options.nEvaluations ?? AUTO_SELECTION_DEFAULTS.nEvaluations,
              seed: AUTO_SELECTION_DEFAULTS.seed,
              maxWorkers: AUTO_SELECTION_DEFAULTS.maxWorkers,
              criteria: CODING_AGENT_ACTION_CRITERIA,
              groundTruthNote: CODING_AGENT_ACTION_GROUND_TRUTH_NOTE,
              onError: "tie",
              cacheFile: state.cacheFile,
              progress: false,
              signal: controller.signal,
              client: state.verifierClient,
              taskName: "current_action",
              images: verificationContext.images,
            },
          );
          if (!selection.paperEquivalent) {
            reportDegraded(state, {
              reason: "non_probabilistic_scores",
              granularity: AUTO_VERIFICATION_GRANULARITY,
              candidateCount,
              successfulCandidates,
              toolUseCandidates,
              terminalCandidates,
              nonterminalCandidates: toolUseCandidates,
              scoreSources: selection.scoreSources,
              scoreDistribution: selection.scoreDistribution,
            });
          }
          winner = successful[selection.index] ?? winner;
          decision = {
            ...decision,
            path: "verifier",
            winnerIndex: winner.index,
            winnerStopReason: winner.message.stopReason,
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
            granularity: AUTO_VERIFICATION_GRANULARITY,
            candidateCount,
            successfulCandidates,
            toolUseCandidates,
            terminalCandidates,
            nonterminalCandidates: toolUseCandidates,
            error: errorMessage(error),
          });
          decision = {
            ...decision,
            path: "fallback",
            error: errorMessage(error),
          };
        }
      }
    }
    reportPhase(state, {
      phase: "replaying_winner",
      candidateCount,
      successfulCandidates,
    });
    reportDecision(state, {
      ...decision,
      durationMs: Date.now() - startedAt,
    });
    replayAssistantMessage(output, winner.message);
  } catch (error) {
    const reason = controller.signal.aborted || isAbortError(error) ? "aborted" : "error";
    reportDecision(state, {
      path: reason,
      granularity: AUTO_VERIFICATION_GRANULARITY,
      candidateCount,
      sampledCandidates,
      successfulCandidates,
      checkpointReason,
      toolUseCandidates,
      terminalCandidates,
      nonterminalCandidates: toolUseCandidates,
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

function isOmpUtilityRequest(context: Context, streamOptions: SimpleStreamOptions): boolean {
  return (
    streamOptions.disableReasoning === true &&
    streamOptions.temperature === 0 &&
    typeof streamOptions.maxTokens === "number" &&
    streamOptions.maxTokens > 0 &&
    streamOptions.maxTokens <= OMP_UTILITY_REQUEST_MAX_TOKENS &&
    streamOptions.sessionId === undefined &&
    streamOptions.promptCacheKey === undefined &&
    !context.tools?.length
  );
}

async function runUtilityRequest(
  state: AutoVerifierState,
  context: Context,
  streamOptions: SimpleStreamOptions,
  output: AssistantMessageEventStream,
): Promise<void> {
  const signal = streamOptions.signal ?? new AbortController().signal;
  try {
    const candidate = await generateCandidate(
      state,
      context,
      streamOptions,
      0,
      signal,
      true,
    );
    replayAssistantMessage(output, candidate.message);
  } catch (error) {
    const reason = signal.aborted || isAbortError(error) ? "aborted" : "error";
    output.push({
      type: "error",
      reason,
      error: terminalMessage(state.originalModel, reason, error),
    });
  }
}

type RuntimeApprovalTool = NonNullable<Context["tools"]>[number] & {
  approval?: ToolApproval;
  formatApprovalDetails?: (args: unknown) => string | string[] | undefined;
};

// OMP's approval tier models execution risk, while PRM scheduling needs action
// effects. These built-ins are stable, read-tier observations whose execution
// only obtains evidence or advances an internal wait. Every other read-tier
// tool remains a checkpoint until OMP exposes first-class effect metadata.
const OMP_OBSERVATION_TOOL_NAMES = new Set([
  "ast_grep",
  "computer",
  "debug",
  "find",
  "glob",
  "grep",
  "github",
  "inspect_image",
  "ls",
  "lsp",
  "read",
  "recall",
  "reflect",
  "think",
  "vibe_list",
  "vibe_wait",
  "web_search",
]);

// These OMP tools deliberately use approval="read" for UX/security purposes,
// yet they commit control flow, session state, worker state, or durable memory.
// Keeping the list explicit makes upgrades auditable and prevents them from
// silently taking the PRM k=1 observation path.
const OMP_STATEFUL_READ_TOOL_NAMES = new Set([
  "approve",
  "ask",
  "checkpoint",
  "learn",
  "memory_edit",
  "retain",
  "rewind",
  "rewrite",
  "todo",
  "vibe_kill",
  "yield",
]);

/**
 * Map one proposed OMP action onto the paper's PRM sample budget.
 *
 * OMP's argument-aware approval tier supplies execution risk and the audited
 * effect adapter above separates read-only evidence gathering from read-tier
 * control/state commits. A batch is an observation only when every emitted
 * call resolves to `read` and every tool is a known observation. Missing tools,
 * unclassified read tools, malformed tool-use messages and approval failures
 * conservatively become consequential checkpoints. Terminal responses are also
 * checkpoints because they commit the agent to yielding its final result.
 */
export function classifyProcessAction(
  message: AssistantMessage,
  tools: Context["tools"],
): ProcessActionClassification {
  const toolCalls = Array.isArray(message.content)
    ? message.content.filter((block): block is ToolCall => block.type === "toolCall")
    : [];
  if (toolCalls.length === 0) {
    return {
      scope: "checkpoint",
      reason: message.stopReason === "toolUse" ? "missing_tool_call" : "terminal_response",
      toolTiers: [],
    };
  }

  const toolTiers: ToolTier[] = [];
  let hasStatefulReadTool = false;
  let hasUnclassifiedReadTool = false;
  for (const toolCall of toolCalls) {
    const tool = tools?.find((candidate) =>
      candidate.name === toolCall.name || candidate.customWireName === toolCall.name
    ) as RuntimeApprovalTool | undefined;
    if (!tool) {
      return { scope: "checkpoint", reason: "unknown_tool", toolTiers };
    }
    let tier: ToolTier;
    try {
      tier = resolveToolTier(tool, toolCall.arguments);
    } catch {
      return { scope: "checkpoint", reason: "approval_error", toolTiers };
    }
    toolTiers.push(tier);
    if (tier !== "read") continue;

    const effect = classifyReadTierEffect(tool, toolCall.name, toolCall.arguments);
    if (effect === "stateful") hasStatefulReadTool = true;
    else if (effect === "unclassified") hasUnclassifiedReadTool = true;
  }
  if (toolTiers.includes("exec")) {
    return { scope: "checkpoint", reason: "exec_tool", toolTiers };
  }
  if (toolTiers.includes("write")) {
    return { scope: "checkpoint", reason: "write_tool", toolTiers };
  }
  if (hasStatefulReadTool) {
    return { scope: "checkpoint", reason: "stateful_read_tool", toolTiers };
  }
  if (hasUnclassifiedReadTool) {
    return { scope: "checkpoint", reason: "unclassified_read_tool", toolTiers };
  }
  return { scope: "observation", reason: "read_only_tools", toolTiers };
}

function classifyReadTierEffect(
  tool: RuntimeApprovalTool,
  emittedName: string,
  args: unknown,
): "observation" | "stateful" | "unclassified" {
  const names = [tool.name, tool.customWireName, emittedName]
    .filter((name): name is string => typeof name === "string")
    .map((name) => name.trim().toLowerCase());
  if (names.some((name) => OMP_STATEFUL_READ_TOOL_NAMES.has(name))) return "stateful";
  if (names.some((name) => OMP_OBSERVATION_TOOL_NAMES.has(name))) return "observation";
  if (names.includes("hub")) {
    const op = typeof args === "object" && args !== null && "op" in args
      ? String((args as { op?: unknown }).op).toLowerCase()
      : "";
    return op === "cancel" || op === "send" ? "stateful" : "observation";
  }
  return "unclassified";
}

/** Classify the winning action for telemetry. */
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
  preferReasoningCompatibility = false,
): Promise<CandidateResult> {
  let transientAttempts = 0;
  for (;;) {
    const candidateRequest = candidateStreamOptions(
      state,
      streamOptions,
      signal,
      preferReasoningCompatibility,
    );
    try {
      const stream = (state.streamSimpleFn ?? streamSimple)(
        state.originalModel,
        cloneContext(context),
        candidateRequest.options,
      );
      const message = await stream.result();
      if (message.stopReason === "error" || message.stopReason === "aborted") {
        throw candidateGenerationError(message, index);
      }
      return { index, message };
    } catch (error) {
      if (signal.aborted || isAbortError(error)) throw error;
      if (!candidateRequest.mandatoryReasoning && isMandatoryReasoningRejection(error)) {
        // Dynamic endpoints can tighten capabilities before OMP's model
        // catalog catches up. Learn the requirement once for this wrapper and
        // retry without lowering any explicit user-selected effort.
        state.mandatoryReasoningRequired = true;
        continue;
      }
      if (
        transientAttempts >= CANDIDATE_TRANSIENT_RETRIES ||
        !isProviderRetryableError(error, { provider: state.originalModel.provider })
      ) {
        throw error;
      }
      transientAttempts += 1;
      const delayMs = state.candidateRetryDelayMs ??
        CANDIDATE_RETRY_BASE_DELAY_MS + index * CANDIDATE_RETRY_STAGGER_MS;
      await waitForCandidateRetry(delayMs, signal);
    }
  }
}

function candidateStreamOptions(
  state: AutoVerifierState,
  streamOptions: SimpleStreamOptions,
  signal: AbortSignal,
  preferReasoningCompatibility: boolean,
): { options: SimpleStreamOptions; mandatoryReasoning: boolean } {
  const mandatoryReasoning =
    state.mandatoryReasoningRequired === true ||
    state.originalModel.thinking?.requiresEffort === true ||
    (preferReasoningCompatibility && state.originalModel.reasoning);
  const minimumEffort = state.originalModel.thinking?.efforts[0];
  return {
    mandatoryReasoning,
    options: {
      ...streamOptions,
      ...(mandatoryReasoning
        ? {
            reasoning: streamOptions.reasoning ?? minimumEffort,
            disableReasoning: undefined,
            forceReasoningOff: undefined,
          }
        : {}),
      apiKey: state.apiKeyResolver,
      signal,
      temperature: streamOptions.temperature ?? 1,
      // Candidates must be independent, not chained to the OMP conversation, so
      // server-side turn chaining is disabled. But sessionId / promptCacheKey /
      // providerSessionState are OMP's default-call identity: keep them so all
      // candidates share the full-context prefix cache instead of paying one
      // independent uncached-prefix write per candidate.
      statefulResponses: false,
      // These are side-channel requests. The primary agent loop owns cache keep-
      // alive requests, and candidate generation must always produce an action.
      anthropicCacheRefresh: false,
      anthropicCacheRefreshRequest: false,
    },
  };
}

function isMandatoryReasoningRejection(error: unknown): boolean {
  const status = typeof error === "object" && error !== null && "status" in error
    ? (error as { status?: unknown }).status
    : undefined;
  const message = errorMessage(error);
  return (status === 400 || /\b400\b/.test(message)) && MANDATORY_REASONING_REJECTION.test(message);
}

function candidateGenerationError(message: AssistantMessage, index: number): Error {
  const error = new Error(message.errorMessage ?? "Candidate " + index + " failed") as Error & {
    status?: number;
  };
  if (message.errorStatus !== undefined) error.status = message.errorStatus;
  return error;
}

function waitForCandidateRetry(delayMs: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason ?? abortReason());
  return new Promise((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      reject(signal.reason ?? abortReason());
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, delayMs);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
  });
}

interface GatheredCandidates {
  results: Array<PromiseSettledResult<CandidateResult>>;
  /** Candidates still in flight when a strict majority became guaranteed. */
  discardedCandidates: number;
}

/**
 * Race the candidate fan-out and finish once a strict action majority is
 * mathematically guaranteed (some action identity has > N/2 confirmations).
 * The majority verdict cannot change with the remaining candidates, so the
 * replayed winner is identical to waiting for all N — this only skips the
 * slowest candidate's tail latency and its generation cost. Remaining
 * in-flight requests are aborted through a private signal (never the
 * caller's), and their settled results are swallowed so no rejection becomes
 * unhandled. `discardedCandidates` reports how many were cancelled.
 */
async function gatherCandidates(
  generate: (index: number, signal: AbortSignal) => Promise<CandidateResult>,
  count: number,
  external: AbortSignal,
  initial: ReadonlyArray<PromiseSettledResult<CandidateResult> | undefined> = [],
): Promise<GatheredCandidates> {
  const discard = new AbortController();
  const merged = mergeAbortSignals(external, discard.signal);
  const results = new Array<PromiseSettledResult<CandidateResult>>(count);
  const promises = new Array<Promise<CandidateResult> | undefined>(count);
  let seeded = 0;
  for (let index = 0; index < Math.min(initial.length, count); index += 1) {
    const result = initial[index];
    if (!result) continue;
    results[index] = result;
    seeded += 1;
  }
  for (let index = 0; index < count; index += 1) {
    if (results[index]) continue;
    promises[index] = generate(index, merged.signal);
  }
  let pending = count - seeded;
  let resolved = false;

  return await new Promise<GatheredCandidates>((resolveOuter) => {
    const finish = (): void => {
      if (resolved) return;
      resolved = true;
      const discardedReason = discard.signal.reason ?? abortReason();
      const snapshot: Array<PromiseSettledResult<CandidateResult>> = Array.from(
        { length: count },
        (_, index) => {
          const result = results[index];
          if (result) return result;
          // Late candidates after a short circuit: swallow their outcome.
          void promises[index]?.then(() => undefined, () => undefined);
          return { status: "rejected", reason: discardedReason };
        },
      );
      resolveOuter({ results: snapshot, discardedCandidates: pending });
    };
    const afterSettle = (): void => {
      if (resolved) return;
      if (pending > 0) {
        // A strict majority could be decided by the candidates that already
        // finished; the rest cannot change it. Count only fulfilled actions.
        const counts = new Map<string, number>();
        let confirmed = 0;
        for (const result of results) {
          if (!result || result.status !== "fulfilled") continue;
          confirmed += 1;
          const identity = serializeActionIdentity(result.value.message);
          counts.set(identity, (counts.get(identity) ?? 0) + 1);
        }
        for (const tally of counts.values()) {
          if (tally > count / 2 && confirmed < count) {
            discard.abort(new DOMException(
              "Strict action majority reached before all candidates finished",
              "AbortError",
            ));
            finish();
            return;
          }
        }
      }
      if (pending === 0) finish();
    };
    if (pending === 0) {
      finish();
      return;
    }
    for (const [index, promise] of promises.entries()) {
      if (!promise) continue;
      promise.then(
        (value) => {
          results[index] = { status: "fulfilled", value };
          pending -= 1;
          afterSettle();
        },
        (reason) => {
          results[index] = { status: "rejected", reason };
          pending -= 1;
          afterSettle();
        },
      );
    }
  }).finally(() => merged.dispose());
}

/** Combine two abort signals into once; call `dispose()` to release listeners. */
function mergeAbortSignals(
  a: AbortSignal,
  b: AbortSignal,
): { signal: AbortSignal; dispose(): void } {
  if (a.aborted) return { signal: a, dispose: () => undefined };
  if (b.aborted) return { signal: b, dispose: () => undefined };
  const controller = new AbortController();
  const onA = () => controller.abort(a.reason);
  const onB = () => controller.abort(b.reason);
  a.addEventListener("abort", onA, { once: true });
  b.addEventListener("abort", onB, { once: true });
  return {
    signal: controller.signal,
    dispose() {
      a.removeEventListener("abort", onA);
      b.removeEventListener("abort", onB);
    },
  };
}
/** TurboAgent's exact-action majority shortcut (§ verifier.py). */
function exactActionMajority(
  actions: readonly string[],
): { index: number; scores: number[] } | undefined {
  const counts = new Map<string, number>();
  for (const action of actions) counts.set(action, (counts.get(action) ?? 0) + 1);
  let majorityAction: string | undefined;
  let majorityCount = 0;
  for (const [action, count] of counts) {
    if (count > majorityCount) {
      majorityAction = action;
      majorityCount = count;
    }
  }
  if (majorityAction === undefined || majorityCount <= actions.length / 2) return undefined;
  return {
    index: actions.indexOf(majorityAction),
    scores: actions.map((action) => action === majorityAction ? 1 : 0),
  };
}

/**
 * TurboAgent action identity: visible text plus tool name/arguments, with
 * provider-generated call IDs treated as transport metadata. This string is
 * intentionally unbounded so a shared long prefix cannot create a false exact
 * majority; bounded traces remain a verifier-prompt concern.
 */
function serializeActionIdentity(message: AssistantMessage): string {
  if (!message || !Array.isArray(message.content)) return "(empty response)";
  const parts: string[] = [];
  for (const block of message.content) {
    if (!block || typeof block !== "object") continue;
    const value = block as unknown as Record<string, unknown>;
    if (value.type === "text" && typeof value.text === "string" && value.text) {
      parts.push(value.text);
    } else if (value.type === "toolCall") {
      const toolCall = value as unknown as ToolCall;
      parts.push(
        "[tool_call: " + toolCall.name + "(" + compactJson(toolCall.arguments) + ")]",
      );
    } else if (
      value.type === "image" && typeof value.mimeType === "string" &&
      typeof value.data === "string"
    ) {
      parts.push("[image: " + value.mimeType + ":" + value.data + "]");
    }
  }
  return parts.join("\n") || "(empty response)";
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
 * the most recent user message (the request being answered). TurboAgent passes
 * the full request history, so any remaining budget is backfilled with the
 * most recent prior conversation. Shared images are carried separately to the
 * multimodal verifier and very long requests are capped so the prompt prefix
 * stays stable across repeated comparisons.
 *
 * The verifier's criteria instruct it to treat OBSERVED tool results as ground
 * truth, so the problem also carries a recency-bounded chronological slice of
 * the trajectory since that request: visible assistant actions and tool outputs
 * (per-block truncated), still excluding system/developer prompts, reasoning,
 * tool schemas, and image payloads. Conversation images travel separately in
 * chronological order. Prior context, the task, separators, and current
 * evidence share the 16k problem budget — the hard cap on every pairwise
 * prompt.
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
  const images = sharedContextImages(messages, 0);
  if (!task) return { problem: "(no user request captured)", images };
  const currentTask = truncateWithMarker(task, PROBLEM_MAX_CHARS, "\n... [task truncated]");

  const separator = "\n\n";
  const evidenceBudget = PROBLEM_MAX_CHARS - currentTask.length - separator.length;
  const currentEvidence = evidenceBudget > 0
    ? recentTrajectoryEvidence(messages, taskIndex, evidenceBudget)
    : "";
  const used = currentTask.length + (currentEvidence ? separator.length + currentEvidence.length : 0);
  const priorBudget = PROBLEM_MAX_CHARS - used - separator.length;
  const priorContext = taskIndex > 0 && priorBudget > 0
    ? recentTrajectoryEvidence(messages, -1, priorBudget, taskIndex)
    : "";
  const problem = [priorContext, currentTask, currentEvidence].filter(Boolean).join(separator);
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
function sharedContextImages(messages: Message[], startIndex: number): ImageContent[] {
  const images: ImageContent[] = [];
  for (let index = Math.max(0, startIndex); index < messages.length; index += 1) {
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
  beforeIndex = messages.length,
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

  for (let index = Math.min(messages.length, beforeIndex) - 1; index > afterIndex; index -= 1) {
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
      if (messageParts.length) append("[assistant]\n" + messageParts.join("\n"));
    } else if (message.role === "user") {
      const messageParts: string[] = [];
      const content = message.content;
      if (typeof content === "string") {
        const text = content.trim();
        if (text) messageParts.push(truncateBlock(text));
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!block || typeof block !== "object") continue;
          const value = block as unknown as Record<string, unknown>;
          if (value.type === "text" && typeof value.text === "string") {
            const text = value.text.trim();
            if (text) messageParts.push(truncateBlock(text));
          } else if (value.type === "image") {
            messageParts.push("[image attached]");
          }
        }
      }
      if (messageParts.length) append("[user]\n" + messageParts.join("\n"));
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

function reportPhase(state: AutoVerifierState, event: AutoVerifierPhaseEvent): void {
  try {
    state.onPhase?.(event);
  } catch {
    // A UI phase callback cannot change the provider result.
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
