/** OMP extension entry point for process-reward checkpoint verification. */

import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { getPluginSettings } from "@oh-my-pi/pi-coding-agent/extensibility/plugins/loader";
import { extractExplicitThinkingSelector } from "@oh-my-pi/pi-coding-agent/config/model-resolver";
import type { Model } from "@oh-my-pi/pi-ai";
import { createHash } from "node:crypto";
import { join } from "node:path";
import {
  AUTO_CANDIDATE_COUNT,
  AUTO_SELECTION_DEFAULTS,
  createWrappedProvider,
  normalizeCandidateCount,
  normalizeEvaluations,
  normalizePivots,
  type AutoVerifierDegradedEvent,
  type AutoVerifierPhaseEvent,
} from "./auto.ts";
import {
  createVerifierClient,
  isVerifierSupportedApi,
  isVerifierLogprobsUnsupportedError,
  VerifierClient,
  VerifierLogprobsUnsupportedError,
} from "./client.ts";

const PLUGIN_NAME = "omp-llm-verifier";
const WRAPPER_KEY = "omp-llm-verifier-internal";
const WRAPPER_API_PREFIX = "omp-llm-verifier-api-";
const MODEL_REBIND_INTERVAL_MS = 500;
const CAPABILITY_PROBE_RETRY_MS = 60_000;

class VerifierCapabilityProbeError extends Error {
  constructor(message: string, cause: unknown) {
    super(message, { cause });
    this.name = "VerifierCapabilityProbeError";
  }
}

export interface VerifierPluginSettings {
  enabled: boolean;
  candidateCount: number;
  /** Repeated verifications per criterion (paper §4.2); online default K=1. */
  nEvaluations: number;
  /** PPT pivot count k (paper §3.2); TurboAgent online default k=2. */
  pivots: number;
  /** OMP model selector for the verifier (e.g. "deepseek/deepseek-v4-flash:high"); empty follows the session default model. */
  verifierModel?: string;
}

export interface VerificationBinding {
  sourceKey: string;
  originalModel: Model;
  wrapperModel: Model;
  providerName: string;
}

export interface AutomaticVerificationRuntime {
  bindings: Map<string, VerificationBinding>;
  capabilityErrors: Map<string, string>;
  probeErrors: Map<string, { message: string; retryAt: number }>;
  probeVerifier?: (client: VerifierClient, model: Model) => Promise<void>;
  now: () => number;
  activeSourceKey?: string;
  inFlight?: Promise<VerificationBinding>;
  inFlightSourceKey?: string;
  generation: number;
}

export function degradedWarningMessage(event: AutoVerifierDegradedEvent): string {
  const candidateSummary = event.successfulCandidates + "/" + event.candidateCount +
    " candidates succeeded";
  if (event.reason === "insufficient_candidates") {
    return "LLM-as-a-Verifier " + candidateSummary +
      "; returned the only successful candidate.";
  }
  if (event.reason === "verification_error") {
    const detail = event.error ? " Error: " + event.error : "";
    return "LLM-as-a-Verifier PPT verification failed after " + candidateSummary +
      "; returned the earliest successful candidate." + detail;
  }
  const sources = event.scoreSources;
  if (sources && sources.logprobs > 0) {
    const totalScores = sources.logprobs + sources.textFallback +
      sources.neutralTie + sources.unknown;
    const totalComparisons = Math.ceil(totalScores / 2);
    if (sources.neutralTie > 0 || sources.unknown > 0) {
      const failedComparisons = Math.ceil((sources.neutralTie + sources.unknown) / 2);
      return "LLM-as-a-Verifier " + candidateSummary + "; " +
        failedComparisons + "/" + totalComparisons +
        " PPT comparisons failed before complete score tags, while the configured verifier returned " +
        "valid token logprobs for " + sources.logprobs + "/" + totalScores +
        " score tags. Selection used neutral ties for failed comparisons; repeated occurrences warrant " +
        "checking the verifier output budget and provider errors.";
    }
    return "LLM-as-a-Verifier " + candidateSummary + "; the configured verifier returned valid token " +
      "logprobs for " + sources.logprobs + "/" + totalScores +
      " score tags, with literal-text fallback for the remainder. Selection completed with mixed score evidence.";
  }
  return "LLM-as-a-Verifier PPT scoring lacked complete token-logprob evidence although " +
    candidateSummary +
    "; selected with neutral-tie fallback. Pin a logprobs-capable verifier with " +
    "`omp plugin config set omp-llm-verifier verifierModel <provider/model>`.";
}

export function automaticVerificationWorkingMessage(
  event: AutoVerifierPhaseEvent,
): string | undefined {
  if (event.phase === "generating_candidates") {
    return "Expanding a consequential action to " + event.candidateCount + " candidates…";
  }
  if (event.phase === "verifying_candidates") {
    return "Verifying " + (event.successfulCandidates ?? event.candidateCount) +
      " candidate actions with PPT…";
  }
  return undefined;
}

export default function verifierExtension(pi: ExtensionAPI): void {
  pi.setLabel("LLM-as-a-Verifier");
  pi.registerFlag("llm-verifier", {
    type: "boolean",
    description: "Enable automatic candidate verification for the current OMP session",
    default: false,
  });

  const runtime = createAutomaticVerificationRuntime({
    probeVerifier: (client) => client.probeLogprobs(),
  });
  let sessionEnabled = false;
  let nEvaluations: number = AUTO_SELECTION_DEFAULTS.nEvaluations;
  let pivots: number = AUTO_SELECTION_DEFAULTS.pivots;
  let candidateCount = AUTO_CANDIDATE_COUNT;
  let verifierModel: string | undefined;
  let lastRebindError = "";
  let lastRebindErrorAt = 0;
  let lastUiWarning = "";
  const notifyWarning = (
    ctx: ExtensionContext,
    error: unknown,
    event = "runtime_warning",
  ): void => {
    const message = formatVerifierError(error, resolveSourceModel(ctx, runtime));
    if (message === lastUiWarning) return;
    lastUiWarning = message;
    pi.logger.warn("LLM-as-a-Verifier " + event, {
      component: PLUGIN_NAME,
      event,
      error: message,
    });
    ctx.ui.notify(message, "warning");
  };

  pi.on("session_start", async (_event, ctx) => {
    try {
      const settings = await getPluginSettings(PLUGIN_NAME, ctx.cwd);
      const pluginSettings = resolvePluginSettings(settings);
      sessionEnabled = pluginSettings.enabled || pi.getFlag("llm-verifier") === true;
      candidateCount = pluginSettings.candidateCount;
      nEvaluations = pluginSettings.nEvaluations;
      pivots = pluginSettings.pivots;
      verifierModel = pluginSettings.verifierModel;
      if (!sessionEnabled) return;
      ctx.setInterval(() => {
        if (!sessionEnabled || !ctx.isIdle()) return;
        void ensureAutomaticVerification(pi, ctx, runtime, candidateCount, verifierModel, nEvaluations, pivots)
          .then(() => {
            lastRebindError = "";
            lastRebindErrorAt = 0;
            lastUiWarning = "";
          })
          .catch((error) => {
            const message = error instanceof Error ? error.message : String(error);
            const now = Date.now();
            if (message === lastRebindError && now - lastRebindErrorAt < 5_000) return;
            lastRebindError = message;
            lastRebindErrorAt = now;
            notifyWarning(ctx, error, "model_rebind_failed");
          });
      }, MODEL_REBIND_INTERVAL_MS);
      await ensureAutomaticVerification(pi, ctx, runtime, candidateCount, verifierModel, nEvaluations, pivots);
      ctx.ui.notify(
        "LLM-as-a-Verifier enabled: audited observations use one sample; state-changing and terminal actions expand to verified candidates.",
        "info",
      );
    } catch (error) {
      notifyWarning(ctx, error, "startup_verifier_unavailable");
    }
  });

  // OMP has no model-changed extension event. before_agent_start is awaited
  // before the agent loop builds its provider context, so it is the reliable
  // seam for rebinding after modelRoles.default changes.
  pi.on("before_agent_start", async (_event, ctx) => {
    if (!sessionEnabled) return;
    const previousSourceKey = runtime.activeSourceKey;
    try {
      const binding = await ensureAutomaticVerification(pi, ctx, runtime, candidateCount, verifierModel, nEvaluations, pivots);
      if (previousSourceKey && previousSourceKey !== binding.sourceKey) {
        ctx.ui.notify("LLM-as-a-Verifier followed the OMP default model switch and rebound successfully.", "info");
      }
    } catch (error) {
      notifyWarning(ctx, error, "model_switch_verifier_unavailable");
    }
  });
}

export function createAutomaticVerificationRuntime(options: {
  probeVerifier?: (client: VerifierClient, model: Model) => Promise<void>;
  now?: () => number;
} = {}): AutomaticVerificationRuntime {
  return {
    bindings: new Map(),
    capabilityErrors: new Map(),
    probeErrors: new Map(),
    probeVerifier: options.probeVerifier,
    now: options.now ?? Date.now,
    generation: 0,
  };
}

export async function ensureAutomaticVerification(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  runtime: AutomaticVerificationRuntime,
  candidateCount = AUTO_CANDIDATE_COUNT,
  verifierSelector?: string,
  nEvaluations: number = AUTO_SELECTION_DEFAULTS.nEvaluations,
  pivots: number = AUTO_SELECTION_DEFAULTS.pivots,
): Promise<VerificationBinding> {
  while (true) {
    const originalModel = resolveSourceModel(ctx, runtime);
    if (!originalModel) throw new Error("OMP modelRoles.default did not resolve to an available model.");
    // The paper scores the agent's candidates with a verifier model of its
    // own (Gemini/DeepSeek in the reference runs), so the logprobs-capable
    // API requirement applies to the verifier model, not the agent's model.
    const verifierModel = verifierSelector ? ctx.models.resolve(verifierSelector) : originalModel;
    if (verifierSelector && !verifierModel) {
      throw new Error(
        "LLM-as-a-Verifier verifier model " + JSON.stringify(verifierSelector) +
        " did not resolve to an available OMP model; check the omp-llm-verifier verifierModel setting.",
      );
    }
    if (!verifierModel) throw new Error("OMP modelRoles.default did not resolve to an available model.");
    if (!isVerifierSupportedApi(verifierModel.api)) {
      throw new VerifierLogprobsUnsupportedError(unsupportedModelMessage(verifierModel, {
        verifierSelector,
        available: potentialVerifierModels(ctx, verifierModel),
        reason: "uses " + verifierModel.api +
          ", which does not expose the OpenAI-compatible token logprobs required by LLM-as-a-Verifier",
      }), { retryable: false });
    }

    const sessionId = ctx.sessionManager.getSessionId();
    const sourceKey = getSourceKey(pi, ctx, runtime, sessionId, verifierSelector);
    if (!sourceKey) throw new Error("OMP modelRoles.default did not resolve to an available model.");
    const providerName = sourceKey;
    const capabilityError = runtime.capabilityErrors.get(sourceKey);
    if (capabilityError) {
      throw new VerifierLogprobsUnsupportedError(capabilityError);
    }
    const probeError = runtime.probeErrors.get(sourceKey);
    if (probeError) {
      if (runtime.now() < probeError.retryAt) throw new Error(probeError.message);
      runtime.probeErrors.delete(sourceKey);
    }
    const activeBinding = runtime.bindings.get(sourceKey);
    if (
      runtime.activeSourceKey === sourceKey &&
      activeBinding &&
      ctx.model &&
      ctx.model.provider === activeBinding.wrapperModel.provider &&
      ctx.model.id === activeBinding.wrapperModel.id
    ) {
      return activeBinding;
    }
    const inFlight = runtime.inFlight;
    if (inFlight && runtime.inFlightSourceKey === sourceKey) {
      const waitingGeneration = runtime.generation;
      const binding = await inFlight;
      if (runtime.generation === waitingGeneration && getSourceKey(pi, ctx, runtime, sessionId, verifierSelector) === sourceKey) {
        return binding;
      }
      continue;
    }

    const previous = runtime.inFlight;
    const generation = ++runtime.generation;
    const task = (previous ? previous.catch(() => undefined) : Promise.resolve()).then(async () => {
      let binding = runtime.bindings.get(sourceKey);
      if (!binding) {
        try {
          binding = await createVerificationBinding(
            pi,
            ctx,
            originalModel,
            verifierModel,
            providerName,
            sourceKey,
            candidateCount,
            runtime.probeVerifier,
            verifierSelector,
            nEvaluations,
            pivots,
          );
        } catch (error) {
          if (isVerifierLogprobsUnsupportedError(error)) {
            const message = unsupportedModelMessage(verifierModel, {
              verifierSelector,
              available: potentialVerifierModels(ctx, verifierModel),
            });
            runtime.capabilityErrors.set(sourceKey, message);
            throw new VerifierLogprobsUnsupportedError(message);
          }
          if (!(error instanceof VerifierCapabilityProbeError)) throw error;
          const detail = error instanceof Error ? error.message : String(error);
          const message = "LLM-as-a-Verifier capability probe failed for " +
            verifierModel.provider + "/" + verifierModel.id + ": " + detail +
            ". The original agent model remains active; capability probing will retry after 60 seconds. " +
            verifierRecoveryHint(verifierSelector);
          runtime.probeErrors.set(sourceKey, {
            message,
            retryAt: runtime.now() + CAPABILITY_PROBE_RETRY_MS,
          });
          throw new Error(message, { cause: error });
        }
        runtime.bindings.set(sourceKey, binding);
      }

      if (generation !== runtime.generation) {
        return binding;
      }
      if (getSourceKey(pi, ctx, runtime, sessionId, verifierSelector) !== sourceKey) {
        return binding;
      }

      const currentModel = ctx.model;
      if (currentModel && !isWrapperModel(currentModel) && !sameModelIdentity(currentModel, originalModel)) {
        return binding;
      }
      if (
        !currentModel ||
        currentModel.provider !== binding.wrapperModel.provider ||
        currentModel.id !== binding.wrapperModel.id
      ) {
        const thinkingLevel = pi.getThinkingLevel();
        if (!await pi.setModel(binding.wrapperModel)) {
          throw new Error("OMP could not switch to the automatic-verification wrapper model.");
        }
        if (thinkingLevel !== undefined) pi.setThinkingLevel(thinkingLevel);
      }
      if (generation !== runtime.generation || getSourceKey(pi, ctx, runtime, sessionId, verifierSelector) !== sourceKey) {
        return binding;
      }
      runtime.activeSourceKey = sourceKey;
      return binding;
    });
    runtime.inFlight = task;
    runtime.inFlightSourceKey = sourceKey;
    try {
      const binding = await task;
      if (generation === runtime.generation && getSourceKey(pi, ctx, runtime, sessionId, verifierSelector) === sourceKey) {
        return binding;
      }
    } finally {
      if (runtime.inFlight === task) {
        runtime.inFlight = undefined;
        runtime.inFlightSourceKey = undefined;
      }
    }
  }
}

async function createVerificationBinding(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  originalModel: Model,
  verifierModel: Model,
  providerName: string,
  sourceKey: string,
  candidateCount: number,
  probeVerifier?: (client: VerifierClient, model: Model) => Promise<void>,
  verifierSelector?: string,
  nEvaluations: number = AUTO_SELECTION_DEFAULTS.nEvaluations,
  pivots: number = AUTO_SELECTION_DEFAULTS.pivots,
): Promise<VerificationBinding> {
  const registry = ctx.modelRegistry as ExtensionContext["modelRegistry"] & {
    getProviderHeaders?: (provider: string) => Record<string, string> | undefined;
  };
  const providerHeaders = registry.getProviderHeaders?.(originalModel.provider);
  const candidateModel = providerHeaders
    ? { ...originalModel, headers: { ...providerHeaders, ...originalModel.headers } }
    : originalModel;
  const resolver = ctx.modelRegistry.resolver(originalModel, ctx.sessionManager.getSessionId());
  const verifierClient = await createDefaultVerifierClient(pi, {
    models: ctx.models,
    modelRegistry: ctx.modelRegistry,
    sessionManager: ctx.sessionManager,
  }, verifierModel, verifierSelector);
  if (probeVerifier) {
    try {
      await probeVerifier(verifierClient, verifierModel);
    } catch (error) {
      pi.logger.warn("LLM-as-a-Verifier capability probe failed", {
        component: PLUGIN_NAME,
        event: "capability_probe_failed",
        model: verifierModel.provider + "/" + verifierModel.id,
        error: error instanceof Error ? error.message : String(error),
      });
      if (isVerifierLogprobsUnsupportedError(error)) throw error;
      throw new VerifierCapabilityProbeError(
        error instanceof Error ? error.message : String(error),
        error,
      );
    }
  }
  const wrapperApi = wrapperApiName(providerName);
  const modelId = "default";
  const config = {
    baseUrl: originalModel.baseUrl,
    api: wrapperApi,
    apiKey: WRAPPER_KEY,
    authHeader: false,
    streamSimple: createWrappedProvider(
      {
        originalModel: candidateModel,
        verifierClient,
        apiKeyResolver: resolver,
        cacheFile: ctx.cwd ? join(ctx.cwd, ".omp-llm-verifier-cache.json") : undefined,
        onDegraded: (event) => {
          pi.logger.warn("LLM-as-a-Verifier degraded", {
            component: PLUGIN_NAME,
            event: "degraded",
            ...event,
          });
          ctx.ui.notify(degradedWarningMessage(event), "warning");
        },
        onPhase: (event) => {
          ctx.ui.setWorkingMessage(automaticVerificationWorkingMessage(event));
        },
        onDecision: (decision) => {
          pi.logger.info("LLM-as-a-Verifier decision", {
            component: PLUGIN_NAME,
            event: "decision",
            ...decision,
            winnerScore: decision.winnerScore === undefined ? undefined : Number(decision.winnerScore.toFixed(4)),
            scores: decision.scores?.map((score) => Number(score.toFixed(4))),
          });
        },
      },
      { candidateCount, nEvaluations, pivots },
    ),
    models: [{
      id: modelId,
      name: "LLM-as-a-Verifier (" + originalModel.name + ")",
      api: wrapperApi,
      reasoning: originalModel.reasoning,
      thinking: originalModel.thinking,
      input: [...originalModel.input],
      supportsTools: originalModel.supportsTools,
      cost: {
        input: originalModel.cost.input,
        output: originalModel.cost.output,
        cacheRead: originalModel.cost.cacheRead,
        cacheWrite: originalModel.cost.cacheWrite,
      },
      contextWindow: originalModel.contextWindow,
      maxTokens: originalModel.maxTokens,
      headers: candidateModel.headers,
      compat: candidateModel.compat,
    }],
  } as unknown as Parameters<ExtensionAPI["registerProvider"]>[1];
  pi.registerProvider(providerName, config);

  let wrapperModel = ctx.models.resolve(providerName + "/" + modelId);
  if (!wrapperModel) {
    await ctx.modelRegistry.refreshRuntimeProviders();
    wrapperModel = ctx.models.resolve(providerName + "/" + modelId);
  }
  if (!wrapperModel) throw new Error("Could not register the automatic-verification wrapper model.");
  return { sourceKey, originalModel, wrapperModel, providerName };
}

export function resolvePluginSettings(
  settings: Record<string, unknown>,
): VerifierPluginSettings {
  return {
    enabled: settings.enabled === true,
    candidateCount: normalizeCandidateCount(settings.candidateCount),
    nEvaluations: normalizeEvaluations(settings.nEvaluations),
    pivots: normalizePivots(settings.pivots),
    verifierModel:
      typeof settings.verifierModel === "string" && settings.verifierModel.trim()
        ? settings.verifierModel.trim()
        : undefined,
  };
}

export async function createDefaultVerifierClient(
  pi: ExtensionAPI,
  ctx: Pick<ExtensionContext, "models" | "modelRegistry"> & {
    sessionManager?: ExtensionContext["sessionManager"];
  },
  model?: Model,
  verifierSelector?: string,
) {
  const settings = pi.pi.settings;
  return createVerifierClient({
    ...ctx,
    model,
    sessionId: ctx.sessionManager?.getSessionId(),
    defaultThinkingLevel: extractExplicitThinkingSelector(
      verifierSelector ?? settings.getModelRole("default"),
      settings,
    ),
  });
}

function wrapperProviderName(model: Model, sessionId: string, selector?: string, verifierSelector?: string): string {
  const identity = JSON.stringify({
    provider: model.provider,
    id: model.id,
    api: model.api,
    baseUrl: model.baseUrl,
    requestModelId: model.requestModelId,
    selector,
    verifierSelector,
    sessionId,
  });
  return "omp-llm-verifier-" + createHash("sha256").update(identity).digest("hex").slice(0, 12);
}

function wrapperApiName(providerName: string): string {
  return WRAPPER_API_PREFIX + providerName.slice("omp-llm-verifier-".length);
}

function isWrapperModel(model: Model): boolean {
  return model.provider.startsWith("omp-llm-verifier-");
}

function resolveSourceModel(
  ctx: ExtensionContext,
  runtime: AutomaticVerificationRuntime,
): Model | undefined {
  const current = ctx.model;
  if (current && !isWrapperModel(current)) return current;

  if (current && isWrapperModel(current) && runtime.activeSourceKey) {
    const binding = runtime.bindings.get(runtime.activeSourceKey);
    if (
      binding &&
      binding.wrapperModel.provider === current.provider &&
      binding.wrapperModel.id === current.id
    ) {
      return binding.originalModel;
    }
  }

  if (current && isWrapperModel(current)) {
    for (const binding of runtime.bindings.values()) {
      if (
        binding.wrapperModel.provider === current.provider &&
        binding.wrapperModel.id === current.id
      ) {
        return binding.originalModel;
      }
    }
  }

  const configured = ctx.models.resolve("@default");
  return configured && !isWrapperModel(configured) ? configured : undefined;
}

function getSourceKey(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  runtime: AutomaticVerificationRuntime,
  sessionId: string,
  verifierSelector?: string,
): string | undefined {
  const model = resolveSourceModel(ctx, runtime);
  if (!model) return undefined;
  return wrapperProviderName(model, sessionId, getDefaultModelSelector(pi), verifierSelector);
}

function sameModelIdentity(left: Model, right: Model): boolean {
  return left.provider === right.provider &&
    left.id === right.id &&
    left.baseUrl === right.baseUrl &&
    left.requestModelId === right.requestModelId;
}

function getDefaultModelSelector(pi: ExtensionAPI): string | undefined {
  const selector = pi.pi.settings.getModelRole("default");
  return typeof selector === "string" ? selector : undefined;
}

/**
 * Providers whose OpenAI-compatible surface is declared openai-completions
 * but rejects the logprobs parameters the paper's verifier needs (verified
 * against Kimi for Coding today). Kept as a small denylist so the startup
 * warning never suggests a verifier that is guaranteed to fail.
 */
const LOGPROBS_HOSTILE_PROVIDERS = new Set(["kimi-code"]);

function potentialVerifierModels(ctx: ExtensionContext, excludedModel?: Model): string[] {
  if (!ctx.models || typeof ctx.models.list !== "function") return [];
  const seen = new Set<string>();
  const models: string[] = [];
  for (const candidate of ctx.models.list()) {
    if (!candidate || !isVerifierSupportedApi(candidate.api)) continue;
    if (isWrapperModel(candidate)) continue;
    if (LOGPROBS_HOSTILE_PROVIDERS.has(candidate.provider)) continue;
    if (
      excludedModel &&
      candidate.provider === excludedModel.provider &&
      candidate.id === excludedModel.id
    ) continue;
    const id = candidate.provider + "/" + candidate.id;
    if (seen.has(id)) continue;
    seen.add(id);
    models.push(id);
    if (models.length >= 5) break;
  }
  return models;
}

function unsupportedModelMessage(
  model: Model,
  options: {
    verifierSelector?: string;
    available?: string[];
    reason?: string;
  } = {},
): string {
  const subject = options.verifierSelector ? "Configured verifier model " : "OMP default model ";
  const reason = options.reason ??
    "does not provide the token logprobs required by LLM-as-a-Verifier";
  const available = options.available && options.available.length > 0
    ? " Potential verifier models configured in this session: " +
      options.available.join(", ") + "; each will be probed when selected."
    : "";
  return subject + model.provider + "/" + model.id + " " + reason + ". " +
    verifierRecoveryHint(options.verifierSelector) + available;
}

function verifierRecoveryHint(verifierSelector?: string): string {
  if (verifierSelector) {
    return "Choose one: (1) configure another logprobs-capable verifier model with " +
      "`omp plugin config set omp-llm-verifier verifierModel <provider/model>`; " +
      "(2) clear verifierModel to follow the OMP default model.";
  }
  return "Choose one: (1) switch the OMP default model to one that supports token logprobs; " +
    "(2) configure a separate verifier model with " +
    "`omp plugin config set omp-llm-verifier verifierModel <provider/model>`.";
}

function formatVerifierError(error: unknown, model?: Model): string {
  if (isVerifierLogprobsUnsupportedError(error)) {
    const detail = error instanceof Error ? error.message : "";
    if (detail) return detail;
    return model ? unsupportedModelMessage(model) :
      "LLM-as-a-Verifier cannot use the active model because it did not return token logprobs. Choose a model that supports token logprobs.";
  }
  const detail = error instanceof Error ? error.message : String(error);
  return "LLM-as-a-Verifier is unavailable: " + detail;
}
