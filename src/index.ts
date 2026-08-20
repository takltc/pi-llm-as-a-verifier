/** OMP extension entry point for request-level LLM-as-a-Verifier. */

import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { getPluginSettings } from "@oh-my-pi/pi-coding-agent/extensibility/plugins/loader";
import { extractExplicitThinkingSelector } from "@oh-my-pi/pi-coding-agent/config/model-resolver";
import type { Model } from "@oh-my-pi/pi-ai";
import { createHash } from "node:crypto";
import { join } from "node:path";
import {
  AUTO_CANDIDATE_COUNT,
  createWrappedProvider,
  normalizeCandidateCount,
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

export interface VerifierPluginSettings {
  enabled: boolean;
  candidateCount: number;
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
  probeVerifier?: (client: VerifierClient, model: Model) => Promise<void>;
  activeSourceKey?: string;
  inFlight?: Promise<VerificationBinding>;
  inFlightSourceKey?: string;
  generation: number;
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
  let candidateCount = AUTO_CANDIDATE_COUNT;
  let lastRebindError = "";
  let lastRebindErrorAt = 0;
  let lastUiWarning = "";
  const notifyWarning = (ctx: ExtensionContext, error: unknown): void => {
    const message = formatVerifierError(error, resolveSourceModel(ctx, runtime));
    if (message === lastUiWarning) return;
    lastUiWarning = message;
    console.warn("Warning: " + message);
    ctx.ui.notify(message, "warning");
  };

  pi.on("session_start", async (_event, ctx) => {
    try {
      const settings = await getPluginSettings(PLUGIN_NAME, ctx.cwd);
      const pluginSettings = resolvePluginSettings(settings);
      sessionEnabled = pluginSettings.enabled || pi.getFlag("llm-verifier") === true;
      candidateCount = pluginSettings.candidateCount;
      if (!sessionEnabled) return;
      ctx.setInterval(() => {
        if (!sessionEnabled || !ctx.isIdle()) return;
        void ensureAutomaticVerification(pi, ctx, runtime, candidateCount)
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
            console.warn(JSON.stringify({
              component: PLUGIN_NAME,
              event: "model_rebind_failed",
              error: message,
            }));
            notifyWarning(ctx, error);
          });
      }, MODEL_REBIND_INTERVAL_MS);
      await ensureAutomaticVerification(pi, ctx, runtime, candidateCount);
      ctx.ui.notify("LLM-as-a-Verifier enabled: ordinary requests now generate candidates and replay the verified winner.", "info");
    } catch (error) {
      notifyWarning(ctx, error);
    }
  });

  // OMP has no model-changed extension event. before_agent_start is awaited
  // before the agent loop builds its provider context, so it is the reliable
  // seam for rebinding after modelRoles.default changes.
  pi.on("before_agent_start", async (_event, ctx) => {
    if (!sessionEnabled) return;
    const previousSourceKey = runtime.activeSourceKey;
    try {
      const binding = await ensureAutomaticVerification(pi, ctx, runtime, candidateCount);
      if (previousSourceKey && previousSourceKey !== binding.sourceKey) {
        ctx.ui.notify("LLM-as-a-Verifier followed the OMP default model switch and rebound successfully.", "info");
      }
    } catch (error) {
      notifyWarning(ctx, error);
    }
  });
}

export function createAutomaticVerificationRuntime(options: {
  probeVerifier?: (client: VerifierClient, model: Model) => Promise<void>;
} = {}): AutomaticVerificationRuntime {
  return {
    bindings: new Map(),
    capabilityErrors: new Map(),
    probeVerifier: options.probeVerifier,
    generation: 0,
  };
}

export async function ensureAutomaticVerification(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
  runtime: AutomaticVerificationRuntime,
  candidateCount = AUTO_CANDIDATE_COUNT,
): Promise<VerificationBinding> {
  while (true) {
    const originalModel = resolveSourceModel(ctx, runtime);
    if (!originalModel) throw new Error("OMP modelRoles.default did not resolve to an available model.");
    if (!isVerifierSupportedApi(originalModel.api)) {
      throw new Error(
        "OMP default model " + originalModel.provider + "/" + originalModel.id +
        " uses " + originalModel.api + "; the verifier requires OpenAI token logprobs.",
      );
    }

    const sessionId = ctx.sessionManager.getSessionId();
    const sourceKey = getSourceKey(pi, ctx, runtime, sessionId);
    if (!sourceKey) throw new Error("OMP modelRoles.default did not resolve to an available model.");
    const providerName = sourceKey;
    const capabilityError = runtime.capabilityErrors.get(sourceKey);
    if (capabilityError) {
      throw new VerifierLogprobsUnsupportedError(capabilityError);
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
      if (runtime.generation === waitingGeneration && getSourceKey(pi, ctx, runtime, sessionId) === sourceKey) {
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
            providerName,
            sourceKey,
            candidateCount,
            runtime.probeVerifier,
          );
        } catch (error) {
          if (isVerifierLogprobsUnsupportedError(error)) {
            const message = unsupportedModelMessage(originalModel);
            runtime.capabilityErrors.set(sourceKey, message);
            throw new VerifierLogprobsUnsupportedError(message);
          }
          throw error;
        }
        runtime.bindings.set(sourceKey, binding);
      }

      if (generation !== runtime.generation) {
        return binding;
      }
      if (getSourceKey(pi, ctx, runtime, sessionId) !== sourceKey) {
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
      if (generation !== runtime.generation || getSourceKey(pi, ctx, runtime, sessionId) !== sourceKey) {
        return binding;
      }
      runtime.activeSourceKey = sourceKey;
      return binding;
    });
    runtime.inFlight = task;
    runtime.inFlightSourceKey = sourceKey;
    try {
      const binding = await task;
      if (generation === runtime.generation && getSourceKey(pi, ctx, runtime, sessionId) === sourceKey) {
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
  providerName: string,
  sourceKey: string,
  candidateCount: number,
  probeVerifier?: (client: VerifierClient, model: Model) => Promise<void>,
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
  }, originalModel);
  if (probeVerifier) {
    try {
      await probeVerifier(verifierClient, originalModel);
    } catch (error) {
      if (isVerifierLogprobsUnsupportedError(error)) throw error;
      console.warn(JSON.stringify({
        component: PLUGIN_NAME,
        event: "capability_probe_failed",
        model: originalModel.provider + "/" + originalModel.id,
        error: error instanceof Error ? error.message : String(error),
      }));
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
          console.warn(JSON.stringify({ component: PLUGIN_NAME, event: "degraded", ...event }));
          ctx.ui.notify(
            event.reason === "verification_error"
              ? "LLM-as-a-Verifier verification failed; returned the first candidate response."
              : "LLM-as-a-Verifier had too few usable candidates; returned the only successful candidate.",
            "warning",
          );
        },
      },
      { candidateCount },
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
  };
}

export async function createDefaultVerifierClient(
  pi: ExtensionAPI,
  ctx: Pick<ExtensionContext, "models" | "modelRegistry"> & {
    sessionManager?: ExtensionContext["sessionManager"];
  },
  model?: Model,
) {
  const settings = pi.pi.settings;
  return createVerifierClient({
    ...ctx,
    model,
    sessionId: ctx.sessionManager?.getSessionId(),
    defaultThinkingLevel: extractExplicitThinkingSelector(
      settings.getModelRole("default"),
      settings,
    ),
  });
}

function wrapperProviderName(model: Model, sessionId: string, selector?: string): string {
  const identity = JSON.stringify({
    provider: model.provider,
    id: model.id,
    api: model.api,
    baseUrl: model.baseUrl,
    requestModelId: model.requestModelId,
    selector,
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
): string | undefined {
  const model = resolveSourceModel(ctx, runtime);
  if (!model) return undefined;
  return wrapperProviderName(model, sessionId, getDefaultModelSelector(pi));
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

function unsupportedModelMessage(model: Model): string {
  return "LLM-as-a-Verifier cannot use " + model.provider + "/" + model.id +
    ": this model did not return token logprobs. Choose a model that supports token logprobs.";
}

function formatVerifierError(error: unknown, model?: Model): string {
  if (isVerifierLogprobsUnsupportedError(error)) {
    return model ? unsupportedModelMessage(model) :
      "LLM-as-a-Verifier cannot use the active model because it did not return token logprobs. Choose a model that supports token logprobs.";
  }
  const detail = error instanceof Error ? error.message : String(error);
  return "LLM-as-a-Verifier is unavailable: " + detail;
}
