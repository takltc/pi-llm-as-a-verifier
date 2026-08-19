/** OMP extension entry point for request-level LLM-as-a-Verifier. */

import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { getPluginSettings } from "@oh-my-pi/pi-coding-agent/extensibility/plugins/loader";
import { extractExplicitThinkingSelector } from "@oh-my-pi/pi-coding-agent/config/model-resolver";
import type { Model } from "@oh-my-pi/pi-ai";
import { createHash } from "node:crypto";
import { createWrappedProvider } from "./auto.ts";
import { createVerifierClient } from "./client.ts";

const PLUGIN_NAME = "omp-llm-verifier";
const WRAPPER_KEY = "omp-llm-verifier-internal";
const WRAPPER_API_PREFIX = "omp-llm-verifier-api-";

export default function verifierExtension(pi: ExtensionAPI): void {
  pi.setLabel("LLM-as-a-Verifier");
  pi.registerFlag("llm-verifier", {
    type: "boolean",
    description: "为当前 OMP 会话启用自动候选验证",
    default: false,
  });

  pi.on("session_start", async (_event, ctx) => {
    try {
      const settings = await getPluginSettings(PLUGIN_NAME, ctx.cwd);
      const enabled = settings.enabled === true || pi.getFlag("llm-verifier") === true;
      if (!enabled) return;
      await enableAutomaticVerification(pi, ctx);
      ctx.ui.notify("LLM-as-a-Verifier 已启用：普通请求会自动生成候选并回放验证胜者。", "info");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      ctx.ui.notify("LLM-as-a-Verifier 暂不可用：" + message, "warning");
    }
  });
}

async function enableAutomaticVerification(
  pi: ExtensionAPI,
  ctx: ExtensionContext,
): Promise<void> {
  const originalModel = ctx.models.resolve("@default") ?? ctx.model;
  if (!originalModel) throw new Error("OMP modelRoles.default 没有解析到可用模型。");
  if (originalModel.provider.startsWith("omp-llm-verifier-")) return;
  if (originalModel.api !== "openai-completions" && originalModel.api !== "openai-responses") {
    throw new Error(
      "OMP 默认模型 " + originalModel.provider + "/" + originalModel.id +
      " 使用 " + originalModel.api + "，验证器需要 OpenAI token logprobs。",
    );
  }
  const registry = ctx.modelRegistry as ExtensionContext["modelRegistry"] & {
    getProviderHeaders?: (provider: string) => Record<string, string> | undefined;
  };
  const providerHeaders = registry.getProviderHeaders?.(originalModel.provider);
  const candidateModel = providerHeaders
    ? { ...originalModel, headers: { ...providerHeaders, ...originalModel.headers } }
    : originalModel;

  const resolver = ctx.modelRegistry.resolver(
    originalModel,
    ctx.sessionManager.getSessionId(),
  );
  const verifierClient = await createDefaultVerifierClient(pi, {
    models: ctx.models,
    modelRegistry: ctx.modelRegistry,
    sessionManager: ctx.sessionManager,
  });
  const providerName = wrapperProviderName(originalModel, ctx.sessionManager.getSessionId());
  const wrapperApi = wrapperApiName(providerName);
  const modelId = "default";
  const config = {
    baseUrl: originalModel.baseUrl,
    api: wrapperApi,
    apiKey: WRAPPER_KEY,
    authHeader: false,
    streamSimple: createWrappedProvider({
      originalModel: candidateModel,
      verifierClient,
      apiKeyResolver: resolver,
      onDegraded: (event) => {
        console.warn(JSON.stringify({
          component: "omp-llm-verifier",
          event: "degraded",
          ...event,
        }));
        ctx.ui.notify(
          event.reason === "verification_error"
            ? "LLM-as-a-Verifier 本次评审失败，已返回首个候选响应。"
            : "LLM-as-a-Verifier 可用候选不足，已返回唯一成功候选。",
          "warning",
        );
      },
    }),
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
  if (!wrapperModel) throw new Error("无法注册自动验证包装模型。");

  const thinkingLevel = pi.getThinkingLevel();
  if (!await pi.setModel(wrapperModel)) {
    throw new Error("OMP 无法切换到自动验证包装模型。");
  }
  if (thinkingLevel !== undefined) pi.setThinkingLevel(thinkingLevel);
}

export async function createDefaultVerifierClient(
  pi: ExtensionAPI,
  ctx: Pick<ExtensionContext, "models" | "modelRegistry"> & {
    sessionManager?: ExtensionContext["sessionManager"];
  },
) {
  const settings = pi.pi.settings;
  return createVerifierClient({
    ...ctx,
    sessionId: ctx.sessionManager?.getSessionId(),
    defaultThinkingLevel: extractExplicitThinkingSelector(
      settings.getModelRole("default"),
      settings,
    ),
  });
}

function wrapperProviderName(model: Model, sessionId: string): string {
  const identity = model.provider + "\\0" + model.id + "\\0" + model.baseUrl + "\\0" + sessionId;
  return "omp-llm-verifier-" + createHash("sha256").update(identity).digest("hex").slice(0, 12);
}

function wrapperApiName(providerName: string): string {
  return WRAPPER_API_PREFIX + providerName.slice("omp-llm-verifier-".length);
}
