/** OpenAI-compatible verifier client bound to OMP's configured default model. */

import type { ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import {
  isAuthRetryableError,
  seedApiKeyResolver,
  withAuth,
  type ApiKeyResolver,
} from "@oh-my-pi/pi-ai/auth-retry";
import type { Model } from "@oh-my-pi/pi-ai";
import { createHash } from "node:crypto";
import type { VerifierReply } from "./scale.ts";

// Terminal-Bench 2.1 self-verification reference defaults for DeepSeek.
export const DEFAULT_EFFORT = "high";
export const DEFAULT_MAX_TOKENS = 32768;

export interface VerifierConfig {
  baseUrl: string;
  apiKey: string;
  apiKeyResolver?: ApiKeyResolver;
  provider: string;
  api: string;
  modelId: string;
  headers?: Record<string, string>;
  compat?: VerifierCompat;
  keyless?: boolean;
  effort: string;
  maxTokens: number;
}

/** OMP's OpenAI transport compatibility policy, kept structural for plugin portability. */
interface VerifierCompat {
  supportsReasoningParams?: boolean;
  supportsReasoningEffort?: boolean;
  reasoningEffortMap?: Record<string, string>;
  reasoningDisableMode?: string;
  omitReasoningEffort?: boolean;
  thinkingFormat?: string;
  includeEncryptedReasoning?: boolean;
}

export interface UsageSnapshot {
  calls: number;
  inputTokens: number;
  cachedInputTokens: number;
  uncachedInputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheHitRate: number;
}

export function diffUsage(after: UsageSnapshot, before: UsageSnapshot): UsageSnapshot {
  const inputTokens = Math.max(0, after.inputTokens - before.inputTokens);
  const cachedInputTokens = Math.max(
    0,
    after.cachedInputTokens - before.cachedInputTokens,
  );
  return {
    calls: Math.max(0, after.calls - before.calls),
    inputTokens,
    cachedInputTokens,
    uncachedInputTokens: Math.max(0, inputTokens - cachedInputTokens),
    outputTokens: Math.max(0, after.outputTokens - before.outputTokens),
    reasoningTokens: Math.max(0, after.reasoningTokens - before.reasoningTokens),
    cacheHitRate: inputTokens ? cachedInputTokens / inputTokens : 0,
  };
}

/** Process-wide, thread-safe verifier token counter (mirrors llm_verifier.USAGE). */
class TokenUsage {
  calls = 0;
  inputTokens = 0;
  cachedInputTokens = 0;
  outputTokens = 0;
  reasoningTokens = 0;

  add(input = 0, cached = 0, output = 0, reasoning = 0, calls = 1): void {
    this.calls += calls;
    this.inputTokens += input;
    this.cachedInputTokens += cached;
    this.outputTokens += output;
    this.reasoningTokens += reasoning;
  }

  snapshot(): UsageSnapshot {
    return {
      calls: this.calls,
      inputTokens: this.inputTokens,
      cachedInputTokens: this.cachedInputTokens,
      uncachedInputTokens: this.inputTokens - this.cachedInputTokens,
      outputTokens: this.outputTokens,
      reasoningTokens: this.reasoningTokens,
      cacheHitRate: this.inputTokens
        ? this.cachedInputTokens / this.inputTokens
        : 0,
    };
  }
}

export const USAGE = new TokenUsage();

interface ParsedPositionLogprobs {
  tokens: string[];
  positionLogprobs: Array<Array<[string, number]>>;
}

function parsePositionLogprobs(raw: unknown): ParsedPositionLogprobs | undefined {
  if (!Array.isArray(raw) || raw.length === 0) return undefined;
  const tokens: string[] = [];
  const positionLogprobs: Array<Array<[string, number]>> = [];
  for (const rawPosition of raw) {
    if (!rawPosition || typeof rawPosition !== "object" || Array.isArray(rawPosition)) {
      tokens.push("");
      positionLogprobs.push([]);
      continue;
    }
    const position = rawPosition as Record<string, unknown>;
    const token = String(position.token ?? "");
    tokens.push(token);
    const alternatives: Array<[string, number]> = [];
    if (Array.isArray(position.top_logprobs)) {
      for (const rawAlternative of position.top_logprobs) {
        if (!rawAlternative || typeof rawAlternative !== "object" || Array.isArray(rawAlternative)) {
          continue;
        }
        const alternative = rawAlternative as Record<string, unknown>;
        const logprob = Number(alternative.logprob);
        if (!Number.isFinite(logprob)) continue;
        alternatives.push([String(alternative.token ?? ""), logprob]);
      }
    }
    if (alternatives.length > 0) {
      positionLogprobs.push(alternatives);
      continue;
    }
    const logprob = Number(position.logprob);
    positionLogprobs.push(token && Number.isFinite(logprob) ? [[token, logprob]] : []);
  }
  return { tokens, positionLogprobs };
}

export function formatUsage(usage: UsageSnapshot): string[] {
  return [
    `Verifier tokens (${usage.calls} verifier calls)`,
    `  input                        ${usage.inputTokens.toLocaleString()}`,
    `    cached input               ${usage.cachedInputTokens.toLocaleString()}  (${(100 * usage.cacheHitRate).toFixed(1)}% hit rate)`,
    `    uncached input             ${usage.uncachedInputTokens.toLocaleString()}`,
    `  output                       ${usage.outputTokens.toLocaleString()}`,
    `    reasoning                  ${usage.reasoningTokens.toLocaleString()}`,
  ];
}

export class MissingAPIKeyError extends Error {}

interface MergedAbortSignal {
  signal: AbortSignal;
  dispose(): void;
}

/** Combine an external abort signal with an internal timeout and release listeners. */
function mergeAbortSignals(
  external: AbortSignal | undefined,
  timeoutMs: number,
): MergedAbortSignal {
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort(external?.reason);
  if (external?.aborted) onExternalAbort();
  else external?.addEventListener("abort", onExternalAbort, { once: true });
  const timeout = setTimeout(
    () => controller.abort(new DOMException("The operation timed out", "TimeoutError")),
    timeoutMs,
  );
  return {
    signal: controller.signal,
    dispose() {
      clearTimeout(timeout);
      external?.removeEventListener("abort", onExternalAbort);
    },
  };
}

export class VerifierClient {
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly apiKeyResolver?: ApiKeyResolver;
  readonly provider: string;
  readonly api: string;
  readonly model: string;
  readonly effort: string;
  readonly maxTokens: number;
  readonly headers: Record<string, string>;
  readonly compat: VerifierCompat;
  readonly keyless: boolean;
  readonly requestIdentity: string;

  constructor(cfg: VerifierConfig) {
    this.baseUrl = cfg.baseUrl.replace(/\/+$/, "");
    this.keyless = cfg.keyless === true || cfg.apiKey.trim() === "N/A";
    this.apiKey = this.keyless ? "" : cfg.apiKey.trim();
    this.apiKeyResolver = cfg.apiKeyResolver;
    this.provider = cfg.provider;
    this.api = cfg.api;
    this.model = cfg.modelId;
    this.effort = cfg.effort;
    const maxTokens = cfg.maxTokens;
    if (!Number.isInteger(maxTokens) || maxTokens < 1) {
      throw new Error("Verifier maxTokens must be a positive integer");
    }
    if (
      !this.baseUrl ||
      (!this.apiKey && !this.apiKeyResolver && !this.keyless) ||
      !this.provider ||
      !this.api ||
      !this.model
    ) {
      throw new Error("Verifier client requires an authenticated OMP model.");
    }
    this.maxTokens = maxTokens;
    this.headers = { ...cfg.headers };
    this.compat = { ...cfg.compat };
    this.requestIdentity = buildRequestIdentity({
      apiKey: this.apiKey,
      keyless: this.keyless,
      headers: this.headers,
      compat: this.compat,
    });
  }

  get ready(): boolean {
    return this.keyless || this.apiKey.length > 0 || this.apiKeyResolver !== undefined;
  }

  /** Request a verifier response with token-level logprobs. */
  async scoreReply(prompt: string, opts: { signal?: AbortSignal } = {}): Promise<VerifierReply> {
    if (!prompt.trim()) throw new Error("Verifier prompt must be non-empty");
    if (!this.ready) {
      throw new MissingAPIKeyError(
        "OMP 当前模型没有可用凭据。",
      );
    }
    const effort = this.effort;
    const reasoningEnabled = effort !== "off" && effort !== "disabled" && effort !== "none";
    const responses = this.api === "openai-responses";
    const body: Record<string, unknown> = responses
      ? {
          model: this.model,
          input: [
            {
              role: "user",
              content: [{ type: "input_text", text: prompt }],
            },
          ],
          max_output_tokens: this.maxTokens,
          top_logprobs: 20,
          include: [
            "message.output_text.logprobs",
            ...(this.compat.includeEncryptedReasoning ? ["reasoning.encrypted_content"] : []),
          ],
          store: false,
          stream: false,
        }
      : {
          model: this.model,
          messages: [{ role: "user", content: prompt }],
          max_tokens: this.maxTokens,
          logprobs: true,
          top_logprobs: 20,
        };

    if (responses) {
      this.applyResponsesReasoning(body, effort, reasoningEnabled);
    } else {
      this.applyChatReasoning(body, effort, reasoningEnabled);
    }

    const requestAbort = mergeAbortSignals(opts.signal, 600_000);
    try {
      const credential = this.apiKeyResolver ?? (this.keyless ? "N/A" : this.apiKey);
      const data = await withAuth(
        credential,
        async (apiKey) => {
          const endpoint = this.baseUrl + "/" + (responses ? "responses" : "chat/completions");
          const res = await fetch(endpoint, {
            method: "POST",
            headers: this.requestHeaders(apiKey),
            body: JSON.stringify(body),
            signal: requestAbort.signal,
          });
          if (!res.ok) {
            const detail = redactSecrets(
              (await res.text()).slice(0, 500),
              [
                apiKey,
                this.apiKey,
                ...Object.values(this.headers),
                ...Object.values(this.headers).flatMap(headerSecretParts),
              ],
            );
            const error = new Error(
              "Verifier API " + res.status + " " + res.statusText + ": " + detail,
            ) as Error & { status?: number };
            error.status = res.status;
            throw error;
          }
          try {
            const parsed: unknown = await res.json();
            if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
              throw new Error("response is not an object");
            }
            return parsed as Record<string, unknown>;
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            throw new Error("Verifier API returned invalid JSON: " + detail);
          }
        },
        {
          isAuthError: isAuthRetryableError,
          signal: requestAbort.signal,
          missingKeyMessage: "OMP 当前模型没有可用凭据。",
        },
      );
      this.recordUsage(data);
      if (responses) return this.parseResponsesReply(data);

      const choices = Array.isArray(data.choices)
        ? (data.choices as Array<Record<string, unknown>>)
        : [];
      const choice = choices[0];
      if (!choice) throw new Error("Verifier API returned no choices");
      const msg = choice.message as Record<string, unknown> | undefined;
      const text: string =
        typeof msg?.content === "string" ? msg.content : "";
      const logprobs = choice.logprobs as
        | { content?: Array<Record<string, unknown>> }
        | undefined;

      const parsedLogprobs = parsePositionLogprobs(logprobs?.content);
      const tokens = parsedLogprobs?.tokens;
      const positionLogprobs = parsedLogprobs?.positionLogprobs;
      if (!positionLogprobs || positionLogprobs.length === 0) {
        throw new Error(
          `Verifier returned no answer logprobs (finish_reason=${String(
            choice.finish_reason ?? "?",
          )}) — 当前 OMP 模型没有返回 token logprobs。`,
        );
      }
      return { text, tokens, positionLogprobs };
    } finally {
      requestAbort.dispose();
    }
  }

  private parseResponsesReply(data: Record<string, unknown>): VerifierReply {
    const textParts: string[] = [];
    const positions: unknown[] = [];
    const output = Array.isArray(data.output) ? data.output : [];
    for (const rawItem of output) {
      if (!rawItem || typeof rawItem !== "object" || Array.isArray(rawItem)) continue;
      const item = rawItem as Record<string, unknown>;
      const content = Array.isArray(item.content) ? item.content : [];
      for (const rawPart of content) {
        if (!rawPart || typeof rawPart !== "object" || Array.isArray(rawPart)) continue;
        const part = rawPart as Record<string, unknown>;
        if (part.type !== "output_text" && typeof part.text !== "string") continue;
        if (typeof part.text === "string") textParts.push(part.text);
        if (Array.isArray(part.logprobs)) positions.push(...part.logprobs);
      }
    }
    const parsedLogprobs = parsePositionLogprobs(positions);
    if (!parsedLogprobs || parsedLogprobs.positionLogprobs.length === 0) {
      const status = typeof data.status === "string" ? data.status : "?";
      throw new Error(
        `Verifier returned no answer logprobs (status=${status}) — 当前 OMP 默认模型没有返回 token logprobs。`,
      );
    }
    return {
      text:
        textParts.join("") ||
        (typeof data.output_text === "string" ? data.output_text : ""),
      tokens: parsedLogprobs.tokens,
      positionLogprobs: parsedLogprobs.positionLogprobs,
    };
  }

  private requestHeaders(apiKey: string): Record<string, string> {
    const headers = new Headers(this.headers);
    if (!this.keyless && apiKey !== "N/A" && !headers.has("authorization")) {
      headers.set("Authorization", "Bearer " + apiKey);
    }
    headers.set("Content-Type", "application/json");
    return Object.fromEntries(headers.entries());
  }

  private applyChatReasoning(
    body: Record<string, unknown>,
    effort: string,
    enabled: boolean,
  ): void {
    const supportsParams = this.compat.supportsReasoningParams !== false;
    const supportsEffort = this.compat.supportsReasoningEffort !== false;
    const mode = this.compat.reasoningDisableMode ?? "default";
    const wireEffort = mapReasoningEffort(this.compat, effort);
    if (!supportsParams) return;
    if (enabled) {
      switch (mode) {
        case "zai-thinking-disabled":
          body.thinking = { type: "enabled" };
          if (supportsEffort) body.reasoning_effort = wireEffort;
          break;
        case "qwen-enable-thinking-false":
          body.enable_thinking = true;
          break;
        case "qwen-template-false":
          body.chat_template_kwargs = { enable_thinking: true };
          break;
        case "openrouter-enabled-false":
          body.reasoning = { effort: wireEffort };
          break;
        default:
          if (supportsEffort && !this.compat.omitReasoningEffort) {
            body.reasoning_effort = wireEffort;
          }
          break;
      }
      return;
    }
    switch (mode) {
      case "none-effort":
        if (supportsEffort && !this.compat.omitReasoningEffort) body.reasoning_effort = "none";
        break;
      case "zai-thinking-disabled":
        body.thinking = { type: "disabled" };
        break;
      case "qwen-enable-thinking-false":
        body.enable_thinking = false;
        break;
      case "qwen-template-false":
        body.chat_template_kwargs = { enable_thinking: false };
        break;
      case "openrouter-enabled-false":
        body.reasoning = { enabled: false };
        break;
    }
  }

  private applyResponsesReasoning(
    body: Record<string, unknown>,
    effort: string,
    enabled: boolean,
  ): void {
    if (this.compat.supportsReasoningParams === false || this.compat.omitReasoningEffort) return;
    if (enabled && this.compat.supportsReasoningEffort !== false) {
      body.reasoning = { effort: mapReasoningEffort(this.compat, effort) };
      return;
    }
    if (!enabled && this.compat.supportsReasoningEffort !== false) {
      body.reasoning = { effort: "none" };
    }
  }

  private recordUsage(data: Record<string, unknown>): void {
    const usage = data.usage as
      | {
          prompt_tokens?: number;
          input_tokens?: number;
          prompt_cache_hit_tokens?: number;
          completion_tokens?: number;
          output_tokens?: number;
          prompt_tokens_details?: { cached_tokens?: number };
          input_tokens_details?: { cached_tokens?: number };
          completion_tokens_details?: { reasoning_tokens?: number };
          output_tokens_details?: { reasoning_tokens?: number };
        }
      | undefined;
    if (!usage) return;
    const numberOrZero = (value: unknown): number =>
      typeof value === "number" && Number.isFinite(value) && value >= 0
        ? value
        : 0;
    let cached = numberOrZero(usage.prompt_cache_hit_tokens);
    if (!cached) cached = numberOrZero(usage.prompt_tokens_details?.cached_tokens);
    if (!cached) cached = numberOrZero(usage.input_tokens_details?.cached_tokens);
    const reasoning =
      numberOrZero(usage.completion_tokens_details?.reasoning_tokens) ||
      numberOrZero(usage.output_tokens_details?.reasoning_tokens);
    USAGE.add(
      numberOrZero(usage.prompt_tokens ?? usage.input_tokens),
      cached,
      numberOrZero(usage.completion_tokens ?? usage.output_tokens),
      reasoning,
    );
  }
}

function mapReasoningEffort(compat: VerifierCompat, effort: string): string {
  return compat.reasoningEffortMap?.[effort] ?? effort;
}

function normalizeCompat(raw: unknown): VerifierCompat {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
  const value = raw as Record<string, unknown>;
  const booleanField = (name: string): boolean | undefined =>
    typeof value[name] === "boolean" ? value[name] : undefined;
  const stringField = (name: string): string | undefined =>
    typeof value[name] === "string" ? value[name] : undefined;
  const reasoningEffortMap =
    value.reasoningEffortMap && typeof value.reasoningEffortMap === "object" && !Array.isArray(value.reasoningEffortMap)
      ? Object.fromEntries(
          Object.entries(value.reasoningEffortMap as Record<string, unknown>)
            .filter(([, effort]) => typeof effort === "string"),
        ) as Record<string, string>
      : undefined;
  return {
    supportsReasoningParams: booleanField("supportsReasoningParams"),
    supportsReasoningEffort: booleanField("supportsReasoningEffort"),
    reasoningEffortMap,
    reasoningDisableMode: stringField("reasoningDisableMode"),
    omitReasoningEffort: booleanField("omitReasoningEffort"),
    thinkingFormat: stringField("thinkingFormat"),
    includeEncryptedReasoning: booleanField("includeEncryptedReasoning"),
  };
}

function buildRequestIdentity(input: {
  apiKey: string;
  keyless: boolean;
  headers: Record<string, string>;
  compat: VerifierCompat;
}): string {
  const headers = Object.entries(input.headers)
    .map(([name, value]) => [name.toLowerCase(), value.trim()] as const)
    .sort(([a], [b]) => a.localeCompare(b));
  return createHash("sha256")
    .update(JSON.stringify({
      auth: input.keyless ? "keyless" : "api-key",
      apiKey: input.apiKey,
      headers,
      compat: stableValue(input.compat),
    }))
    .digest("hex")
    .slice(0, 32);
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, item]) => [key, stableValue(item)]),
    );
  }
  return value;
}

function redactSecrets(text: string, secrets: string[]): string {
  return [...new Set(secrets.map((secret) => secret.trim()).filter(Boolean))]
    .sort((a, b) => b.length - a.length)
    .reduce((redacted, secret) => redacted.split(secret).join("[redacted]"), text);
}

function headerSecretParts(value: string): string[] {
  const trimmed = value.trim();
  if (!trimmed) return [];
  const parts = new Set<string>([trimmed]);
  const whitespace = trimmed.split(/\s+/);
  if (whitespace.length > 1) parts.add(whitespace.at(-1) ?? "");
  for (const assignment of trimmed.split(";")) {
    const equals = assignment.indexOf("=");
    if (equals >= 0) parts.add(assignment.slice(equals + 1).trim());
  }
  return [...parts].filter(Boolean);
}

type OmpModelContext = Pick<
  ExtensionContext,
  "models" | "modelRegistry"
> & {
  model?: Model;
  defaultThinkingLevel?: string;
  sessionId?: string;
};

function resolveOmpEffort(
  model: {
    reasoning?: boolean;
    thinking?: { efforts?: readonly unknown[]; defaultLevel?: unknown };
  },
  configured?: string,
): string {
  const supported = model.thinking?.efforts?.map(String) ?? [];
  if (!model.reasoning) return "off";
  const requested = configured?.trim();
  if (requested && requested !== "auto" && supported.includes(requested)) {
    return requested;
  }
  const modelDefault = model.thinking?.defaultLevel
    ? String(model.thinking.defaultLevel)
    : undefined;
  if (modelDefault && supported.includes(modelDefault)) return modelDefault;
  if (supported.includes(DEFAULT_EFFORT)) return DEFAULT_EFFORT;
  return supported.at(-1) ?? "off";
}

/** Create a verifier from OMP's configured default model and credentials. */
export async function createVerifierClient(
  ctx: OmpModelContext,
): Promise<VerifierClient> {
  const model = ctx.model ?? ctx.models.resolve("@default");
  if (!model) {
    throw new MissingAPIKeyError("OMP modelRoles.default 没有解析到可用模型。");
  }
  if (model.api !== "openai-completions" && model.api !== "openai-responses") {
    throw new Error(
      "OMP 默认模型 " + model.provider + "/" + model.id + " 使用 " + model.api +
        "，LLM-as-a-Verifier 需要支持 OpenAI token logprobs 的模型。",
    );
  }
  const resolver = ctx.modelRegistry.resolver(model, ctx.sessionId);
  const auth = await resolveOmpAuth(ctx, model);
  if (!auth.ok) {
    throw new MissingAPIKeyError(
      "OMP 默认模型 " + model.provider + "/" + model.id + " 没有可用凭据，请先在 OMP 登录对应 provider。",
    );
  }
  const initialKey = auth.apiKey ?? "";
  const apiKeyResolver = initialKey && initialKey !== "N/A"
    ? seedApiKeyResolver(initialKey, resolver)
    : undefined;
  if (!initialKey && !apiKeyResolver) {
    throw new MissingAPIKeyError(
      "OMP 默认模型 " + model.provider + "/" + model.id + " 没有可用凭据，请先在 OMP 登录对应 provider。",
    );
  }
  const maxTokens =
    typeof model.maxTokens === "number" && model.maxTokens > 0
      ? Math.min(DEFAULT_MAX_TOKENS, model.maxTokens)
      : DEFAULT_MAX_TOKENS;
  return new VerifierClient({
    baseUrl: model.baseUrl,
    apiKey: initialKey,
    apiKeyResolver,
    keyless: initialKey === "N/A",
    provider: model.provider,
    api: model.api,
    modelId: model.requestModelId ?? model.id,
    headers: { ...model.headers, ...auth.headers },
    compat: normalizeCompat(model.compat),
    effort: resolveOmpEffort(
      model,
      ctx.defaultThinkingLevel,
    ),
    maxTokens,
  });
}

async function resolveOmpAuth(
  ctx: OmpModelContext,
  model: Model,
): Promise<{ ok: true; apiKey?: string; headers?: Record<string, string> } | { ok: false; error: string }> {
  const registry = ctx.modelRegistry as ExtensionContext["modelRegistry"] & {
    getProviderHeaders?: (provider: string) => Record<string, string> | undefined;
  };
  if (typeof registry.getApiKey === "function") {
    try {
      const apiKey = await registry.getApiKey(model, ctx.sessionId);
      if (apiKey === undefined) return { ok: false, error: "No API key found for " + model.provider };
      return {
        ok: true,
        apiKey,
        headers: registry.getProviderHeaders?.(model.provider),
      };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }
  return ctx.modelRegistry.getApiKeyAndHeaders(model);
}
