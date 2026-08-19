/**
 * Verifier backend client for opencode-go (https://opencode.ai/zen/go/v1),
 * the OpenAI-compatible gateway omp uses for `opencode-go/deepseek-v4-flash`.
 *
 * The verifier needs token-level logprobs, which the omp session API does
 * not expose, so the plugin talks to the backend directly. Credentials are
 * resolved like omp does: OPENCODE_API_KEY env var first, then the stored
 * login credential in the omp auth store (`~/.omp/agent/agent.db`).
 *
 * DeepSeek-family models emit the `<score_X> letter </score_X>` tags
 * themselves, so unlike vLLM-style open models we read the distribution
 * straight from the response logprobs (no prefill trick needed).
 */

import type { VerifierReply } from "./scale.ts";
import { Database } from "bun:sqlite";

export const DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1";
export const DEFAULT_MODEL = "deepseek-v4-flash";
export const DEFAULT_EFFORT = "xhigh"; // omp: opencode-go/deepseek-v4-flash:xhigh
export const DEFAULT_MODEL_SELECTOR =
  "opencode-go/deepseek-v4-flash:xhigh";
export const DEFAULT_MAX_TOKENS = 65536; // xhigh thinking shares the budget

export interface VerifierConfig {
  baseUrl?: string;
  apiKey?: string;
  model?: string;
  effort?: string; // reasoning_effort: off | low | high | xhigh | max
  maxTokens?: number;
  maxWorkers?: number;
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

/** Combine an external abort signal with an internal timeout into one. */
function mergeAbortSignals(
  external: AbortSignal | undefined,
  timeout: AbortSignal,
): AbortSignal {
  if (!external) return timeout;
  if (external.aborted) return external;
  const controller = new AbortController();
  const onAbort = () => controller.abort(external.reason);
  external.addEventListener("abort", onAbort, { once: true });
  timeout.addEventListener("abort", () => controller.abort(timeout.reason), {
    once: true,
  });
  return controller.signal;
}

export interface AuthCredential {
  key?: string;
  [k: string]: unknown;
}

/** Directory that holds agent.db (PI_CODING_AGENT_DIR wins). */
export function agentDir(): string {
  return (
    process.env.PI_CODING_AGENT_DIR ||
    `${process.env.HOME || Bun.env.HOME || ""}/.omp/agent`
  );
}

export interface ModelSelector {
  provider?: string;
  model: string;
  effort?: string;
}

/** Split omp's provider/model:effort spelling into API fields. */
export function parseModelSelector(selector: string): ModelSelector {
  const trimmed = selector.trim();
  if (!trimmed) throw new Error("Verifier model selector must be non-empty");
  const colon = trimmed.lastIndexOf(":");
  const withoutEffort = colon > trimmed.lastIndexOf("/")
    ? trimmed.slice(0, colon)
    : trimmed;
  const effort = colon > trimmed.lastIndexOf("/")
    ? trimmed.slice(colon + 1)
    : undefined;
  const slash = withoutEffort.indexOf("/");
  const provider = slash > 0 ? withoutEffort.slice(0, slash) : undefined;
  const model = slash > 0 ? withoutEffort.slice(slash + 1) : withoutEffort;
  if (!model) throw new Error(`Invalid verifier model selector: ${selector}`);
  return { provider, model, effort: effort || undefined };
}

/** Read the stored opencode-go API key from omp's auth store. */
export function storedApiKey(provider = "opencode-go"): string | undefined {
  try {
    const db = new Database(`${agentDir()}/agent.db`, { readonly: true });
    try {
      const row = db
        .query(
          "SELECT data FROM auth_credentials WHERE provider = ? AND credential_type = 'api_key' ORDER BY updated_at DESC LIMIT 1",
        )
        .get(provider) as { data: string } | null;
      if (row) {
        const cred = JSON.parse(row.data) as AuthCredential;
        if (typeof cred.key === "string" && cred.key.length > 0) {
          return cred.key;
        }
      }
    } finally {
      db.close();
    }
  } catch {
    // Not running inside omp (no auth store) — caller falls back to env.
  }
  return undefined;
}

/** Resolve the verifier API key: OPENCODE_API_KEY env, then auth store. */
export function resolveApiKey(): string | undefined {
  return process.env.OPENCODE_API_KEY || storedApiKey();
}

export class VerifierClient {
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly model: string;
  readonly effort: string;
  readonly maxTokens: number;

  constructor(cfg: VerifierConfig = {}) {
    this.baseUrl = (cfg.baseUrl ||
      process.env.OPENCODE_BASE_URL ||
      DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = (cfg.apiKey || resolveApiKey() || "").trim();
    const selector = parseModelSelector(
      cfg.model || process.env.VERIFIER_MODEL || DEFAULT_MODEL_SELECTOR,
    );
    this.model = selector.model;
    this.effort =
      cfg.effort ||
      process.env.VERIFIER_EFFORT ||
      selector.effort ||
      DEFAULT_EFFORT;
    const maxTokens = cfg.maxTokens ?? DEFAULT_MAX_TOKENS;
    if (!Number.isInteger(maxTokens) || maxTokens < 1) {
      throw new Error("Verifier maxTokens must be a positive integer");
    }
    this.maxTokens = maxTokens;
  }

  get ready(): boolean {
    return this.apiKey.length > 0;
  }

  /** Request a chat completion with token-level logprobs (DeepSeek path). */
  async scoreReply(prompt: string, opts: {
    signal?: AbortSignal;
    maxTokens?: number;
  } = {}): Promise<VerifierReply> {
    if (!prompt.trim()) throw new Error("Verifier prompt must be non-empty");
    if (!this.ready) {
      throw new MissingAPIKeyError(
        "No verifier API key. Set OPENCODE_API_KEY or log in to " +
          "opencode-go in omp (`/login opencode-go`).",
      );
    }
    const maxTokens = opts.maxTokens ?? this.maxTokens;
    const body: Record<string, unknown> = {
      model: this.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 1.0,
      logprobs: true,
      top_logprobs: 20,
    };
    const effort = this.effort;
    if (effort !== "off" && effort !== "disabled" && effort !== "none") {
      body.thinking = { type: "enabled" };
      body.reasoning_effort = effort;
    } else {
      body.thinking = { type: "disabled" };
    }

    const res = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: mergeAbortSignals(
        opts.signal,
        AbortSignal.timeout(600_000), // xhigh thinking can take many minutes
      ),
    });
    if (!res.ok) {
      const detail = redactSecret((await res.text()).slice(0, 500), this.apiKey);
      throw new Error(
        `Verifier API ${res.status} ${res.statusText}: ${detail}`,
      );
    }
    let data: Record<string, unknown>;
    try {
      const parsed: unknown = await res.json();
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("response is not an object");
      }
      data = parsed as Record<string, unknown>;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`Verifier API returned invalid JSON: ${detail}`);
    }
    this.recordUsage(data);

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

    let tokens: string[] | undefined;
    let positionLogprobs: VerifierReply["positionLogprobs"];
    if (logprobs?.content?.length) {
      tokens = [];
      positionLogprobs = [];
      for (const rawPos of logprobs.content) {
        if (!rawPos || typeof rawPos !== "object" || Array.isArray(rawPos)) {
          tokens.push("");
          positionLogprobs.push([]);
          continue;
        }
        const pos = rawPos as Record<string, unknown>;
        const tok = String(pos.token ?? "");
        tokens.push(tok);
        const rawAlternatives = Array.isArray(pos.top_logprobs)
          ? (pos.top_logprobs as Array<unknown>)
          : undefined;
        const alts = rawAlternatives
          ?.map((rawAlt) => {
            if (!rawAlt || typeof rawAlt !== "object" || Array.isArray(rawAlt)) {
              return undefined;
            }
            const alt = rawAlt as Record<string, unknown>;
            return [String(alt.token ?? ""), Number(alt.logprob)] as [string, number];
          })
          .filter((alt): alt is [string, number] => alt !== undefined)
          .filter(([, logprob]) => Number.isFinite(logprob));
        if (alts && alts.length > 0) {
          positionLogprobs.push(alts);
        } else {
          const logprob = Number(pos.logprob);
          if (tok && Number.isFinite(logprob)) {
            positionLogprobs.push([[tok, logprob]]);
          } else {
            positionLogprobs.push([]);
          }
        }
      }
    }
    if (!positionLogprobs || positionLogprobs.length === 0) {
      throw new Error(
        `Verifier returned no answer logprobs (finish_reason=${String(
          choice.finish_reason ?? "?",
        )}) — xhigh thinking may have consumed the ${maxTokens}-token budget; ` +
          "raise maxTokens or lower the reasoning effort.",
      );
    }
    return { text, tokens, positionLogprobs };
  }

  private recordUsage(data: Record<string, unknown>): void {
    const usage = data.usage as
      | {
          prompt_tokens?: number;
          prompt_cache_hit_tokens?: number;
          completion_tokens?: number;
          prompt_tokens_details?: { cached_tokens?: number };
          completion_tokens_details?: { reasoning_tokens?: number };
        }
      | undefined;
    if (!usage) return;
    const numberOrZero = (value: unknown): number =>
      typeof value === "number" && Number.isFinite(value) && value >= 0
        ? value
        : 0;
    let cached = numberOrZero(usage.prompt_cache_hit_tokens);
    if (!cached) cached = numberOrZero(usage.prompt_tokens_details?.cached_tokens);
    const reasoning = numberOrZero(usage.completion_tokens_details?.reasoning_tokens);
    USAGE.add(
      numberOrZero(usage.prompt_tokens),
      cached,
      numberOrZero(usage.completion_tokens),
      reasoning,
    );
  }
}

function redactSecret(text: string, secret: string): string {
  if (!secret) return text;
  return text.split(secret).join("[redacted]");
}
