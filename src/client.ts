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
  return process.env.PI_CODING_AGENT_DIR || `${Bun.env.HOME}/.omp/agent`;
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
    this.apiKey = cfg.apiKey || resolveApiKey() || "";
    this.model = cfg.model || DEFAULT_MODEL;
    this.effort = cfg.effort || process.env.VERIFIER_EFFORT || DEFAULT_EFFORT;
    this.maxTokens = cfg.maxTokens || DEFAULT_MAX_TOKENS;
  }

  get ready(): boolean {
    return this.apiKey.length > 0;
  }

  /** Request a chat completion with token-level logprobs (DeepSeek path). */
  async scoreReply(prompt: string, opts: {
    signal?: AbortSignal;
    maxTokens?: number;
  } = {}): Promise<VerifierReply> {
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
      const detail = (await res.text()).slice(0, 500);
      throw new Error(
        `Verifier API ${res.status} ${res.statusText}: ${detail}`,
      );
    }
    const data = (await res.json()) as Record<string, unknown>;
    this.recordUsage(data);

    const choice = (data.choices as Array<Record<string, unknown>>)[0];
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
      for (const pos of logprobs.content) {
        const tok = String(pos.token ?? "");
        tokens.push(tok);
        const alts = (pos.top_logprobs as
          | Array<{ token?: unknown; logprob?: unknown }>
          | undefined)?.map((alt) => [String(alt.token ?? ""), Number(alt.logprob ?? 0)] as [string, number]);
        if (alts && alts.length > 0) {
          positionLogprobs.push(alts);
        } else {
          positionLogprobs.push([[tok, Number(pos.logprob ?? 0)]]);
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
    let cached = usage.prompt_cache_hit_tokens ?? 0;
    if (!cached) cached = usage.prompt_tokens_details?.cached_tokens ?? 0;
    const reasoning = usage.completion_tokens_details?.reasoning_tokens ?? 0;
    USAGE.add(
      usage.prompt_tokens ?? 0,
      cached,
      usage.completion_tokens ?? 0,
      reasoning,
    );
  }
}
