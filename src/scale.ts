/**
 * Fine-grained reward scale and score extraction.
 *
 * Core idea from the LLM-as-a-Verifier paper (arXiv:2607.05391): instead of
 * collapsing the verifier's judgement into one discrete label, read its
 * probability distribution over an ordered set of score tokens and take the
 * expectation:
 *
 *   R = (1 / C K) sum_c sum_k sum_g  p_theta(v_g | x, c, tau) * phi(v_g)
 *
 * with C criteria, K repeated verifications, and G score tokens
 * (granularity); phi maps each score token to its scalar value. Higher
 * granularity separates positive from negative trajectories better and
 * yields more calibrated comparisons.
 *
 * We use a 20-point scale spelled as letters A..T (A = best, T = worst):
 * letter tokens survive BPE tokenization as single units, which lets us read
 * the model's probability mass over the whole scale from `top_logprobs`.
 */

export const GRANULARITY = 20;

/** Letter -> raw score (A=20 ... T=1). */
const LETTER_VALUES: Record<string, number> = {
  ...Object.fromEntries(
    Array.from({ length: GRANULARITY }, (_, i) => [
      String.fromCharCode(65 + i),
      GRANULARITY - i,
    ]),
  ),
  ...Object.fromEntries(
    Array.from({ length: GRANULARITY }, (_, i) => [
      String.fromCharCode(97 + i),
      GRANULARITY - i,
    ]),
  ),
};

export const SCALE = {
  scaleDescription:
    "Rate how likely the agent correctly solved the task on a " +
    "20-point scale using letters A through T:\n" +
    "  A = clearly and completely succeeded with verified output (best)\n" +
    "  B-D = succeeded with only minor issues\n" +
    "  E-G = above average, mostly correct with some issues\n" +
    "  H-J = uncertain, leans toward success\n" +
    "  K-M = uncertain, leans toward failure\n" +
    "  N-P = below average, significant issues remain\n" +
    "  Q-S = failed with some partial progress\n" +
    "  T = clearly and completely failed (worst)",
  scoreFormat: "LETTER_A_TO_T",
} as const;

/** Distinct raw values 1..20; used to normalize an expectation to [0, 1]. */
const RAW_VALUES = Array.from(
  new Set(Object.values(LETTER_VALUES)),
).sort((a, b) => a - b);
const MIN_VAL = RAW_VALUES[0];
const MAX_VAL = RAW_VALUES[RAW_VALUES.length - 1];

/**
 * A position in the response with its token distribution: list of
 * (token, logprob) alternatives from the model's top_logprobs.
 */
export type PositionLogprobs = Array<[token: string, logprob: number]>;

export interface VerifierReply {
  /** Full assistant text (analysis + score block). */
  text: string;
  /** Tokens as emitted by the backend. */
  tokens?: string[];
  /** Per-token alternative distributions; parallel to `tokens`. */
  positionLogprobs?: PositionLogprobs[];
}

/**
 * Locate the token distribution immediately after `<score_X>` (or its
 * fused `>`-less form). The LAST match wins: the verdict is the score block
 * at the end of the reply, not the model quoting the format mid-analysis.
 */
export function findTagLogprobs(
  reply: VerifierReply,
  tag: string,
): PositionLogprobs | undefined {
  const { tokens, positionLogprobs } = reply;
  if (!tokens || !positionLogprobs || tokens.length === 0) return undefined;
  for (const suffix of [tag, tag.slice(0, -1)]) {
    let found: PositionLogprobs | undefined;
    let textSoFar = "";
    for (let i = 0; i < tokens.length; i++) {
      textSoFar += tokens[i];
      if (textSoFar.trimEnd().endsWith(suffix)) {
        if (i + 1 < positionLogprobs.length) {
          found = positionLogprobs[i + 1];
        }
      }
    }
    if (found !== undefined) return found;
  }
  return undefined;
}

/** Normalize a raw expectation over the scale to [0, 1]. */
function normalize(raw: number): number {
  if (MAX_VAL <= MIN_VAL) return 0.5;
  return (raw - MIN_VAL) / (MAX_VAL - MIN_VAL);
}

/**
 * Expected score over the verifier's token distribution at `tag`,
 * normalized to [0, 1]. Falls back to parsing the literal `<tag> X </tag>`
 * text when no logprobs were returned.
 */
export function extractScore(reply: VerifierReply, tag: string): number {
  const tagLp = findTagLogprobs(reply, tag);
  const probs: Record<number, number> = {};
  if (tagLp) {
    for (const [tokStr, logprob] of tagLp) {
      let tok = tokStr.trim();
      if (tok.startsWith(">")) tok = tok.slice(1).trim(); // fused '>A'
      const raw = LETTER_VALUES[tok];
      if (raw !== undefined) {
        probs[raw] = Math.max(probs[raw] ?? 0, Math.exp(logprob));
      }
    }
  }
  if (Object.keys(probs).length > 0) {
    const totalP = Object.values(probs).reduce((a, b) => a + b, 0);
    const expected =
      Object.entries(probs).reduce((a, [v, p]) => a + Number(v) * p, 0) /
      totalP;
    return normalize(expected);
  }

  // Literal-text fallback: last `<tag> letter </tag>` match.
  const tagName = tag.replace(/[<>]/g, "");
  const pattern = new RegExp(
    `<${tagName}>\\s*(.+?)\\s*</${tagName}>`,
    "gi",
  );
  let match: RegExpExecArray | null;
  let last: RegExpExecArray | null = null;
  while ((match = pattern.exec(reply.text ?? "")) !== null) last = match;
  if (last) {
    const tok = last[1].trim();
    let raw = LETTER_VALUES[tok];
    if (raw === undefined) {
      for (const [vt, val] of Object.entries(LETTER_VALUES)) {
        if (tok.toLowerCase() === vt.toLowerCase()) {
          raw = val;
          break;
        }
      }
    }
    if (raw !== undefined) return normalize(raw);
  }
  return 0.5;
}
