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
 *
 * Implemented as one linear scan per request rather than an O(n^2) rebuild of
 * the accumulated text: the joined token stream is walked once, cumulative
 * char lengths and trailing-whitespace runs are precomputed, and each tag
 * boundary is a constant-length slice comparison. Identical selection
 * semantics to the reference `_find_tag_logprobs`; verified by the
 * differential random test in test/core.test.ts.
 */

/** True for the code units Python `str.rstrip()` treats as whitespace (the
 *  same set as the previous char-class regex, matched on code units so no
 *  per-character RegExp is allocated during the linear scan). */
function isPythonWhitespaceCode(unit: number): boolean {
  return (unit >= 0x09 && unit <= 0x0d) ||
    (unit >= 0x1c && unit <= 0x20) ||
    unit === 0x85 || unit === 0xa0 || unit === 0x1680 ||
    (unit >= 0x2000 && unit <= 0x200a) ||
    unit === 0x2028 || unit === 0x2029 || unit === 0x202f ||
    unit === 0x205f || unit === 0x3000;
}

type TagScan = Array<PositionLogprobs | undefined>;

/**
 * For each requested tag, the token distribution immediately after the LAST
 * matching token, preferring the exact `>` form and falling back to the
 * fused `>`-less form — mirroring the reference loop order (exact suffix
 * scanned to its last match first, fused only if exact never matched).
 */
function scanTagMatches(reply: VerifierReply, tags: string[]): TagScan {
  const out: TagScan = tags.map(() => undefined);
  const { tokens, positionLogprobs } = reply;
  if (!tokens || !positionLogprobs || tokens.length === 0 || tags.length === 0) {
    return out;
  }
  const n = tokens.length;

  // Single pass over tokens. For every boundary we need the Python
  // rstrip()-ed length of the accumulated text: the trailing-whitespace run
  // is carried across tokens (a whitespace-only token extends the previous
  // run, any other token resets it to its own trailing whitespace), so the
  // trimmed length of every prefix is O(1) per token. This replaces a
  // per-CHARACTER whitespace scan of the joined text (plus a full-length
  // Int32Array) and accumulates the joined text in the same walk; selection
  // semantics are identical to `_find_tag_logprobs`, verified by the
  // differential random test in test/core.test.ts.
  const trimmed = new Array<number>(n);
  const allWsToken = new Uint8Array(n);
  const joined = tokens.join("");
  let len = 0;
  let run = 0;
  for (let i = 0; i < n; i++) {
    const tok = tokens[i]!;
    let trailing = 0;
    for (let t = tok.length - 1; t >= 0 && isPythonWhitespaceCode(tok.charCodeAt(t)); t--) {
      trailing += 1;
    }
    const allWs = trailing === tok.length;
    allWsToken[i] = allWs ? 1 : 0;
    len += tok.length;
    run = allWs ? run + tok.length : trailing;
    trimmed[i] = len - run;
  }

  const fused = tags.map((tag) => tag.slice(0, -1));
  const exactLens = tags.map((tag) => tag.length);
  const fusedLens = fused.map((s) => s.length);
  const lastExact = new Array<number>(tags.length).fill(-1);
  const lastFused = new Array<number>(tags.length).fill(-1);

  for (let i = 0; i < n; i++) {
    if (i + 1 >= positionLogprobs.length) continue;
    // Latest reference behavior: a whitespace-only token leaves rstrip() at
    // the same tag boundary and must not shadow the distribution captured at
    // the preceding position.
    if (allWsToken[i]) continue;
    const end = trimmed[i]!;
    for (let t = 0; t < tags.length; t++) {
      const el = exactLens[t]!;
      if (end >= el && joined.slice(end - el, end) === tags[t]) {
        lastExact[t] = i;
      }
      const fl = fusedLens[t]!;
      if (end >= fl && joined.slice(end - fl, end) === fused[t]) {
        lastFused[t] = i;
      }
    }
  }
  for (let t = 0; t < tags.length; t++) {
    const exact = lastExact[t]!;
    out[t] = exact >= 0
      ? positionLogprobs[exact + 1]
      : lastFused[t]! >= 0
        ? positionLogprobs[lastFused[t]! + 1]
        : undefined;
  }
  return out;
}

export function findTagLogprobs(
  reply: VerifierReply,
  tag: string,
): PositionLogprobs | undefined {
  return scanTagMatches(reply, [tag])[0];
}

/** Normalize a raw expectation over the scale to [0, 1]. */
function normalize(raw: number): number {
  if (MAX_VAL <= MIN_VAL) return 0.5;
  return (raw - MIN_VAL) / (MAX_VAL - MIN_VAL);
}

export type ScoreEvidenceSource = "logprobs" | "text_fallback" | "missing";

interface ExtractedScore {
  score: number;
  source: ScoreEvidenceSource;
  support: number;
  probabilityMass: number;
}

/** Expected score plus the evidence source that produced it. */
function scoreFromPosition(
  reply: VerifierReply,
  tag: string,
  position: PositionLogprobs | undefined,
): ExtractedScore {
  const probs: Record<number, number> = {};
  if (position) {
    for (const [tokStr, logprob] of position) {
      if (!Number.isFinite(logprob)) continue;
      let tok = tokStr.trim();
      if (tok.startsWith(">")) tok = tok.slice(1).trim();
      const raw = LETTER_VALUES[tok];
      if (raw !== undefined) {
        const probability = Math.exp(logprob);
        if (Number.isFinite(probability) && probability > 0) {
          probs[raw] = Math.max(probs[raw] ?? 0, probability);
        }
      }
    }
  }
  if (Object.keys(probs).length > 0) {
    const totalP = Object.values(probs).reduce((a, b) => a + b, 0);
    const expected =
      Object.entries(probs).reduce((a, [v, p]) => a + Number(v) * p, 0) /
      totalP;
    return {
      score: normalize(expected),
      source: "logprobs",
      support: Object.keys(probs).length,
      probabilityMass: totalP,
    };
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
    if (raw !== undefined) {
      return {
        score: normalize(raw),
        source: "text_fallback",
        support: 0,
        probabilityMass: 0,
      };
    }
  }
  return { score: 0.5, source: "missing", support: 0, probabilityMass: 0 };
}

/**
 * Expected score over the verifier's token distribution at `tag`,
 * normalized to [0, 1]. Falls back to parsing the literal `<tag> X </tag>`
 * text when no logprobs were returned.
 */
export function extractScore(reply: VerifierReply, tag: string): number {
  return scoreFromPosition(reply, tag, findTagLogprobs(reply, tag)).score;
}

/** Whether a reply contains a usable score for `tag`. */
export function hasExtractableScore(reply: VerifierReply, tag: string): boolean {
  return scoreFromPosition(reply, tag, findTagLogprobs(reply, tag)).source !== "missing";
}

export interface ScorePair {
  scoreA: number;
  scoreB: number;
  extractableA: boolean;
  extractableB: boolean;
  sourceA: ScoreEvidenceSource;
  sourceB: ScoreEvidenceSource;
  supportA: number;
  supportB: number;
  probabilityMassA: number;
  probabilityMassB: number;
}

/**
 * Both fine-grained rewards and their extractability from ONE linear scan of
 * the reply. The reference pipeline scores `<score_A>` and `<score_B>` from
 * the same response; a single pass over the token stream locates both
 * distributions, so PPT scoring of a long verifier reply parses once instead
 * of four separate O(n) tag scans.
 */
export function extractScorePair(
  reply: VerifierReply,
): ScorePair {
  const [posA, posB] = scanTagMatches(reply, ["<score_A>", "<score_B>"]);
  const scoreA = scoreFromPosition(reply, "<score_A>", posA);
  const scoreB = scoreFromPosition(reply, "<score_B>", posB);
  return {
    scoreA: scoreA.score,
    scoreB: scoreB.score,
    extractableA: scoreA.source !== "missing",
    extractableB: scoreB.source !== "missing",
    sourceA: scoreA.source,
    sourceB: scoreB.source,
    supportA: scoreA.support,
    supportB: scoreB.support,
    probabilityMassA: scoreA.probabilityMass,
    probabilityMassB: scoreB.probabilityMass,
  };
}
