/**
 * Verifier prompts and benchmark criteria.
 *
 * The pairwise prompt keeps everything not specific to the criterion (task,
 * both trajectories, rating scale) first and only the criterion at the
 * tail, so a prefix-caching backend serves the trace-heavy body from cache
 * across criteria and repeats.
 */

import { SCALE } from "./scale.ts";

export interface Criterion {
  id: string;
  name: string;
  description: string;
}

export const GROUND_TRUTH_NOTE =
  "**IMPORTANT:** Focus on TERMINAL OUTPUT as ground truth. Do NOT trust " +
  "the agent's self-assessment or claims of success. Agents often claim " +
  "success when the terminal shows errors.";

export const TERMINAL_BENCH_CRITERIA: Criterion[] = [
  {
    id: "specification",
    name: "Specification Adherence",
    description:
      "Re-read the task description and check the SPECIFIC requirements: " +
      "exact file paths, install locations, output formats, naming, and any " +
      "explicit constraints (e.g. \"no X11 support\", \"install to " +
      "/usr/local/bin/X\", \"output JSON to /app/out.json\"). Did the agent " +
      "meet these specific requirements, or did they produce a solution that " +
      "solves a similar but different problem (right idea, wrong place / " +
      "wrong format / missing constraint)?",
  },
  {
    id: "output_match",
    name: "Output Match",
    description:
      "Find the FINAL verification command the agent ran (the one that " +
      "should prove the solution works). Compare its actual stdout/stderr " +
      "output, character-by-character if needed, to what the task " +
      "description says the output should look like. For example: if the " +
      "task says it should print \"Results: X Y Z\" with integers, did the " +
      "agent's last test actually print that? If the task asks for a JSON " +
      "file, do the values look plausible and well-formed in the cat " +
      "output? Reward trajectories whose terminal SHOWS the expected output " +
      "literally. Ignore everything except whether the observed output " +
      "matches the expected output.",
  },
  {
    id: "error_signals",
    name: "Error Signal Detection",
    description:
      "Scan the trajectory — especially the later steps — for explicit " +
      "failure markers: error messages, exception tracebacks, segmentation " +
      "faults, \"command not found\", \"No such file or directory\", " +
      "non-zero exit codes that the agent did not subsequently fix, " +
      "compilation failures, test failures, etc. A trajectory that ends " +
      "with unresolved errors is almost certainly broken even if the agent " +
      "claims success. Conversely, a clean trajectory whose final commands " +
      "all succeed without errors is a strong positive signal. Score based " +
      "ONLY on the presence/absence of unresolved error signals.",
  },
];

/** Normalize a flexible criteria argument into the canonical list. */
export function normalizeCriteria(
  criteria: Criterion[] | Record<string, string> | string[] | string,
): Criterion[] {
  if (Array.isArray(criteria)) {
    return criteria.map((c) =>
      typeof c === "string"
        ? { id: slug(c), name: c, description: c }
        : { id: c.id || slug(c.name), name: c.name, description: c.description },
    );
  }
  if (typeof criteria === "string") {
    if (criteria === "terminal_bench" || criteria === "terminal_bench_2.1") {
      return TERMINAL_BENCH_CRITERIA;
    }
    return [{ id: slug(criteria), name: criteria, description: criteria }];
  }
  return Object.entries(criteria).map(([name, description]) => ({
    id: slug(name),
    name,
    description,
  }));
}

function slug(text: string): string {
  return (
    text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 40) || "criterion"
  );
}

/**
 * One pairwise prompt focused on a single criterion. Everything before the
 * criterion block is a shared prefix across criteria for the same
 * (task, slot-A, slot-B) — keep criterion text strictly at the end.
 */
export function buildPrompt(
  problem: string,
  traceA: string,
  traceB: string,
  criterion: Criterion,
  groundTruthNote = GROUND_TRUTH_NOTE,
): string {
  return (
    "You are an expert evaluator of AI coding agents. " +
    "You will see a task description and two agent trajectories, then " +
    "evaluate them on ONE specific criterion, stated at the end.\n\n" +
    `${groundTruthNote}\n\n` +
    `**Task:**\n${problem}\n\n` +
    `**Trajectory A:**\n${traceA}\n\n` +
    `**Trajectory B:**\n${traceB}\n\n` +
    `**Rating Scale:**\n${SCALE.scaleDescription}\n\n` +
    `**Evaluation Guideline — ${criterion.name}:**\n` +
    `${criterion.description}\n\n` +
    `Score each trajectory ONLY on this specific criterion ` +
    `("${criterion.name}"). Ignore other aspects of the trajectory that are ` +
    "not relevant to it.\n\n" +
    "Reason it through first, then END your reply with exactly these two " +
    "lines and nothing after them. Replace each placeholder with a single " +
    "letter A-T, keeping the spaces around the letter exactly as shown:\n" +
    `<score_A> ${SCALE.scoreFormat} </score_A>\n` +
    `<score_B> ${SCALE.scoreFormat} </score_B>\n\n` +
    "Begin your analysis now."
  );
}
