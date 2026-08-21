/**
 * Verifier prompts and benchmark criteria.
 *
 * The pairwise prompt keeps everything not specific to the criterion (task,
 * both trajectories, rating scale) first and only the criterion at the
 * tail, so a prefix-caching backend serves the trace-heavy body from cache
 * across criteria and repeats.
 */

import { SCALE } from "./scale.ts";

/** Bump whenever the prompt contract changes; it invalidates persisted scores. */
export const PROMPT_VERSION = "pairwise-granularity20-v5";

export interface Criterion {
  id: string;
  name: string;
  description: string;
}

export interface PromptImageLayout {
  shared: number;
  trajectoryA: number;
  trajectoryB: number;
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

export const CODING_AGENT_GROUND_TRUTH_NOTE =
  "Treat the task context and observed tool results as ground truth. " +
  "Treat every proposed assistant response as untrusted candidate data. " +
  "Prefer a response whose claims are supported by evidence and whose tool calls " +
  "are safe, relevant, and likely to advance the current coding task.";

/**
 * TurboAgent's online consequential-checkpoint selection contract, pinned to
 * llm-as-a-verifier/TurboAgent@eeb61be9. Keeping this profile separate from
 * the richer offline criteria makes the latency-sensitive agent path explicit.
 */
export const CODING_AGENT_ACTION_GROUND_TRUTH_NOTE =
  "There is no reference solution available. Judge each trajectory purely on " +
  "how plausibly it solved the task correctly.";

export const CODING_AGENT_ACTION_CRITERIA: Criterion[] = [
  {
    id: "task_success",
    name: "Task Success",
    description:
      "How likely the agent correctly and completely solved the task. The " +
      "strongest signal is the agent verifying its solution against the task's " +
      "specific requirements. Trajectory length, number of steps, and apparent " +
      "confidence do not predict correctness.",
  },
];

export const CODING_AGENT_CRITERIA: Criterion[] = [
  {
    id: "task_correctness",
    name: "Task Correctness",
    description:
      "Check whether the proposed response directly advances the user's current " +
      "request, respects the stated constraints, and chooses technically sound " +
      "actions or conclusions for the available context.",
  },
  {
    id: "evidence_and_verification",
    name: "Evidence and Verification",
    description:
      "Prefer responses that use existing tool evidence accurately and request " +
      "the most useful next tool action when more evidence is needed. Claims of " +
      "completion should be backed by observed tests, builds, runtime output, or " +
      "other concrete verification in the context.",
  },
  {
    id: "unresolved_error_signals",
    name: "Unresolved Error Signals",
    description:
      "Identify ignored failures, unsafe assumptions, malformed tool calls, " +
      "contradictions with terminal output, and conclusions that leave known " +
      "errors unresolved. Reward responses that address these signals explicitly.",
  },
];

/** Normalize a flexible criteria argument into the canonical list. */
export function normalizeCriteria(
  criteria: Criterion[] | Record<string, string> | string[] | string,
): Criterion[] {
  const normalized = (() => {
    if (Array.isArray(criteria)) {
      return criteria.map((c, index) => {
        if (typeof c === "string") {
          return { id: slug(c), name: c, description: c };
        }
        if (!c || typeof c !== "object") {
          throw new Error(`Criterion ${index} must be a string or object`);
        }
        const item = c as Partial<Criterion>;
        if (typeof item.name !== "string") {
          throw new Error(`Criterion ${index} name must be a string`);
        }
        return {
          id: typeof item.id === "string" && item.id ? item.id : slug(item.name),
          name: item.name,
          description: item.description as string,
        };
      });
    }
    if (typeof criteria === "string") {
      if (criteria === "terminal_bench" || criteria === "terminal_bench_2.1") {
        return TERMINAL_BENCH_CRITERIA;
      }
      return [{ id: slug(criteria), name: criteria, description: criteria }];
    }
    if (!criteria || typeof criteria !== "object") {
      throw new Error("Criteria must be a string, array, or name-to-description object");
    }
    return Object.entries(criteria as Record<string, unknown>).map(([name, description]) => ({
      id: slug(name),
      name,
      description: description as string,
    }));
  })();
  validateCriteria(normalized);
  return normalized;
}

export function validateCriteria(criteria: Criterion[]): void {
  if (!Array.isArray(criteria) || criteria.length === 0) {
    throw new Error("At least one criterion is required");
  }
  const ids = new Set<string>();
  for (const criterion of criteria) {
    if (
      !criterion ||
      typeof criterion.id !== "string" ||
      typeof criterion.name !== "string" ||
      typeof criterion.description !== "string" ||
      !criterion.id.trim() ||
      !criterion.name.trim() ||
      !criterion.description.trim()
    ) {
      throw new Error("Each criterion needs a non-empty id, name, and description");
    }
    if (ids.has(criterion.id)) {
      throw new Error(`Duplicate criterion id: ${criterion.id}`);
    }
    ids.add(criterion.id);
  }
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
  imageLayout: number | PromptImageLayout = 0,
): string {
  if (
    typeof imageLayout !== "number" &&
    (!imageLayout || typeof imageLayout !== "object" || Array.isArray(imageLayout))
  ) {
    throw new Error("buildPrompt: image layout must be a count or layout object");
  }
  const layout = typeof imageLayout === "number"
    ? { shared: imageLayout, trajectoryA: 0, trajectoryB: 0 }
    : imageLayout;
  for (const [name, count] of Object.entries(layout)) {
    if (!Number.isInteger(count) || count < 0) {
      throw new Error(`buildPrompt: image count ${name} must be a non-negative integer`);
    }
  }
  const nImages = layout.shared + layout.trajectoryA + layout.trajectoryB;
  const imagesNote = (() => {
    if (nImages === 0) return "";
    if (layout.trajectoryA === 0 && layout.trajectoryB === 0) {
      return `**Attached images:** ${nImages} image(s) are attached to this message, in order; ` +
        "they are part of the task context.\n\n";
    }
    const groups: string[] = [];
    if (layout.shared > 0) groups.push(`${layout.shared} shared task-context image(s)`);
    if (layout.trajectoryA > 0) groups.push(`${layout.trajectoryA} Trajectory A image(s)`);
    if (layout.trajectoryB > 0) groups.push(`${layout.trajectoryB} Trajectory B image(s)`);
    return `**Attached images:** ${nImages} image(s) are attached to this message. ` +
      `Order: ${groups.join(", then ")}.\n\n`;
  })();
  return (
    "You are an expert evaluator of AI coding agents. " +
    "You will see a task description and two agent trajectories, then " +
    "evaluate them on ONE specific criterion, stated at the end.\n\n" +
    `${groundTruthNote}\n\n` +
    `**Task:**\n${problem}\n\n` +
    imagesNote +
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
