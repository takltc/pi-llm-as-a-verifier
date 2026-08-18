/**
 * Trajectory loaders.
 *
 * Primary format: Terminal-Bench 2.1 mini-swe-agent output — a directory of
 * `<task>/<trial>_trajectory.json` files, each with
 * `{ reward, trial_name, trajectory: { steps: [...] } }`. The verifier only
 * ever sees `problem` and `trace`; `reward` is the held-out ground truth
 * used to score selection.
 *
 * The same step format is accepted for a plain list of candidate JSON files
 * (one candidate per file), so self-verification works on any agent rollout
 * that records steps with source/message/tool_calls/observation.
 */

export interface Trial {
  trialName: string;
  reward: 0 | 1;
  problem: string;
  trace: string;
}

export type Tasks = Record<string, Trial[]>;

interface TbStep {
  step_id?: number | string;
  source?: string;
  message?: string;
  tool_calls?: Array<{
    arguments?: { keystrokes?: string } | Record<string, unknown>;
  }>;
  observation?: { results?: Array<{ content?: string }> };
}

interface TbTrajectoryFile {
  trial_name?: string;
  reward?: number;
  trajectory?: { steps?: TbStep[] };
}

export function formatTrace(trajectory: { steps?: TbStep[] } | undefined): string {
  if (!trajectory?.steps?.length) return "(no trajectory data)";
  const parts: string[] = [];
  for (const step of trajectory.steps) {
    const source = step.source ?? "";
    const message = step.message ?? "";
    if (source === "system" || source === "user") continue;
    if (source === "agent") {
      parts.push(`--- Agent Step ${step.step_id ?? "?"} ---`);
      if (message) parts.push(message);
      for (const tc of step.tool_calls ?? []) {
        const args = (tc.arguments ?? {}) as { keystrokes?: string };
        const keystrokes = args.keystrokes ?? "";
        if (keystrokes) parts.push(`[Command] ${keystrokes.trimEnd()}`);
      }
      const results = step.observation?.results ?? [];
      for (const result of results) {
        if (result.content) parts.push(`[Output]\n${result.content}`);
      }
      parts.push("");
    }
  }
  return parts.join("\n");
}

export function extractProblem(steps: TbStep[], taskName: string): string {
  for (const step of steps) {
    if (step.source === "user") {
      const msg = step.message ?? "";
      if (msg && !(msg.startsWith("$") && msg.length < 5)) return msg;
    }
  }
  const parts: string[] = [];
  for (const step of steps) {
    if (step.source !== "agent") continue;
    const msg = step.message ?? "";
    if (msg) parts.push(msg);
    if (parts.length >= 2) break;
  }
  if (parts.length > 0) {
    return (
      `[Task: ${taskName}]\n` +
      "The original task instruction was not captured. Below is the " +
      "agent's initial analysis:\n\n" +
      parts.join("\n\n")
    );
  }
  return `(Task: ${taskName})`;
}

function trialFromFile(d: TbTrajectoryFile, taskName: string): Trial | null {
  const trajectory = d.trajectory;
  if (!trajectory) return null;
  const steps = trajectory.steps ?? [];
  return {
    trialName: d.trial_name ?? "",
    reward: d.reward ? 1 : 0,
    problem: extractProblem(steps, taskName),
    trace: formatTrace(trajectory),
  };
}

/**
 * Load a Terminal-Bench style directory: `<dir>/<task>/*_trajectory.json`.
 * Returns (tasks, nRuns) where tasks maps task name -> trials.
 */
export function loadTerminalDir(dir: string): { tasks: Tasks; nRuns: number } {
  const tasks: Tasks = {};
  for (const taskDir of sortedDirEntries(dir)) {
    const taskName = basename(taskDir);
    const trajFiles = sortedDirEntries(taskDir).filter((f) =>
      f.endsWith("_trajectory.json"),
    );
    const trials: Trial[] = [];
    for (const trajFile of trajFiles) {
      const d = JSON.parse(awaitRead(trajFile)) as TbTrajectoryFile;
      const trial = trialFromFile(d, taskName);
      if (trial) trials.push(trial);
    }
    if (trials.length > 0) tasks[taskName] = trials;
  }
  const nRuns = Math.max(0, ...Object.values(tasks).map((t) => t.length));
  return { tasks, nRuns };
}

/**
 * Load a single task from a directory of candidate JSON files
 * (`<dir>/<trial>.json`), each in the same trajectory shape. Returns
 * (tasks, nRuns) with one task whose name is `taskName`.
 */
export function loadCandidateDir(
  dir: string,
  taskName: string,
): { tasks: Tasks; nRuns: number } {
  const files = sortedDirEntries(dir).filter((f) => f.endsWith(".json"));
  const trials: Trial[] = [];
  for (const file of files) {
    const d = JSON.parse(awaitRead(file)) as TbTrajectoryFile;
    const trial = trialFromFile(d, taskName);
    if (trial) {
      trial.trialName = trial.trialName || basename(file);
      trials.push(trial);
    }
  }
  const tasks: Tasks = {};
  if (trials.length > 0) tasks[taskName] = trials;
  return { tasks, nRuns: trials.length };
}

/**
 * Split tasks into all-pass (every trial succeeds) and swing tasks.
 * All-fail tasks (every trial fails) are unwinnable and need no
 * verification, mirroring `scripts/run.py#classify` in the reference.
 */
export function classify(tasks: Tasks): { allPass: string[]; swing: string[] } {
  const allPass: string[] = [];
  const swing: string[] = [];
  for (const [name, trials] of Object.entries(tasks)) {
    if (trials.every((t) => t.reward === 1)) allPass.push(name);
    else if (!trials.every((t) => t.reward === 0)) swing.push(name);
  }
  return { allPass, swing };
}

import { readdirSync, readFileSync } from "node:fs";

// -- tiny fs helpers -------------------------------------------------------

function sortedDirEntries(dir: string): string[] {
  return readdirSync(dir)
    .map((name) => `${dir}/${name}`)
    .sort();
}

function basename(p: string): string {
  const parts = p.split("/");
  return parts[parts.length - 1] ?? p;
}

function awaitRead(path: string): string {
  return readFileSync(path, "utf8");
}
