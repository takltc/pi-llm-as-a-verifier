/**
 * Strict Terminal-Bench trajectory loaders.
 *
 * The loader keeps held-out `reward` values separate from the verifier input:
 * only the task text and rendered trajectory enter the prompt. Invalid files
 * fail with their path and reason so a benchmark cannot silently score a
 * partial or malformed dataset.
 */

import { readdirSync, readFileSync, statSync, type Dirent } from "node:fs";
import { basename, join } from "node:path";
import type { ImageContent } from "@oh-my-pi/pi-ai";

export interface Trial {
  trialName: string;
  reward: 0 | 1;
  problem: string;
  trace: string;
  /** Images shared by every candidate for this task. */
  images?: readonly ImageContent[];
  /** Images produced inside this candidate trajectory. */
  trajectoryImages?: readonly ImageContent[];
}

export type Tasks = Record<string, Trial[]>;

export interface TbStep {
  step_id?: number | string;
  source?: string;
  message?: string;
  tool_calls?: Array<{
    arguments?: { keystrokes?: string } | Record<string, unknown>;
  }>;
  observation?: { results?: Array<{ content?: string }> };
}

interface TbTrajectoryFile {
  trial_name?: unknown;
  reward?: unknown;
  trajectory?: { steps?: unknown };
}

interface CandidateFile {
  task?: unknown;
  name?: unknown;
  trace?: unknown;
}

export interface LoadedCandidate {
  name: string;
  trace: string;
}

export interface CandidateSet {
  task: string;
  candidates: LoadedCandidate[];
}

export type InputLayout = "terminal" | "candidates";

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
      for (const toolCall of step.tool_calls ?? []) {
        const args = (toolCall.arguments ?? {}) as { keystrokes?: unknown };
        if (typeof args.keystrokes === "string" && args.keystrokes) {
          parts.push(`[Command] ${args.keystrokes.trimEnd()}`);
        }
      }
      for (const result of step.observation?.results ?? []) {
        if (typeof result.content === "string" && result.content) {
          parts.push(`[Output]\n${result.content}`);
        }
      }
      parts.push("");
    }
  }
  return parts.join("\n").trim() || "(no trajectory data)";
}

export function extractProblem(steps: TbStep[], taskName: string): string {
  for (const step of steps) {
    if (step.source === "user") {
      const message = step.message ?? "";
      if (message && !(message.startsWith("$") && message.length < 5)) {
        return message;
      }
    }
  }
  const initialAnalysis: string[] = [];
  for (const step of steps) {
    if (step.source !== "agent") continue;
    if (step.message) initialAnalysis.push(step.message);
    if (initialAnalysis.length >= 2) break;
  }
  if (initialAnalysis.length > 0) {
    return (
      `[Task: ${taskName}]\n` +
      "The original task instruction was not captured. Below is the agent's initial analysis:\n\n" +
      initialAnalysis.join("\n\n")
    );
  }
  return `(Task: ${taskName})`;
}

function requireDirectory(dir: string): void {
  let stats: ReturnType<typeof statSync>;
  try {
    stats = statSync(dir);
  } catch {
    throw new Error(`Trajectory directory does not exist: ${dir}`);
  }
  if (!stats.isDirectory()) throw new Error(`Trajectory path is not a directory: ${dir}`);
}

function parseJson(path: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid trajectory JSON at ${path}: ${detail}`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Trajectory JSON must be an object: ${path}`);
  }
  return parsed as Record<string, unknown>;
}

function parseSteps(value: unknown, path: string): TbStep[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Missing trajectory object at ${path}`);
  }
  const steps = (value as { steps?: unknown }).steps;
  if (!Array.isArray(steps) || steps.length === 0) {
    throw new Error(`Trajectory has no steps at ${path}`);
  }
  const invalidIndex = steps.findIndex(
    (step) => !step || typeof step !== "object" || Array.isArray(step),
  );
  if (invalidIndex >= 0) {
    throw new Error(`Trajectory step ${invalidIndex} must be an object at ${path}`);
  }
  return steps as TbStep[];
}

function parseReward(value: unknown, path: string): 0 | 1 {
  if (value === 0 || value === 1) return value;
  throw new Error(`Trajectory reward must be numeric 0 or 1 at ${path}`);
}

function trialFromFile(path: string, taskName: string): Trial {
  const data = parseJson(path) as TbTrajectoryFile;
  const steps = parseSteps(data.trajectory, path);
  const trace = formatTrace({ steps });
  if (trace === "(no trajectory data)") {
    throw new Error(`Trajectory has no agent steps at ${path}`);
  }
  const trialName =
    typeof data.trial_name === "string" && data.trial_name.trim()
      ? data.trial_name.trim()
      : basename(path);
  return {
    trialName,
    reward: parseReward(data.reward, path),
    problem: extractProblem(steps, taskName),
    trace,
  };
}

function candidateFromFile(
  path: string,
): LoadedCandidate & { task: string } {
  const data = parseJson(path) as CandidateFile;
  if (typeof data.task !== "string" || !data.task.trim()) {
    throw new Error(`Candidate task must be a non-empty string at ${path}`);
  }
  if (typeof data.trace !== "string" || !data.trace.trim()) {
    throw new Error(`Candidate trace must be a non-empty string at ${path}`);
  }
  if (data.name !== undefined && (typeof data.name !== "string" || !data.name.trim())) {
    throw new Error(`Candidate name must be a non-empty string at ${path}`);
  }
  return {
    task: data.task.trim(),
    name:
      typeof data.name === "string"
        ? data.name.trim()
        : basename(path, ".json"),
    trace: data.trace.trim(),
  };
}

export function loadTrajectoryFile(path: string, taskName = "task"): Trial {
  return trialFromFile(path, taskName);
}

function entries(dir: string): Dirent<string>[] {
  return readdirSync(dir, { withFileTypes: true, encoding: "utf8" }).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

function jsonFiles(dir: string): string[] {
  return entries(dir)
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => join(dir, entry.name));
}

function terminalTaskDirs(dir: string): string[] {
  return entries(dir)
    .filter((entry) => entry.isDirectory())
    .map((entry) => join(dir, entry.name))
    .filter((taskDir) =>
      entries(taskDir).some(
        (entry) => entry.isFile() && entry.name.endsWith("_trajectory.json"),
      ),
    );
}

/** Determine the layout from valid trajectory filenames, not arbitrary subdirs. */
export function detectInputLayout(dir: string): InputLayout {
  requireDirectory(dir);
  const terminalDirs = terminalTaskDirs(dir);
  const flatFiles = jsonFiles(dir);
  if (terminalDirs.length > 0 && flatFiles.length > 0) {
    throw new Error(
      `Mixed trajectory layouts in ${dir}: found task subdirectories and root JSON files`,
    );
  }
  if (terminalDirs.length > 0) return "terminal";
  if (flatFiles.length > 0) return "candidates";
  throw new Error(
    `No trajectory JSON files found in ${dir}; expected <task>/*_trajectory.json or flat *.json`,
  );
}

/** Load `<dir>/<task>/*_trajectory.json` files in stable task/file order. */
export function loadTerminalDir(dir: string): { tasks: Tasks; nRuns: number } {
  requireDirectory(dir);
  const tasks: Tasks = {};
  for (const taskDir of terminalTaskDirs(dir)) {
    const taskName = basename(taskDir);
    const files = entries(taskDir)
      .filter((entry) => entry.isFile() && entry.name.endsWith("_trajectory.json"))
      .map((entry) => join(taskDir, entry.name));
    const trials = files.map((path) => trialFromFile(path, taskName));
    if (trials.length > 0) tasks[taskName] = trials;
  }
  if (Object.keys(tasks).length === 0) {
    throw new Error(`No valid Terminal-Bench task trajectories found in ${dir}`);
  }
  return {
    tasks,
    nRuns: Math.max(...Object.values(tasks).map((trials) => trials.length)),
  };
}

/** Load one selection request from flat `{ task, trace, name? }` JSON files. */
export function loadCandidateDir(dir: string): CandidateSet {
  requireDirectory(dir);
  const files = jsonFiles(dir);
  if (files.length < 2) {
    throw new Error(`At least two candidate JSON files are required in ${dir}`);
  }
  const loaded = files.map(candidateFromFile);
  const task = loaded[0].task;
  for (const candidate of loaded.slice(1)) {
    if (candidate.task !== task) {
      throw new Error(
        `Candidate task mismatch in ${dir}: all files must use the same task`,
      );
    }
  }
  const seenNames = new Set<string>();
  for (const candidate of loaded) {
    if (seenNames.has(candidate.name)) {
      throw new Error(`Duplicate candidate name in ${dir}: ${candidate.name}`);
    }
    seenNames.add(candidate.name);
  }
  return {
    task,
    candidates: loaded.map(({ name, trace }) => ({ name, trace })),
  };
}

/** Split all-pass and swing tasks; all-fail tasks are unwinnable. */
export function classify(tasks: Tasks): { allPass: string[]; swing: string[] } {
  const allPass: string[] = [];
  const swing: string[] = [];
  for (const name of Object.keys(tasks).sort()) {
    const trials = tasks[name];
    if (trials.every((trial) => trial.reward === 1)) allPass.push(name);
    else if (!trials.every((trial) => trial.reward === 0)) swing.push(name);
  }
  return { allPass, swing };
}
