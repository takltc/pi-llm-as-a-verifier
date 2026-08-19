/** OMP extension entry point for LLM-as-a-Verifier self-verification. */

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { isAbsolute, resolve } from "node:path";
import {
  DEFAULT_MODEL_SELECTOR,
  VerifierClient,
  formatUsage,
} from "./client.ts";
import {
  GROUND_TRUTH_NOTE,
  TERMINAL_BENCH_CRITERIA,
  buildPrompt,
  normalizeCriteria,
} from "./prompt.ts";
import { extractScore, hasExtractableScore } from "./scale.ts";
import {
  detectInputLayout,
  loadCandidateDir,
  loadTerminalDir,
  loadTrajectoryFile,
  type Tasks,
} from "./loader.ts";
import { renderReport, runBenchmark } from "./run.ts";
import { select } from "./select.ts";

export default function verifierExtension(pi: ExtensionAPI): void {
  pi.setLabel("LLM-as-a-Verifier");

  pi.registerCommand("verify", {
    description:
      "Run Terminal-Bench-style best-of-N self-verification. " +
      "Usage: /verify <traj_dir> [--pivots 1] [--k 2] [--seed 0] " +
      "[--workers 16] [--model opencode-go/deepseek-v4-flash:xhigh]",
    handler: async (args, ctx) => {
      try {
        const parsed = parseArgs(args);
        assertKnownOptions(parsed, [
          "pivots",
          "k",
          "evaluations",
          "seed",
          "workers",
          "effort",
          "model",
          "max-tokens",
          "cache",
          "trials",
          "tasks",
          "note",
          "help",
        ]);
        if (parsed.opts.help) {
          ctx.ui.notify(
            "/verify <traj_dir> [--pivots 1] [--k 2] [--seed 0] [--workers 16] " +
              "[--trials 5] [--tasks a,b] [--cache path] [--effort xhigh]",
            "info",
          );
          return;
        }
        if (parsed.positionals.length > 1) {
          throw new Error("/verify accepts one trajectory directory");
        }
        const dir = resolvePath(ctx.cwd, parsed.positionals[0] || ctx.cwd);
        ctx.ui.notify(`LLM-as-a-Verifier: loading trajectories from ${dir} …`, "info");

        let tasks = loadTasks(dir);
        const trialLimit = parsed.optInt("trials");
        if (trialLimit !== undefined) {
          if (trialLimit < 2) throw new Error("--trials must be at least 2");
          tasks = Object.fromEntries(
            Object.entries(tasks).map(([name, trials]) => [name, trials.slice(0, trialLimit)]),
          );
        }
        const taskFilter = parsed.opts.tasks;
        if (taskFilter) {
          const keep = new Set(
            taskFilter.split(",").map((name) => name.trim()).filter(Boolean),
          );
          if (keep.size === 0) throw new Error("--tasks requires at least one task name");
          tasks = Object.fromEntries(
            Object.entries(tasks).filter(([name]) => keep.has(name)),
          );
          if (Object.keys(tasks).length === 0) {
            throw new Error("--tasks did not match any loaded task");
          }
        }

        const cacheFile = resolvePath(
          ctx.cwd,
          parsed.opts.cache || ".verifier-cache.json",
        );
        const client = new VerifierClient({
          model: parsed.opts.model,
          effort: parsed.opts.effort,
          maxTokens: parsed.optInt("max-tokens"),
        });
        const stats = await runBenchmark(tasks, TERMINAL_BENCH_CRITERIA, {
          pivots: parsed.optInt("pivots"),
          nEvaluations: parsed.optInt("k") ?? parsed.optInt("evaluations"),
          seed: parsed.optInt("seed"),
          maxWorkers: parsed.optInt("workers"),
          cacheFile,
          client,
          groundTruthNote:
            parsed.opts.note === undefined ? GROUND_TRUTH_NOTE : parsed.opts.note,
        });
        const winners = Object.entries(stats.bestPerTask)
          .map(
            ([task, result]) =>
              `  ${task}: trial #${result.index} (reward=${result.reward}, ` +
              `w=${result.w.toFixed(3)}, c=${result.c})`,
          )
          .join("\n");
        const report = [
          renderReport(stats),
          formatUsage(stats.usage).join("\n"),
          winners ? `\nWinners per task (verifier choice, ground-truth reward):\n${winners}` : "",
        ]
          .filter(Boolean)
          .join("\n");
        console.log(report);
        ctx.ui.notify(
          `Verifier: ${stats.verifier}/${stats.nTasks} tasks ` +
            `(${((100 * stats.verifier) / stats.nTasks).toFixed(1)}%) — ` +
            `Oracle Bo${stats.nRuns}: ${((100 * stats.oracle) / stats.nTasks).toFixed(1)}%`,
          "info",
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`verify failed: ${message}`);
        ctx.ui.notify(`verify failed: ${message}`, "error");
      }
    },
  });

  pi.registerCommand("vcompare", {
    description:
      "Score two trajectory JSON files with the Terminal-Bench criteria. " +
      "Usage: /vcompare <a.json> <b.json> [--effort xhigh]",
    handler: async (args, ctx) => {
      try {
        const parsed = parseArgs(args);
        assertKnownOptions(parsed, ["effort", "model", "max-tokens", "note"]);
        const [fileA, fileB] = parsed.positionals;
        if (!fileA || !fileB || parsed.positionals.length !== 2) {
          throw new Error("Usage: /vcompare <a.json> <b.json>");
        }
        const trialA = loadTrajectoryFile(resolvePath(ctx.cwd, fileA), "task");
        const trialB = loadTrajectoryFile(resolvePath(ctx.cwd, fileB), "task");
        const note = parsed.opts.note === undefined ? GROUND_TRUTH_NOTE : parsed.opts.note;
        const client = new VerifierClient({
          model: parsed.opts.model,
          effort: parsed.opts.effort,
          maxTokens: parsed.optInt("max-tokens"),
        });
        const output: string[] = [];
        for (const criterion of TERMINAL_BENCH_CRITERIA) {
          const reply = await client.scoreReply(
            buildPrompt(trialA.problem, trialA.trace, trialB.trace, criterion, note),
          );
          if (!hasExtractableScore(reply, "<score_A>") || !hasExtractableScore(reply, "<score_B>")) {
            throw new Error(`Verifier response omitted score tags for ${criterion.id}`);
          }
          output.push(
            `  ${criterion.id}: A=${extractScore(reply, "<score_A>").toFixed(4)}  ` +
              `B=${extractScore(reply, "<score_B>").toFixed(4)}`,
          );
        }
        console.log(`Comparison ${fileA} vs ${fileB}:\n${output.join("\n")}`);
        ctx.ui.notify("vcompare done", "info");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`vcompare failed: ${message}`, "error");
      }
    },
  });

  const z = pi.zod;
  pi.registerTool({
    name: "verifier_select",
    label: "Verifier Select",
    description:
      "Select the strongest candidate trajectory using the 20-level " +
      "logprob reward and Probabilistic Pivot Tournament. Default verifier: " +
      DEFAULT_MODEL_SELECTOR +
      ".",
    parameters: z.object({
      task: z.string().min(1).describe("Task description shown to the verifier."),
      candidates: z
        .array(
          z.object({
            name: z.string().describe("Short candidate label."),
            trace: z.string().min(1).describe("Full trajectory including terminal output."),
          }),
        )
        .min(2)
        .max(20),
      criteria: z.array(z.string().min(1)).optional(),
      pivots: z.number().int().min(1).max(20).optional().describe("Pivot count (default 1)."),
      nEvaluations: z.number().int().min(1).max(8).optional().describe("Repeats per criterion (default 2)."),
      seed: z.number().int().optional(),
      maxWorkers: z.number().int().min(1).max(64).optional(),
      model: z.string().optional().describe(`OMP model selector (default ${DEFAULT_MODEL_SELECTOR}).`),
      effort: z.string().optional(),
      maxTokens: z.number().int().min(1).optional(),
      cacheFile: z.string().optional(),
      groundTruthNote: z.string().optional(),
    }),
    async execute(_toolCallId, rawParams, signal, _onUpdate, ctx) {
      const params = rawParams as {
        task: string;
        candidates: Array<{ name: string; trace: string }>;
        criteria?: string[];
        pivots?: number;
        nEvaluations?: number;
        seed?: number;
        maxWorkers?: number;
        model?: string;
        effort?: string;
        maxTokens?: number;
        cacheFile?: string;
        groundTruthNote?: string;
      };
      const client = new VerifierClient({
        model: params.model,
        effort: params.effort,
        maxTokens: params.maxTokens,
      });
      const result = await select(params.task, params.candidates, {
        criteria: params.criteria
          ? normalizeCriteria(params.criteria)
          : TERMINAL_BENCH_CRITERIA,
        pivots: params.pivots,
        nEvaluations: params.nEvaluations,
        seed: params.seed,
        maxWorkers: params.maxWorkers,
        cacheFile: params.cacheFile
          ? resolvePath(ctx.cwd, params.cacheFile)
          : undefined,
        groundTruthNote: params.groundTruthNote,
        signal: signal ?? undefined,
        progress: false,
        client,
      });
      return {
        content: [
          {
            type: "text",
            text: [
              `Best candidate: #${result.index} (${result.best})`,
              `Scores: ${result.scores.map((score) => score.toFixed(4)).join(", ")}`,
              `Ranking: ${result.ranking.join(" > ")}`,
              `Comparisons: ${result.nComparisons}; criteria: ${result.criteria.join(",")}`,
            ].join("\n"),
          },
        ],
        details: result,
      };
    },
  });

  pi.on("session_start", async (_event, ctx) => {
    const client = new VerifierClient();
    if (!client.ready) {
      ctx.ui.notify(
        "LLM-as-a-Verifier needs an opencode-go credential; run `/login opencode-go` " +
          "or set OPENCODE_API_KEY.",
        "info",
      );
      return;
    }
    ctx.ui.notify(
      `LLM-as-a-Verifier ready (opencode-go/${client.model}:${client.effort}). ` +
        "Use /verify, /vcompare, or verifier_select.",
      "info",
    );
  });
}

function loadTasks(dir: string): Tasks {
  return detectInputLayout(dir) === "terminal"
    ? loadTerminalDir(dir).tasks
    : loadCandidateDir(dir, "task").tasks;
}

function resolvePath(cwd: string, value: string): string {
  return isAbsolute(value) ? value : resolve(cwd, value);
}

export interface ParsedArgs {
  positionals: string[];
  opts: Record<string, string>;
  optInt(name: string): number | undefined;
}

export function tokenizeArgs(input: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  let escaped = false;
  let started = false;
  const push = () => {
    if (started) tokens.push(current);
    current = "";
    started = false;
  };

  for (const char of input) {
    if (escaped) {
      current += char;
      escaped = false;
      started = true;
      continue;
    }
    if (quote) {
      if (char === quote) {
        quote = undefined;
        started = true;
      } else if (char === "\\" && quote === '"') {
        escaped = true;
      } else {
        current += char;
        started = true;
      }
      continue;
    }
    if (char === "'" || char === '"') {
      quote = char;
      started = true;
    } else if (char === "\\") {
      escaped = true;
      started = true;
    } else if (/\s/.test(char)) {
      push();
    } else {
      current += char;
      started = true;
    }
  }
  if (escaped) throw new Error("Trailing escape in command arguments");
  if (quote) throw new Error("Unterminated quote in command arguments");
  push();
  return tokens;
}

export function parseArgs(args: string): ParsedArgs {
  const tokens = tokenizeArgs(args);
  const positionals: string[] = [];
  const opts: Record<string, string> = {};
  let optionsEnded = false;
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if (!optionsEnded && token === "--") {
      optionsEnded = true;
      continue;
    }
    if (!optionsEnded && token.startsWith("--")) {
      const raw = token.slice(2);
      if (!raw) throw new Error("Invalid empty option");
      const equals = raw.indexOf("=");
      const name = equals >= 0 ? raw.slice(0, equals) : raw;
      if (!name) throw new Error(`Invalid option: ${token}`);
      if (equals >= 0) {
        opts[name] = raw.slice(equals + 1);
      } else if (index + 1 < tokens.length && !tokens[index + 1].startsWith("--")) {
        opts[name] = tokens[++index];
      } else {
        opts[name] = "true";
      }
    } else {
      positionals.push(token);
    }
  }
  return {
    positionals,
    opts,
    optInt(name) {
      const raw = opts[name];
      if (raw === undefined) return undefined;
      if (raw === "true") throw new Error(`--${name} requires an integer value`);
      if (!/^-?\d+$/.test(raw)) throw new Error(`--${name} must be an integer`);
      const value = Number(raw);
      if (!Number.isSafeInteger(value)) throw new Error(`--${name} is outside the safe integer range`);
      return value;
    },
  };
}

function assertKnownOptions(parsed: ParsedArgs, allowed: string[]): void {
  const known = new Set(allowed);
  for (const option of Object.keys(parsed.opts)) {
    if (!known.has(option)) throw new Error(`Unknown option: --${option}`);
  }
}
