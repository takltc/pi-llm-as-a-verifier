/**
 * omp LLM-as-a-Verifier plugin.
 *
 * Implements the LLM-as-a-Verifier framework (arXiv:2607.05391) inside omp:
 * fine-grained, logprob-based reward scoring and Probabilistic Pivot
 * Tournament best-of-N selection, using the same model the session already
 * uses — `opencode-go/deepseek-v4-flash:xhigh` by default — as the
 * self-verifier. The verifier reads token-level logprobs directly from the
 * opencode-go endpoint, so no session-model round-trip is involved.
 *
 * Surfaces:
 *   - `/verify <traj_dir>`        run self-verification over a trajectory
 *                                 directory (Terminal-Bench 2.1 layout), pick
 *                                 the best trial per task, report Pass@1 vs
 *                                 verifier vs oracle.
 *   - `/vcompare <a.json> <b.json>` fine-grained reward for one comparison.
 *   - tool `verifier_select`      best-of-N selection callable by the agent.
 */

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { statSync, readdirSync, readFileSync } from "node:fs";
import { VerifierClient, formatUsage } from "./client.ts";
import { buildPrompt, TERMINAL_BENCH_CRITERIA } from "./prompt.ts";
import { extractScore } from "./scale.ts";
import {
  loadCandidateDir,
  loadTerminalDir,
  formatTrace,
  extractProblem,
} from "./loader.ts";
import { runBenchmark, renderReport } from "./run.ts";
import { select } from "./select.ts";

export default function (pi: ExtensionAPI) {
  pi.setLabel("LLM-as-a-Verifier");

  // ------------------------------------------------------------------
  // /verify — batch self-verification over a trajectory directory
  // ------------------------------------------------------------------
  pi.registerCommand("verify", {
    description:
      "Run LLM-as-a-Verifier best-of-N selection over trajectory files " +
      "(Terminal-Bench layout: <dir>/<task>/*_trajectory.json) and report " +
      "Pass@1 / verifier / oracle. Options: --pivots N --k N --seed N " +
      "--workers N --effort xhigh --cache <path> --trials N --tasks a,b",
    handler: async (args, ctx) => {
      const cwd = ctx.cwd;
      const parsed = parseArgs(args);
      const dir = parsed.positionals[0] || cwd;
      const options = {
        pivots: parsed.optInt("pivots"),
        nEvaluations: parsed.optInt("k") ?? parsed.optInt("evaluations"),
        seed: parsed.optInt("seed"),
        maxWorkers: parsed.optInt("workers"),
        effort: parsed.opts["effort"],
        maxTokens: parsed.optInt("max-tokens"),
      };

      ctx.ui.notify(
        `LLM-as-a-Verifier: loading trajectories from ${dir} …`,
        "info",
      );
      try {
        let tasks;
        if (fsIsDir(dir) && hasSubdirs(dir)) {
          ({ tasks } = loadTerminalDir(dir));
        } else {
          ({ tasks } = loadCandidateDir(dir, "task"));
        }
        if (Object.keys(tasks).length === 0) {
          ctx.ui.notify("No trajectory JSON files found.", "error");
          return;
        }
        const trialLimit = parsed.optInt("trials");
        if (trialLimit) {
          for (const name of Object.keys(tasks)) {
            tasks[name] = tasks[name].slice(0, trialLimit);
          }
        }
        const taskFilter = parsed.opts["tasks"];
        if (taskFilter) {
          const keep = new Set(taskFilter.split(",").map((s) => s.trim()));
          tasks = Object.fromEntries(
            Object.entries(tasks).filter(([name]) => keep.has(name)),
          );
        }
        const cacheFile =
          parsed.opts["cache"] || `${cwd}/.verifier-cache.json`;
        const client = new VerifierClient({
          effort: options.effort,
          maxTokens: options.maxTokens,
        });
        const stats = await runBenchmark(tasks, TERMINAL_BENCH_CRITERIA, {
          pivots: options.pivots,
          nEvaluations: options.nEvaluations,
          seed: options.seed,
          maxWorkers: options.maxWorkers,
          cacheFile,
          client,
          groundTruthNote: parsed.opts["note"] ?? "",
        });
        const report = renderReport(stats) + "\n" +
          formatUsage(stats.usage).join("\n") + "\n\n" +
          "Winners per task (verifier choice, ground-truth reward):\n" +
          Object.entries(stats.bestPerTask)
            .map(
              ([task, b]) =>
                `  ${task}: trial #${b.index} (reward=${b.reward}, ` +
                `w=${b.w.toFixed(3)}, c=${b.c})`,
            )
            .join("\n");
        console.log(report);
        ctx.ui.notify(
          `Verifier: ${stats.verifier}/${stats.nTasks} tasks ` +
            `(${((100 * stats.verifier) / stats.nTasks).toFixed(1)}%) — ` +
            `best-of-${stats.nRuns} Oracle: ` +
            `${((100 * stats.oracle) / stats.nTasks).toFixed(1)}%`,
          "info",
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error(e);
        ctx.ui.notify(`verify failed: ${msg}`, "error");
      }
    },
  });

  // ------------------------------------------------------------------
  // /vcompare — one directed comparison
  // ------------------------------------------------------------------
  pi.registerCommand("vcompare", {
    description:
      "Score one pairwise comparison between two trajectory JSON files. " +
      "Usage: /vcompare <a.json> <b.json> [--effort xhigh]",
    handler: async (args, ctx) => {
      const parsed = parseArgs(args);
      const [fa, fb] = parsed.positionals;
      if (!fa || !fb) {
        ctx.ui.notify("Usage: /vcompare <a.json> <b.json>", "error");
        return;
      }
      const resolve = (p: string) =>
        p.startsWith("/") ? p : `${ctx.cwd}/${p}`;
      try {
        const da = JSON.parse(readFileSync(resolve(fa), "utf8").toString()) as {
          trajectory?: { steps?: Array<Record<string, unknown>> };
        };
        const db = JSON.parse(readFileSync(resolve(fb), "utf8").toString()) as {
          trajectory?: { steps?: Array<Record<string, unknown>> };
        };
        const traceA = formatTrace(da.trajectory as never);
        const traceB = formatTrace(db.trajectory as never);
        const problem = extractProblem(
          (da.trajectory?.steps ?? []) as never,
          "task",
        );
        const client = new VerifierClient({ effort: parsed.opts["effort"] });
        const note = "";
        const out: string[] = [];
        for (const crit of TERMINAL_BENCH_CRITERIA) {
          const prompt = buildPrompt(problem, traceA, traceB, crit, note);
          const reply = await client.scoreReply(prompt);
          const ra = extractScore(reply, "<score_A>");
          const rb = extractScore(reply, "<score_B>");
          out.push(
            `  ${crit.id}: A=${ra.toFixed(4)}  B=${rb.toFixed(4)}`,
          );
        }
        const text = `Comparison ${fa} vs ${fb}:\n${out.join("\n")}`;
        console.log(text);
        ctx.ui.notify("vcompare done", "info");
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        ctx.ui.notify(`vcompare failed: ${msg}`, "error");
      }
    },
  });

  // ------------------------------------------------------------------
  // verifier_select tool — agent-callable best-of-N selection
  // ------------------------------------------------------------------
  const z = pi.zod;
  pi.registerTool({
    name: "verifier_select",
    label: "Verifier Select",
    description:
      "Select the best of N candidate agent trajectories for a task using " +
      "the LLM-as-a-Verifier framework (fine-grained logprob rewards + " +
      "Probabilistic Pivot Tournament, O(Nk) comparisons). Verifier model: " +
      "opencode-go/deepseek-v4-flash:xhigh by default. Use when you have " +
      "several candidate solutions/rollouts for the same task and want the " +
      "most likely correct one, or to rank alternatives.",
    parameters: z.object({
      task: z
        .string()
        .describe("The task description shown to the verifier."),
      candidates: z
        .array(
          z.object({
            name: z.string().describe("Short label for this candidate."),
            trace: z
              .string()
              .describe(
                "The full agent trajectory for this candidate: steps, " +
                  "commands run, terminal outputs, results.",
              ),
          }),
        )
        .min(2)
        .max(20)
        .describe("Candidate trajectories to rank (2-20)."),
      criteria: z
        .array(z.string())
        .optional()
        .describe(
          "Optional criteria names (default: terminal-bench criteria " +
            "Specification Adherence / Output Match / Error Signal Detection).",
        ),
      pivots: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Number of tournament pivots k (default 2)."),
      nEvaluations: z
        .number()
        .int()
        .min(1)
        .max(8)
        .optional()
        .describe("Repeated verifications K per criterion (default 2)."),
      cacheFile: z
        .string()
        .optional()
        .describe(
          "Optional path to reuse/update a score cache across calls.",
        ),
    }),
    async execute(_toolCallId, rawParams, signal, _onUpdate, ctx) {
      const params = rawParams as {
        task: string;
        candidates: Array<{ name: string; trace: string }>;
        criteria?: string[];
        pivots?: number;
        nEvaluations?: number;
        cacheFile?: string;
      };
      const criteria = params.criteria
        ? params.criteria.map((name: string) => ({
            id: name.toLowerCase().replace(/\s+/g, "_"),
            name,
            description: name,
          }))
        : TERMINAL_BENCH_CRITERIA;
      const result = await select(
        params.task,
        params.candidates.map(
          (c: { name: string; trace: string }, i: number) => ({
            name: c.name || `candidate_${i}`,
            trace: c.trace,
          }),
        ),
        {
          criteria,
          pivots: params.pivots,
          nEvaluations: params.nEvaluations,
          cacheFile: params.cacheFile,
          signal: signal ?? undefined,
          progress: false,
        },
      );
      const lines = [
        `Best candidate: #${result.index} (${result.best})`,
        `Scores (mean preference): ${result.scores
          .map((s) => s.toFixed(4))
          .join(", ")}`,
        `Ranking: ${result.ranking.join(" > ")}`,
        `Comparisons: ${result.nComparisons}, criteria: ${result.criteria.join(",")}`,
      ];
      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: result,
      };
    },
  });

  pi.on("session_start", async (_event, ctx) => {
    const client = new VerifierClient();
    if (!client.ready) {
      ctx.ui.notify(
        "LLM-as-a-Verifier: no opencode-go credential found — run " +
          "`/login opencode-go` or set OPENCODE_API_KEY.",
        "info",
      );
    } else {
      ctx.ui.notify(
        `LLM-as-a-Verifier ready (${client.model}@${client.effort}). ` +
          "Use /verify or /vcompare.",
        "info",
      );
    }
  });
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

interface ParsedArgs {
  positionals: string[];
  opts: Record<string, string>;
  optInt(name: string): number | undefined;
}

function parseArgs(args: string): ParsedArgs {
  const tokens = args.split(/\s+/).filter(Boolean);
  const positionals: string[] = [];
  const opts: Record<string, string> = {};
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t.startsWith("--")) {
      const name = t.slice(2);
      const eq = name.indexOf("=");
      if (eq >= 0) {
        opts[name.slice(0, eq)] = name.slice(eq + 1);
      } else if (i + 1 < tokens.length && !tokens[i + 1].startsWith("--")) {
        opts[name] = tokens[++i];
      } else {
        opts[name] = "true";
      }
    } else {
      positionals.push(t);
    }
  }
  return {
    positionals,
    opts,
    optInt(name) {
      const v = opts[name];
      if (v === undefined || v === "true") return undefined;
      const n = Number(v);
      return Number.isFinite(n) ? Math.trunc(n) : undefined;
    },
  };
}

function fsIsDir(p: string): boolean {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function hasSubdirs(p: string): boolean {
  try {
    return readdirSync(p, { withFileTypes: true }).some((e) => e.isDirectory());
  } catch {
    return false;
  }
}
