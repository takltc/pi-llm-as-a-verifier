/** OMP extension entry point for LLM-as-a-Verifier self-verification. */

import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";
import { extractExplicitThinkingSelector } from "@oh-my-pi/pi-coding-agent/config/model-resolver";
import { isAbsolute, resolve } from "node:path";
import {
  createVerifierClient,
  formatUsage,
} from "./client.ts";
import {
  TERMINAL_BENCH_CRITERIA,
} from "./prompt.ts";
import {
  detectInputLayout,
  loadCandidateDir,
  loadTerminalDir,
  type LoadedCandidate,
} from "./loader.ts";
import { renderReport, runBenchmark } from "./run.ts";
import { select, type SelectResult } from "./select.ts";

const SELECTION_DEFAULTS = {
  criteria: TERMINAL_BENCH_CRITERIA,
  onError: "raise",
  progress: false,
} as const;

export default function verifierExtension(pi: ExtensionAPI): void {
  pi.setLabel("LLM-as-a-Verifier");

  pi.registerCommand("verify", {
    description: "验证轨迹目录并自动选出最佳候选。用法：/verify <traj_dir>",
    handler: async (args, ctx) => {
      try {
        const parsed = parseArgs(args);
        assertKnownOptions(parsed, ["help"]);
        if (parsed.opts.help) {
          ctx.ui.notify("用法：/verify <traj_dir>", "info");
          return;
        }
        if (parsed.positionals.length !== 1) {
          throw new Error("用法：/verify <traj_dir>");
        }
        const dir = resolvePath(ctx.cwd, parsed.positionals[0]);
        ctx.ui.notify(`LLM-as-a-Verifier: loading trajectories from ${dir} …`, "info");

        const client = await createDefaultVerifierClient(pi, ctx);
        const cacheFile = resolvePath(ctx.cwd, ".verifier-cache.json");
        if (detectInputLayout(dir) === "terminal") {
          const stats = await runBenchmark(
            loadTerminalDir(dir).tasks,
            TERMINAL_BENCH_CRITERIA,
            { cacheFile, client, onError: "raise" },
          );
          const winners = Object.entries(stats.bestPerTask)
            .map(
              ([task, result]) =>
                `  ${task}: trial #${result.index} (reward=${result.reward}, ` +
                `w=${result.w.toFixed(3)}, c=${result.c})`,
            )
            .join("\n");
          console.log(
            [
              renderReport(stats),
              formatUsage(stats.usage).join("\n"),
              winners
                ? `\nWinners per task (verifier choice, ground-truth reward):\n${winners}`
                : "",
            ]
              .filter(Boolean)
              .join("\n"),
          );
          ctx.ui.notify(
            `Verifier: ${stats.verifier}/${stats.nTasks} tasks ` +
              `(${((100 * stats.verifier) / stats.nTasks).toFixed(1)}%) — ` +
              `Oracle Bo${stats.nRuns}: ${((100 * stats.oracle) / stats.nTasks).toFixed(1)}%`,
            "info",
          );
          return;
        }

        const input = loadCandidateDir(dir);
        const result = await select(input.task, input.candidates, {
          cacheFile,
          ...SELECTION_DEFAULTS,
          client,
        });
        console.log(renderSelectionReport(input.task, input.candidates, result));
        ctx.ui.notify(
          `Verifier selected #${result.index} (${result.best}), ` +
            `score=${result.scores[result.index].toFixed(4)}`,
          "info",
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`verify failed: ${message}`);
        ctx.ui.notify(`verify failed: ${message}`, "error");
      }
    },
  });

  const z = pi.zod;
  pi.registerTool({
    name: "verifier_select",
    label: "Verifier Select",
    description: "在候选轨迹中自动选出最佳结果。使用当前 OMP 默认模型和论文默认验证设置。",
    parameters: z.object({
      task: z.string().min(1).describe("Task description shown to the verifier."),
      candidates: z
        .array(
          z.object({
            name: z.string().min(1).optional().describe("Optional candidate label."),
            trace: z.string().min(1).describe("Full trajectory including terminal output."),
          }),
        )
        .min(2)
        .max(20),
    }),
    async execute(_toolCallId, rawParams, signal, _onUpdate, ctx) {
      const params = rawParams as {
        task: string;
        candidates: Array<{ trace: string; name?: string }>;
      };
      const client = await createDefaultVerifierClient(pi, ctx);
      const result = await select(params.task, params.candidates, {
        cacheFile: resolvePath(ctx.cwd, ".verifier-cache.json"),
        ...SELECTION_DEFAULTS,
        signal: signal ?? undefined,
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
    try {
      const client = await createDefaultVerifierClient(pi, ctx);
      ctx.ui.notify(
        "LLM-as-a-Verifier 已就绪，使用 OMP 默认模型 " +
          client.provider + "/" + client.model + ":" + client.effort +
          "。可直接调用 verifier_select 或 /verify <traj_dir>。",
        "info",
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      ctx.ui.notify(
        "LLM-as-a-Verifier 暂不可用：" + message,
        "info",
      );
    }
  });
}

export function createDefaultVerifierClient(
  pi: ExtensionAPI,
  ctx: Pick<ExtensionContext, "models" | "modelRegistry">,
) {
  const settings = pi.pi.settings;
  return createVerifierClient({
    ...ctx,
    defaultThinkingLevel: extractExplicitThinkingSelector(
      settings.getModelRole("default"),
      settings,
    ),
  });
}

export function renderSelectionReport(
  task: string,
  candidates: LoadedCandidate[],
  result: SelectResult,
): string {
  const ranked = result.ranking.map(
    (index, rank) =>
      `  ${rank + 1}. #${index} ${candidates[index].name} ` +
      `score=${result.scores[index].toFixed(4)}`,
  );
  return [
    "LLM-as-a-Verifier selection",
    `Task: ${task}`,
    `Best candidate: #${result.index} (${result.best})`,
    "Ranking:",
    ...ranked,
    `Comparisons: ${result.nComparisons}`,
    formatUsage(result.usage).join("\n"),
  ].join("\n");
}

function resolvePath(cwd: string, value: string): string {
  return isAbsolute(value) ? value : resolve(cwd, value);
}

export interface ParsedArgs {
  positionals: string[];
  opts: Record<string, string>;
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
  return { positionals, opts };
}

function assertKnownOptions(parsed: ParsedArgs, allowed: string[]): void {
  const known = new Set(allowed);
  for (const option of Object.keys(parsed.opts)) {
    if (!known.has(option)) throw new Error(`Unknown option: --${option}`);
  }
}
