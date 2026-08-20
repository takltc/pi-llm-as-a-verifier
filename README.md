# omp-llm-verifier

## Citations

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 reference implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

The plugin applies the paper's 20-level token-logprob scoring and Probabilistic Pivot Tournament. Each terminal OMP answer expands to three complete candidates by default, scores them with the configured verifier model (the session default when unset), and replays the winning response with its text, reasoning, images, and stop state. Tool-use turns continue the active trajectory directly.

The versioned sources, formal invariants, product boundaries, and drift gates live in [docs/theory-baseline.md](docs/theory-baseline.md).

See the [Chinese guide](README.zh-CN.md) for the same workflow in Chinese.

## Install and use

```bash
omp plugin install https://github.com/takltc/pi-llm-as-a-verifier.git
omp plugin config set omp-llm-verifier enabled true
omp plugin config set omp-llm-verifier candidateCount 3
omp plugin doctor
```

`enabled` controls automatic verification. `candidateCount` accepts 2-8 and defaults to 3. Settings take effect in a new OMP session. By default the verifier inherits the OMP `modelRoles.default` model, thinking level, credentials, headers, and compatibility settings.

Following the paper's agent≠verifier setup (the reference scores GPT/Claude trajectories with Gemini or DeepSeek), a different verifier model can be pinned with an OMP model selector:

```bash
omp plugin config set omp-llm-verifier verifierModel deepseek/deepseek-v4-flash
```

Leave `verifierModel` empty to verify with the session default model (self-verification). kimi-code (Kimi for Coding) rejects logprobs requests, so with kimi-code as the session model a `verifierModel` override is required.

Disable automatic verification while keeping the plugin installed:

```bash
omp plugin config set omp-llm-verifier enabled false
```

Disable or re-enable plugin loading:

```bash
omp plugin disable omp-llm-verifier
omp plugin enable omp-llm-verifier
```

The verifier model must expose OpenAI Chat Completions or Responses token logprobs — the paper's fine-grained reward is read off the score-token distribution, so a model without token logprobs cannot run the paper's verifier. When the resolved verifier model cannot provide logprobs, the plugin refuses to wrap the session model and shows a capability warning instead of degrading every request. Tasks with visual evidence also require image input support on the verifier model. Every pairwise request carries shared user/assistant/tool-result images first, followed by labeled Trajectory A and Trajectory B images, preserving the reference implementation's multimodal evidence path.

Verified comparisons are cached on disk at `.omp-llm-verifier-cache.json` in the project root, keyed by a fingerprint of the task, ordered shared and candidate-specific images, both candidate traces, criteria, model, and prompt version, so repeat verifications of identical content cost no verifier tokens. The file is safe to delete and worth git-ignoring.

Every verified answer is observable: on the OMP console the wrapper logs one
`event:decision` JSON line per final answer with `path` (how the winner was
chosen — `verifier` for the paper's PPT tournament, `fallback` when verification
could not run, plus `aborted` and `error` terminal states), the
winner's candidate index and mean score, the directed-comparison count, the
verifier model and prompt contract that scored it, and the verifier token
usage for that request. `scoreSources` separately counts logprob expectations,
literal-text fallbacks, runtime neutral ties, and legacy/unknown cache entries;
`paperEquivalent` is true only when every score tag came from token logprobs.
`scoreDistribution` reports the minimum and mean valid A–T support and returned
probability mass for those logprob-backed tags.
The same data is exposed to extensions through the
`onDecision` callback on the wrapped provider state.

Every terminal selection with at least two successful terminal candidates runs
the paper's PPT path, including cases where several candidate texts are identical.
Tool-use turns continue the active agent trajectory directly. Alternative samples
that propose another tool action remain outside the terminal-answer tournament and
are reported through `nonterminalCandidates`.

## Local development

```bash
git clone https://github.com/takltc/pi-llm-as-a-verifier.git omp-llm-verifier
cd omp-llm-verifier
bun install --frozen-lockfile
bun test
bun run typecheck
omp plugin link "$PWD"
omp plugin doctor
```

After editing, rerun the tests, typecheck, and plugin doctor before starting OMP.
