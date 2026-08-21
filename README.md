# omp-llm-verifier

## Citations

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 reference implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)
- [TurboAgent coding-agent extension](https://github.com/llm-as-a-verifier/TurboAgent/tree/eeb61be9cb618ea9c52262cebf15092e7c185146)

The paper defines three reward scopes: outcome reward models (ORM), process reward models (PRM), and trajectory reward models (TRM). This plugin selects PRM for the interactive coding-agent path and applies it at consequential checkpoints. It first samples one action. A batch containing only audited OMP observation tools proceeds with that single sample. Terminal responses and any state-changing, control-flow, `write`, `exec`, unknown, malformed, or unclassified tool action expand to `candidateCount` total samples (2–8, default 3); the additional samples run concurrently, then exact majority or the paper's Probabilistic Pivot Tournament (PPT) selects one winner. Only the selected consequential tool action reaches the agent loop.

This remains the paper's PRM scheme with an adaptive sampling budget: Appendix B.3 evaluates per-step action sampling at `k=1,3,5,9`, and §4 states that verification compute is tunable to the downstream latency budget. OMP's argument-aware approval tier plus an audited effect adapter supplies the product checkpoint boundary. Operation-conditioned routing is an engineering scheduling policy inside PRM, and its benefit is measured separately from the paper's reported results. The verifier algorithm, 20-level token-logprob expectation, and PPT remain unchanged. See [the paper](https://arxiv.org/html/2607.05391v2), [the detailed granularity study](docs/granularity-selection.md), and [the theory baseline](docs/theory-baseline.md).

The versioned sources, formal invariants, product boundaries, and drift gates live in [docs/theory-baseline.md](docs/theory-baseline.md).

See the [Chinese guide](README.zh-CN.md) for the same workflow in Chinese.

## Install and use

```bash
omp plugin install https://github.com/takltc/pi-llm-as-a-verifier.git
omp plugin config set omp-llm-verifier enabled true
omp plugin config set omp-llm-verifier candidateCount 3
omp plugin doctor
```

`enabled` controls automatic verification. `candidateCount` is the total sample budget for consequential checkpoints, accepts 2-8, and defaults to 3. Audited observation tools use one sample. Read-tier tools that commit state/control flow, plus unclassified extension tools, remain consequential checkpoints. Settings take effect in a new OMP session. By default the verifier inherits the OMP `modelRoles.default` model, thinking level, credentials, headers, and compatibility settings.

Following the paper's agent≠verifier setup (the reference scores GPT/Claude trajectories with Gemini or DeepSeek), a different verifier model can be pinned with an OMP model selector:

```bash
omp plugin config set omp-llm-verifier verifierModel deepseek/deepseek-v4-flash
```

Leave `verifierModel` empty to verify with the session default model (self-verification). kimi-code (Kimi for Coding) rejects logprobs requests, so with kimi-code as the session model a `verifierModel` override is required.

Consequential checkpoints use TurboAgent's online selector profile (`K=1`,
`pivots=2`, one Task Success criterion). TurboAgent's reference N is 3; this
plugin exposes N as `candidateCount` with a 2–8 range. The first sample warms
the shared provider prefix before the remaining N-1 samples fan out. The
paper's quality/cost axes remain configurable:

```bash
# Independent repeated verifications per criterion (paper §4.2): the paper's
# main experiments use K=8; the online default is K=1 for latency.
omp plugin config set omp-llm-verifier nEvaluations 8
# PPT pivot count k (paper §3.2); TurboAgent's online default is 2.
omp plugin config set omp-llm-verifier pivots 2
```

`nEvaluations` accepts 1-16 and `pivots` accepts 1-8; both are integer
quality/cost knobs and take effect in a new OMP session.

Disable automatic verification while keeping the plugin installed:

```bash
omp plugin config set omp-llm-verifier enabled false
```

Disable or re-enable plugin loading:

```bash
omp plugin disable omp-llm-verifier
omp plugin enable omp-llm-verifier
```

The verifier model must expose OpenAI Chat Completions or Responses token logprobs — the paper's fine-grained reward is read off the score-token distribution, so a model without token logprobs cannot run the paper's verifier. When the resolved verifier model cannot provide logprobs, the plugin refuses to wrap the session model and shows a capability warning instead of degrading every request. A probe timeout or transport failure also keeps the original agent model active and retries capability probing after 60 seconds. Tasks with visual evidence also require image input support on the verifier model. Every pairwise request carries shared user/assistant/tool-result images first, followed by labeled Trajectory A and Trajectory B images, preserving the reference implementation's multimodal evidence path.

OMP's `--max-time` is an absolute deadline for the whole agent loop, so candidate generation and PPT share that total budget. Omitting it leaves normal interactive sessions without an agent deadline. High-effort headless/CI runs should budget for the complete multi-action turn, for example `--max-time 15m`; each verifier HTTP request allows up to 10 minutes while still honoring OMP's session cancellation signal.

Verified comparisons are cached on disk at `.omp-llm-verifier-cache.json` in the project root, keyed by a fingerprint of the task, ordered shared and candidate-specific images, both candidate traces, criteria, model, and prompt version, so repeat verifications of identical content cost no verifier tokens. The file is safe to delete and worth git-ignoring.

Every process step is observable: on the OMP console the wrapper logs one
`event:decision` JSON line per coding-agent model request with
`granularity=prm` and `path` (`single` for an audited observation,
`majority` for TurboAgent's exact-action shortcut, `verifier` for PPT,
`fallback` for a recoverable degraded selection, plus `aborted` and `error`
terminal states), the
winner's candidate index and mean score, the directed-comparison count, the
verifier model and prompt contract that scored it, and the verifier token
usage for that request. `scoreSources` separately counts logprob expectations,
literal-text fallbacks, runtime neutral ties, and legacy/unknown cache entries;
`paperEquivalent` is true only when every score tag came from token logprobs.
`scoreDistribution` reports the minimum and mean valid A–T support and returned
probability mass for those logprob-backed tags.
The decision also reports `sampledCandidates`, `checkpointReason`,
`toolUseCandidates`, `terminalCandidates`, `discardedCandidates` (in-flight
candidate requests cancelled once a strict
action majority `count > N/2` became unavoidable), and the
winning stop reason. The same data is exposed to extensions through the
`onDecision` callback on the wrapped provider state.

The proposal, candidate expansion, and PPT keep their content buffered until
replay. Single-sample observations use OMP's normal working UI. A consequential
checkpoint shows `Expanding a consequential action to N candidates…` and
`Verifying N candidate actions with PPT…`. The wrapper restores the default
loader before replay so a prior tool intent does not leak across process steps.

OMP utility calls such as automatic title generation stay on a single
original-model track. Process-reward scheduling applies to coding-agent
actions. Reasoning-capable utility models receive the caller's selected effort
on the first dispatch, or the model's lowest supported effort when the utility
caller selected none. If stale model metadata still allows a dynamic endpoint
to reject `disableReasoning` with a mandatory-reasoning 400, the wrapper records
that endpoint capability and retries once. A user's `high` or `max` coding
effort therefore remains unchanged.

Candidate generation preserves the caller's full context, tools, session ID,
prompt-cache key, and provider session state across every sample. Completing
the first proposal before consequential fan-out warms that identical coding
prefix; the remaining N-1 samples run concurrently with high cache affinity.
Provider-native `execHandlers` are rejected before sampling because
they can execute side effects during generation; declarative tool calls remain
fully supported and only the selected call is replayed.

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
