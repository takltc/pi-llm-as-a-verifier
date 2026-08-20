# omp-llm-verifier

## Citations

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 reference implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

The plugin applies the paper's 20-level token-logprob scoring and Probabilistic Pivot Tournament. Each ordinary OMP request generates three complete candidates, verifies them with the active OMP default model, and replays the winning response, including text, reasoning, images, tool calls, and the stop state.

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

The verifier model must expose OpenAI Chat Completions or Responses token logprobs — the paper's fine-grained reward is read off the score-token distribution, so a model without token logprobs cannot run the paper's verifier. When the resolved verifier model cannot provide logprobs, the plugin refuses to wrap the session model and shows a capability warning instead of degrading every request.

Verified comparisons are cached on disk at `.omp-llm-verifier-cache.json` in the project root, keyed by a fingerprint of the task, both candidates, criteria, model, and prompt version, so repeat verifications of identical content cost no verifier tokens. The file is safe to delete and worth git-ignoring.

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
