# omp-llm-verifier

用于 oh-my-pi（OMP）的 LLM-as-a-Verifier 自验证插件。它自动使用 OMP `modelRoles.default` 的模型、凭据和请求头，通过细粒度 logprob 奖励与 Probabilistic Pivot Tournament 从多条候选轨迹中选择最佳结果。

## 引用

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [参考实现：Self-Verification / Terminal-Bench 2.1](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

插件固化参考实现的 Terminal-Bench 2.1 Bo5 设置：`pivots=1`、`K=2`、`seed=0`。验证器沿用 OMP 默认模型及其推理等级；该模型需要通过 OpenAI Chat Completions 或 Responses API 提供 token `logprobs`。

## 安装与使用

```bash
omp plugin install https://github.com/takltc/pi-llm-as-a-verifier.git
omp plugin doctor
```

启动 OMP 后可直接使用：

```text
/verify ./trajectories
```

`./trajectories` 放置至少两个 JSON 文件，每个文件只需任务与完整轨迹：

```json
{
  "task": "修复项目中的失败测试并验证结果",
  "trace": "候选执行轨迹与终端输出"
}
```

所有候选使用相同的 `task`。可选 `name` 用于结果展示；文件名会作为默认名称。Terminal-Bench 的 `<task>/*_trajectory.json` 目录可直接用于复现实验。

对话中的模型可直接调用 `verifier_select`。该工具只有两个顶层输入：任务描述 `task` 与候选轨迹 `candidates`；每条候选只需提供 `trace`，并可选提供 `name`。

## 本地开发

```bash
git clone https://github.com/takltc/pi-llm-as-a-verifier.git omp-llm-verifier
cd omp-llm-verifier
bun install --frozen-lockfile
bun test
bun run typecheck
omp plugin link "$PWD"
omp plugin doctor
```

启动 OMP 后，使用 `/verify <traj_dir>` 或让模型调用 `verifier_select` 验证本地改动。
