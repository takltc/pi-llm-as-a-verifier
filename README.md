# omp-llm-verifier

LLM-as-a-Verifier 插件（oh-my-pi / omp）。实现论文
[LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
中的自验证（Self-Verification）流程，参考
[llm-as-a-verifier/llm-as-a-verifier](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)
的 Terminal-Bench 2.1 复现脚本（`scripts/run_bo5.py`）。

## 核心方法

1. **细粒度奖励（Fine-grained reward）**：不把验证者的判断坍缩成单一离散分数，
   而是读取模型对有序评分 token 集合（A–T 共 20 级）的概率分布并取期望：

   $$R(x, \tau) = \frac{1}{CK}\sum_{c=1}^{C}\sum_{k=1}^{K}\sum_{g=1}^{G}
   p_\theta(v_g \mid x, c, \tau)\,\phi(v_g)$$

   logprob 来自真实 token 分布，因此比 LLM-as-a-Judge 的离散打分更可校准。
2. **Probabilistic Pivot Tournament (PPT)**：O(Nk) 的最佳-of-N 选择。
   ring pass 消除位置偏差 → 选 top-k pivots → pivot rounds 集中比较，
   总比较数 $N + k(N-k) + \binom{k}{2}$（对固定 k 关于 N 线性）。
3. **自验证（Self-Verification）**：同一模型（默认
   `opencode-go/deepseek-v4-flash:xhigh`）生成 N 条候选轨迹，再用同一模型
   作为验证者挑选最佳。

## 为什么直接调用 API

omp 会话 API 不暴露 token 级 logprobs，而细粒度奖励依赖它。插件直接调用
opencode-go 端点（`https://opencode.ai/zen/go/v1`，OpenAI 兼容）获取
`logprobs` + `top_logprobs`。凭据解析顺序：

1. 环境变量 `OPENCODE_API_KEY`
2. omp auth store（`~/.omp/agent/agent.db` 中 `opencode-go` 的登录凭据）

端点与模型可覆盖：`OPENCODE_BASE_URL`、`--model`、`--effort`、`--max-tokens`。

## 安装

```bash
# 从本目录
bun install
```

OMP 17.3.7 及以上支持把此包作为扩展加载：

```bash
# 用户级链接，后续会话自动发现 package.json 中的 omp.extensions
omp plugin link /path/to/pi-llm-as-a-verifier --scope user
omp plugin doctor
```

开发调试也可以使用会话内临时加载：

```bash
omp -e /path/to/pi-llm-as-a-verifier/src/index.ts
```

启动后，session start 会检查 `opencode-go` 凭据并提示可用命令。插件只报告
模型、评分和 token 用量等验证结果，凭据内容不会进入日志。

## 使用

### `/verify <traj_dir>` — 批量自验证

轨迹目录使用 Terminal-Bench 布局（`<dir>/<task>/*_trajectory.json`，
每个文件含 `reward` 与 `trajectory.steps`），或一个平铺目录（候选 JSON 列表）。

```text
/verify data/e2e --pivots 1 --k 2 --seed 0 --workers 8
/verify data/e2e --tasks bn-fit-modify,cancel-async-tasks
/verify data/e2e --model opencode-go/deepseek-v4-flash:xhigh --cache .verifier-cache.json
```

输出：每任务的 Pass@1 / LLM-as-a-Verifier / Oracle（Best-of-N）对比表、
各任务 winner 与 ground-truth reward、token 用量（含前缀缓存命中率）。

### `/vcompare <a.json> <b.json>` — 单次成对比较

按三个 Terminal-Bench 标准（Specification Adherence / Output Match /
Error Signal Detection）分别输出 A、B 的细粒度奖励。

### 工具 `verifier_select` — agent 可调用

```text
调用 verifier_select 工具：
task: <任务描述>
candidates: [{name, trace}, ...]
criteria: 可选（默认 terminal-bench 三条）
pivots: 1, nEvaluations: 2
```

返回最佳候选、各候选 mean preference（w_i/c_i）、排序、比较次数。

## CLI（独立运行，不依赖 omp 会话）

```bash
bun run src/cli.ts <traj_dir> [--pivots N] [--k N] [--seed N]
                 [--workers N] [--effort xhigh] [--max-tokens N]
                 [--model opencode-go/deepseek-v4-flash:xhigh]
                 [--cache <path>] [--trials N] [--tasks a,b]
```

## 验证

```bash
bun install --frozen-lockfile
bun test
bunx tsc --noEmit
bun run verify -- --help
omp plugin doctor
```

真实模型验证需要本机已有 `opencode-go` 凭据。建议先用 `/vcompare` 做单次
短轨迹 smoke，再运行完整 Terminal-Bench 2.1 Bo5；结果应记录实际任务数、
Pass@1、verifier、Oracle、比较次数、token 用量和缓存命中率。历史运行结果不作为
当前仓库的验证结论。

## 设计要点

- **前缀缓存**：成对提示词把 criterion 放在尾部，同一 (task, slot-A, slot-B)
  的 3 个 criterion 共享前缀；评分时每个 distinct prefix 先打一个请求预热，
  其余并发请求命中缓存。
- **缓存身份**：任务描述、候选轨迹、criterion id/name/description、模型、推理强度、
  输出预算、端点、ground-truth note 和提示版本共同生成内容指纹。锁文件含持有者
  token；缓存采用最新磁盘状态合并、临时文件 fsync 和原子 rename。
- **有界并发**：默认 16 个并发 worker，失败按 `on_error="tie"` 记 0.5/0.5
  （仅保留在本次运行内），避免单次失败中断整个 tournament。
- **确定性**：同一 seed 生成相同的 ring（mulberry32），可复现。
- **xhigh 预算**：xhigh 推理与答案共享输出预算，默认 `max_tokens=65536`；
  若调用因推理耗尽预算而无 logprobs，会显式报错并提示调整。
