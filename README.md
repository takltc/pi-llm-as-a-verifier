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
   `opencode-go/deepseek-v4-flash:xhigh`）生成 N 条候选轨迹，再用**同一模型**
   作为验证者挑选最佳——verifier 判断的是自己模型的工作，仍能显著超过 Pass@1。

## 为什么直接调用 API

omp 会话 API 不暴露 token 级 logprobs，而细粒度奖励依赖它。插件直接调用
opencode-go 端点（`https://opencode.ai/zen/go/v1`，OpenAI 兼容）获取
`logprobs` + `top_logprobs`。凭据解析顺序：

1. 环境变量 `OPENCODE_API_KEY`
2. omp auth store（`~/.omp/agent/agent.db` 中 `opencode-go` 的登录凭据）

端点与模型可覆盖：`OPENCODE_BASE_URL`、`--effort`、`--max-tokens`。

## 安装

```bash
# 从本目录
bun install
```

插件加载到 omp 的三种方式任选其一：

```bash
# 1) 会话内临时加载
omp -e /path/to/pi-llm-as-a-verifier/src/index.ts

# 2) 用户级扩展目录（每次会话自动加载）
mkdir -p ~/.omp/agent/extensions
ln -s /path/to/pi-llm-as-a-verifier/src/index.ts ~/.omp/agent/extensions/llm-verifier.ts

# 3) 配置扩展路径
# ~/.omp/agent/config.yml 增加:
#   extensions:
#     - /path/to/pi-llm-as-a-verifier/src/index.ts
```

## 使用

### `/verify <traj_dir>` — 批量自验证

轨迹目录使用 Terminal-Bench 布局（`<dir>/<task>/*_trajectory.json`，
每个文件含 `reward` 与 `trajectory.steps`），或一个平铺目录（候选 JSON 列表）。

```text
/verify data/e2e --pivots 2 --k 2 --seed 0 --workers 8
/verify data/e2e --tasks bn-fit-modify,cancel-async-tasks
/verify data/e2e --effort high --cache .verifier-cache.json
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
pivots: 2, nEvaluations: 2
```

返回最佳候选、各候选 mean preference（w_i/c_i）、排序、比较次数。

## CLI（独立运行，不依赖 omp 会话）

```bash
bun run src/cli.ts <traj_dir> [--pivots N] [--k N] [--seed N]
                 [--workers N] [--effort xhigh] [--max-tokens N]
                 [--cache <path>] [--trials N] [--tasks a,b]
```

## 验证

```bash
bun test                 # extract_score + PPT 单元测试
bun run src/cli.ts data/e2e --k 2 --pivots 2 --seed 0 --workers 8
                         # 端到端真实 API 验证（opencode-go/deepseek-v4-flash:xhigh）
```

### 实测结果（Terminal-Bench 2.1 轨迹，verifier = deepseek-v4-flash:xhigh）

| 任务 | trials 奖励 | Pass@1 | LLM-as-a-Verifier | Oracle (Bo5) |
|---|---|---|---|---|
| bn-fit-modify | [1,1,1,0,1] | 80% | **100%** | 100% |
| cancel-async-tasks | [1,1,0,0,1] | 60% | **100%** | 100% |
| dna-assembly | [1,1,0,0,0] | 40% | **100%** | 100% |
| gcode-to-text | [1,0,0,0,1] | 40% | **100%** | 100% |

平均 Pass@1 55% → **verifier 100%**（4/4 任务选到 ground-truth 最优轨迹），
与论文结论一致：自验证显著超过 Pass@1。缓存复用重跑时命中率 96.6%，
同 seed 结果完全复现；失败调用按 tie（0.5/0.5）兜底且不污染缓存。

## 设计要点

- **前缀缓存**：成对提示词把 criterion 放在尾部，同一 (task, slot-A, slot-B)
  的 3 个 criterion 共享前缀；评分时每个 distinct prefix 先打一个请求预热，
  其余并发请求命中缓存。
- **有界并发**：默认 16 个并发 worker，失败按 `on_error="tie"` 记 0.5/0.5
  （不写入缓存），避免单次失败中断整个 tournament。
- **确定性**：同一 seed 生成相同的 ring（mulberry32），可复现。
- **xhigh 预算**：xhigh 推理与答案共享输出预算，默认 `max_tokens=65536`；
  若调用因推理耗尽预算而无 logprobs，会显式报错并提示调整。
