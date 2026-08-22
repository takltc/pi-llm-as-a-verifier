# omp-llm-verifier 中文说明

## 引用

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 参考实现](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)
- [面向 coding agent 的 TurboAgent 扩展](https://github.com/llm-as-a-verifier/TurboAgent/tree/eeb61be9cb618ea9c52262cebf15092e7c185146)

论文定义 ORM（结果奖励模型）、PRM（过程奖励模型）和 TRM（轨迹奖励模型）三种奖励范围。本插件在交互式 coding agent 路径选择 PRM，并在高影响检查点应用 20 级 token logprob 评分和 Probabilistic Pivot Tournament。每一步先采样一个动作：所有调用都属于经过审计的 OMP 观测工具时，该步骤直接使用这一个样本；终态回复以及状态提交、控制流提交、`write`、`exec`、未知、格式异常或无法分类的工具动作会扩展至 `candidateCount` 个总样本，范围为 2–8、默认值为 3，随后通过精确多数决或 PPT 选择 winner。agent loop 只会执行被选中的高影响工具动作。

这套边界属于论文的 PRM 方案，并采用自适应采样预算。论文 Appendix B.3 对逐步动作采样的 `k=1,3,5,9` 做了实验，§4 明确把验证计算量作为可按下游延迟预算调节的参数；OMP 参数级 approval tier 与经过审计的 effect adapter 共同确定产品检查点。按操作影响面路由是 PRM 内部的工程调度策略，其收益与论文报告结果分开测量。验证器算法、20 级 token-logprob 期望和 PPT 均保持论文语义。来源见[论文](https://arxiv.org/html/2607.05391v2)、[粒度选择研究](docs/granularity-selection.md)和[理论基线](docs/theory-baseline.md)。

版本化来源、形式化不变量、产品边界和防漂移门禁记录在 [docs/theory-baseline.md](docs/theory-baseline.md)。

## OMP 安装与使用

```bash
omp plugin install https://github.com/takltc/pi-llm-as-a-verifier.git
omp plugin config set omp-llm-verifier enabled true
omp plugin config set omp-llm-verifier candidateCount 3
omp plugin doctor
```

`enabled` 控制自动验证。`candidateCount` 表示高影响检查点的候选总数，范围为 2-8、默认值为 3；经过审计的观测工具固定使用一个样本。会提交状态/控制流的 read-tier 工具与未分类扩展工具继续作为高影响检查点。配置在新 OMP 会话中生效。验证器默认继承 OMP `modelRoles.default` 的模型、推理等级、凭据、请求头和兼容配置。

按照论文的 智能体≠验证器 架构（参考实现用 Gemini 或 DeepSeek 给 GPT/Claude 的轨迹打分），可以通过 OMP 模型选择器指定独立的验证器模型：

```bash
omp plugin config set omp-llm-verifier verifierModel deepseek/deepseek-v4-flash
```

`verifierModel` 留空是通用默认行为，此时使用 OMP `modelRoles.default` 做自我验证。可通过 `omp plugin config set omp-llm-verifier verifierModel ''` 清除已有覆盖。插件会在首次进入会话和同一会话切换 default 模型后重新探测实际验证器能力。

高影响检查点使用 TurboAgent 的在线选择配置（`K=1`、`pivots=2`、一个 Task Success 准则）。TurboAgent 参考配置的 N 为 3；本插件通过 `candidateCount` 暴露 2–8 的 N 范围。首个样本会先完成并预热共享 provider 前缀，其余 N-1 个样本随后并发生成。论文的质量/成本轴仍可配置：

```bash
# 每个准则的独立重复验证次数（论文 §4.2）：在线默认 K=1，论文主实验常用 K=8
omp plugin config set omp-llm-verifier nEvaluations 8
# PPT 聚合器的 pivot 数量 k（论文 §3.2）；TurboAgent 在线默认 2
omp plugin config set omp-llm-verifier pivots 2
```

`nEvaluations` 取值 1-16、`pivots` 取值 1-8，均为整数质量/成本参数，新会话生效。

仅关闭自动验证并保留插件安装状态：

```bash
omp plugin config set omp-llm-verifier enabled false
```

停用或恢复插件加载：

```bash
omp plugin disable omp-llm-verifier
omp plugin enable omp-llm-verifier
```

验证器模型必须能提供 OpenAI Chat Completions 或 Responses 的 token logprobs——论文的细粒度奖励从评分 token 分布读取。解析出的验证器模型缺少该能力时，原始智能体模型继续可用，本次模型不启用验证，同时 OMP 明确显示两条处理路径：（1）把 `modelRoles.default` 切换到支持 token logprobs 的模型；（2）单独配置 `verifierModel`。首次进入会话和同会话切换模型都会执行这项检查。探测超时或传输失败时，插件会保持原始智能体模型，并在 60 秒后重新探测能力。插件会根据 provider 与模型标识识别 DeepSeek 模型族，因此 `inferx/deepseek-*` 等托管变体使用参考实现的 32,768 token 输出预算，通用 OpenAI-compatible 验证器继续使用参考实现的 4,096 token 预算。任务包含视觉证据时，验证器模型还需支持图片输入。每次成对评分请求先携带共享的用户、assistant 和工具结果图片，再携带带标签的 Trajectory A 与 Trajectory B 图片，从而保持参考实现的多模态证据路径。

OMP 的 `--max-time` 是整个 agent loop 的绝对截止时间，候选生成与 PPT 共享这份总预算。普通交互会话省略该参数时，agent deadline 保持未设置状态。高思考强度的 headless/CI 运行应给完整的多轮 action 留足预算，例如 `--max-time 15m`；插件的单次验证 HTTP 上限为 10 分钟，同时服从 OMP 的会话取消信号。

已验证的比较结果会缓存在项目根目录的 `.omp-llm-verifier-cache.json`，缓存键覆盖任务、有序共享图片、候选专属图片、两个候选轨迹、评分标准、模型和 prompt 版本的内容指纹，因此相同内容的重复验证不消耗验证器 token。该文件可以直接删除，建议加入 gitignore。

每个过程步骤都有可观测遥测：OMP 控制台会针对每次 coding-agent 模型请求输出一行
`event:decision` JSON，包含 `granularity=prm` 与 `path`（`single`
表示经过审计的观测使用一个样本，`majority` 表示 TurboAgent 的精确动作多数决，`verifier`
表示论文 PPT，`fallback` 表示可恢复降级，`aborted` 与 `error` 表示终止状态）、
获胜候选的索引与平均得分、有向比较次数、参与打分的验证器模型与 prompt
契约版本，以及本次请求的验证器 token 用量。`scoreSources` 分别统计 logprob
期望、文本字母回退、运行期中性平局和旧版/未知缓存项；全部评分标签均来自 token
logprobs 时，`paperEquivalent` 为 `true`。`scoreDistribution` 记录这些标签的有效
A–T 支持数与返回概率质量的最小值和平均值。
决策还包含 `sampledCandidates`、`checkpointReason`、`toolUseCandidates`、
`terminalCandidates`、`discardedCandidates`（严格
多数 `count > N/2` 已不可逆时被取消的在途候选请求）与胜出 stop reason。扩展程序
也可以通过包装 provider 状态上的 `onDecision` 回调读取同样的数据。

首个动作采样、候选扩展与 PPT 在回放前保持内容隔离。单样本观测沿用 OMP 的常规 working
UI；高影响检查点显示 `Expanding a consequential action to N candidates…` 与
`Verifying N candidate actions with PPT…`，并在 winner 回放前恢复默认状态，避免跨过程步骤
沿用上一条工具 intent。

OMP 自动标题等低 token 辅助请求采用单轨原模型调用。PRM 调度范围覆盖 coding-agent
动作。支持 reasoning 的辅助模型在首次请求中使用调用方已选择的思考
强度；辅助调用未选择强度时使用模型支持的最低强度。模型元数据滞后并导致动态 endpoint
以 400 拒绝 `disableReasoning` 时，包装器会记录这项 endpoint 能力并重试一次。用户为
编码轨迹选择的 `high` 或 `max` 会原样保留。

所有样本共享调用方的完整上下文、工具定义、session ID、prompt-cache key 与 provider
session state。首个样本完成后会预热相同的编码上下文前缀，其余 N-1 个高影响动作候选
随后并发生成，从而提高缓存亲和性。插件会在采样前拒绝
可能在生成期间直接执行副作用的 provider-native `execHandlers`；声明式工具调用完整参与
选择，胜出调用才会回放到 agent loop。

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

修改代码后重新执行测试、类型检查和插件诊断，再启动 OMP 验证普通请求链路。
