# LLM-as-a-Verifier 理论与实现基线

> 状态：后续自回归优化的理论门禁
> 核验日期：2026-08-21
> 论文版本：[arXiv:2607.05391v2](https://arxiv.org/abs/2607.05391)，2026-07-07
> 作者参考实现：[commit `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770)
> Coding-agent 参考实现：[TurboAgent commit `eeb61be9cb618ea9c52262cebf15092e7c185146`](https://github.com/llm-as-a-verifier/TurboAgent/tree/eeb61be9cb618ea9c52262cebf15092e7c185146)

本文用于约束本插件的算法、提示词、概率提取、缓存和自动选择优化。理论一致性以论文公式与算法语义为首要依据；作者固定提交提供可执行兼容契约；作者官方文档补充公开配置和使用边界；模型服务商官方规范限定传输能力。来源之间存在歧义时，本文显式记录歧义与当前兼容选择。文中的旧提交行号链接用于固定未变化语义；最新上游行为以页首提交为准。

## 1. 仓库引用盘点

| 位置 | 明确引用或契约 | 核验结果 |
| --- | --- | --- |
| `README.md`、`README.zh-CN.md` | [原始论文](https://arxiv.org/abs/2607.05391)；作者仓库的 [Terminal-Bench 2.1 self-verification](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21) | 两项均为一手来源。 |
| `package.json` | 插件名、默认候选数 3、候选范围 2–8、可选 verifier model | 未增加论文或算法来源。 |
| `src/ppt.ts` | PPT、Hamiltonian ring、Bradley–Terry、比较次数公式 | pair 集采用论文 Appendix B.2 Algorithm 1 的集合定义；正文复杂度公式作为上界；注释定位为论文 §3.2 与 Appendix B.2。 |
| `src/scale.ts`、`src/prompt.ts`、`src/client.ts` | 20 级 A–T、双轨迹 pairwise prompt、token logprobs | 与作者实现的兼容方向一致。 |
| `src/auto.ts` | PRM 高影响检查点、按 effect 分配动作采样预算、N（2–8，默认 3）、k=2、K=1、C=1、精确多数决与 PPT | 经审计的观测采用一个样本；终态、状态/控制提交及 OMP `write`/`exec`/未知动作扩展至 N 个样本。检查点选择属于论文 PRM 的产品映射，选择器默认值与 TurboAgent 一致。 |
| `_ref/` | 作者仓库本地参考副本 | 最新核验固定在提交 `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`；上一轮兼容快照为 `115de305f23ed89bc42e86e010853c40059f3f7d`。 |
| `docs/theory-baseline.md`、`docs/granularity-selection.md` | 理论来源、三种奖励范围、产品映射与防漂移不变量 | 固定论文、作者主仓库、TurboAgent 与 OMP 四类一手来源。 |

## 2. 理论目标与适用边界

论文把“判断候选解是否正确的能力”定义为新的可扩展轴，并面向长时程 agent trajectory 产生连续、细粒度、无需额外训练的验证信号。论文展示的用途包括 Best-of-N、进度估计和强化学习奖励。[论文摘要与 §1，pp. 1–3](https://arxiv.org/pdf/2607.05391#page=1)

适用边界：

- 连续奖励依赖评分 token 的概率或 logprob。只暴露采样文本的模型需要采用“强模型给出分析、开放 verifier 给出概率分布”的两阶段流程。[Appendix A，p. 26](https://arxiv.org/pdf/2607.05391#page=26)；[Appendix B.6，pp. 30–31](https://arxiv.org/pdf/2607.05391#page=30)；[作者官方说明](https://llm-as-a-verifier.com/docs/advanced_features/logit_restricted_models.html)
- 论文把 action 定义为自然语言、代码编辑或工具调用，并把 trajectory 定义为状态/动作序列；验证边界需要同时覆盖这些动作类型。[论文 §2，pp. 3–4](https://arxiv.org/pdf/2607.05391#page=3)
- 论文强化学习实验的范围包含单轮设置，作者在限制章节将多轮扩展列为后续方向。[Appendix A，p. 26](https://arxiv.org/pdf/2607.05391#page=26)
- generator 与 verifier 可以使用同一模型，也可以使用独立模型。作者公开的 Terminal-Bench 2.1 self-verification 使用同一 DeepSeek 模型生成并验证轨迹；论文其他实验也使用独立 verifier。[作者 self-verification](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

### 2.1 三种论文支撑的验证边界

| 方案 | 论文/官方落点 | 对 coding agent 的含义 |
| --- | --- | --- |
| 结果奖励模型 ORM | 对候选最终答案或最终结果评分；Appendix B.3 用 ORM 运行 coding/math Best-of-N。[论文 Appendix B.3](https://arxiv.org/html/2607.05391v2) | 适合单轮补丁与最终结果选择；过程中的工具动作已经发生。 |
| 过程奖励模型 PRM | 在状态 (s_t) 采样候选 action，并对步骤或轨迹前缀评分；Appendix B.3 报告 sampled actions per step 的 (k=1,3,5,9)，随 k 增长时 pass@1 单调提升。[论文 Appendix B.3](https://arxiv.org/html/2607.05391v2) | 声明式工具调用允许先比较候选 action、再只执行 winner，适配交互式 OMP。k=1 是低计算基线，k>1 用于需要提高上限的决策点。 |
| 轨迹奖励模型 TRM | 对包含动作、工具调用和观测结果的完整交互轨迹评分；论文 §5 的主实验采用该范围。[论文 §2、§5](https://arxiv.org/html/2607.05391v2) | 适合离线 benchmark 或具备 N 个隔离工作区、终端与副作用环境的 harness。首个动作分叉后，跨轨迹生成缓存复用下降。 |

TurboAgent 的 per-request fan-out 是 coding-agent 部署 seam，评分范围继续落在 ORM/PRM/TRM 三种论文术语中。[论文 §6 Coding Agent Extension](https://arxiv.org/html/2607.05391v2)；[TurboAgent backend](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/proxy/backend.py)

本插件选择 **逐步 PRM**，并把论文的 sampled-actions-per-step 预算映射为高影响检查点调度。论文 Appendix B.3 给出 k=1 与 k>1 的逐步采样实验；§4 说明验证计算量可按下游延迟预算调节。按操作 effect 路由属于 PRM 内部的工程调度策略，收益通过本插件实验单独验证。[论文 §4 L170–174、Appendix B.3 L492–500](https://arxiv.org/html/2607.05391v2)

1. 每个 coding-agent 步骤先生成一个 proposal。proposal 只包含经审计的 OMP 观测工具时，本步骤使用 PRM 的 k=1 基线并直接执行；该路径不触发 candidate warning、插件 working 文案或 verifier 请求。
2. 终态回复与任一状态/控制提交、`write`、`exec`、未知工具、未分类 read-tier 工具、缺失工具或 approval 解析异常构成高影响检查点。插件随后补充生成 N-1 个候选，使总样本数达到 N=`candidateCount`（2–8，默认 3）。
3. OMP 的参数级 approval tier 与受测试的 effect adapter 共同确定动作影响：动态本地 read/grep、LSP 查询、只读 debug/GitHub/computer 与 job list/logs 等保持观测语义；SSH read 等由 OMP 归入 `exec`；ask/yield/checkpoint/rewind/todo/长期记忆提交及 worker cancel/send 等 read-tier 控制工具进入检查点；省略 approval 的工具按 OMP 契约归入 `exec`。
4. 高影响检查点采用 TurboAgent 的精确动作多数决，剩余分歧进入 G=20 的 PPT。在线默认 k=2、K=1、C=1；工具调用 ID 不进入动作身份，工具名与参数进入身份。
5. proposal 完成后才并发派发 N-1 个额外样本。该调度保持相同采样分布和候选总数，并先预热完整消息历史、工具 schema、session ID、prompt-cache key 与 provider session state 形成的公共前缀。
6. `granularity=prm` 标识 PRM，`path=single` 标识 k=1 观测步骤；`sampledCandidates` 与 `checkpointReason` 公开真实计算量和调度原因。
7. effect-gated 检查点策略是 OMP 场景的产品映射。论文为 PRM、可调 sampled actions 和 PPT 提供理论与实验依据；该策略的端到端收益由插件 benchmark 与真实 OMP 运行数据验证。
8. 最终结果 ORM 与完整轨迹 TRM 继续服务离线 self-verification/benchmark；TurboAgent 每请求全量 fan-out 作为最高密度在线配置保留理论参照。

## 3. 细粒度期望奖励

### 3.1 公式

设任务为 \(x\)，候选轨迹为 \(\tau\)，简单评价准则数为 \(C\)，每个准则的独立重复次数为 \(K\)，有序评分 token 集合为 \(V_{score}=\{v_1,\ldots,v_G\}\)，数值映射为 \(\phi(v_g)\)。论文 Eq. (3.1) 的实现形式为：

\[
R(x,\tau)=\frac{1}{CK}
\sum_{c=1}^{C}\sum_{r=1}^{K}\sum_{g=1}^{G}
p_\theta(v_g\mid x,c,\tau,r)\,\phi(v_g).
\]

作者实现把期望值线性归一化至 \([0,1]\)。默认兼容尺度使用 \(G=20\)，A 映射到 20，依序下降，T 映射到 1；大小写映射到同一数值。[论文 §3.2、Eq. (3.1)，p. 5](https://arxiv.org/pdf/2607.05391#page=5)；[作者实现 `fine_grained_reward.py` L69–89](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L69-L89)

实现含义：

1. 分数来自评分 token 概率分布的期望值。
2. `argmax` token 或模型正文中的单个字母用于 verifier 分数时会改变 Eq. (3.1) 的估计量。TurboAgent 的精确动作多数决是一条独立选择捷径，采用独立 `majority` 遥测路径，`paperEquivalent` 仅描述经过 logprob/PPT 的选择。
3. 作者实现从服务端返回的候选 token 中筛选有效 A–T，执行 `exp(logprob)`，再按有效评分 token 的已返回概率质量归一化并求期望。[作者实现 L621–668](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L621-L668)
4. `top_logprobs=20` 是传输上限配置，服务端可能返回少于请求数量的候选，且返回列表还可能包含 A–T 之外的 token。因此实现需要记录有效评分 token 数量与概率质量，缺失有效支持时进入显式降级路径。[OpenAI Chat Completions 官方规范](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)；[DeepSeek Chat Completions 官方规范](https://api-docs.deepseek.com/api/create-chat-completion/)

### 3.2 C、K、G 的语义

| 轴 | 固定语义 | 可调范围与证据 |
| --- | --- | --- |
| \(G\)：评分粒度 | 有序评分 token 的数量；概率期望把模型信念投影为连续分数。 | 论文实验从较粗尺度扩展到 G=20，G=20 提升正负轨迹分离度与 SNR。本插件的论文兼容基线固定为 A–T 共 20 级。[论文 §4.1，pp. 7–8](https://arxiv.org/pdf/2607.05391#page=7)；[作者官方 scaling 文档](https://llm-as-a-verifier.com/docs/advanced_features/verification_scaling.html) |
| \(K\)：重复评估 | 对同一 pair、同一 criterion 进行独立随机评估并取平均；Monte Carlo 方差按 \(O(1/K)\) 缩减，系统偏差仍需单独控制。 | 论文主实验常用 K=8；作者 Bo3 self-verification 使用 K=2。K 是质量/成本参数。[论文 §4.2，p. 9](https://arxiv.org/pdf/2607.05391#page=9)；[作者 Bo3 脚本 L1–35](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/scripts/run_bo3.py#L1-L35) |
| \(C\)：准则分解 | 将复合判断拆成简单、互补准则，分别求期望后等权平均。 | 代码 agent 实验使用 Specification、Output、Errors 三类准则。领域适配可以改写准则内容，同时保持简单性、独立评分和显式聚合权重。[论文 §4.3，pp. 9–10](https://arxiv.org/pdf/2607.05391#page=9)；[作者 Terminal-Bench criteria L3–19](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/criteria/terminal_bench.md#L3-L19) |

## 4. Pairwise prompt 与 A/B 位置控制

每次 verifier 调用需要同时接收任务、Trajectory A、Trajectory B、统一评分尺度和一个 criterion，并在同一回复尾部输出 `<score_A>` 与 `<score_B>`。两个候选分数共享同一分析上下文，可降低独立打分带来的提示条件差异。[论文 §3.2，p. 5](https://arxiv.org/pdf/2607.05391#page=5)；[作者实现 L696–745](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L696-L745)

评分 token 定位需要满足：

- 使用回复中最后一组 score tag，避免分析正文引用格式时误取前置 token。
- 兼容 tokenizer 把 `>` 与字母融合成 `>A` 的情况。
- `<score_A>` 和 `<score_B>` 各自绑定紧邻标签后的 token 分布。
- 文本字母回退与 0.5 平局属于工程降级结果；遥测需要标识该路径，论文等价统计只接纳有效 logprob 期望。

上述行为来自[作者提取实现 L621–668](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L621-L668)。

A/B 偏差由两层机制控制：

1. PPT ring 是随机有向 Hamiltonian cycle，每个候选恰好一次处于 A、一次处于 B，位置偏差在环上平衡。[论文 §3.2，pp. 6–7](https://arxiv.org/pdf/2607.05391#page=6)
2. 作者实现按重复编号交替位置：偶数重复使用 `(a,b)`，奇数重复交换 prompt slot，再把结果映射回候选身份。K≥2 时，同一个有向比较也获得位置平衡。[作者实现 L748–758、L796–910](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L748-L758)

缓存键必须保留方向、criterion、任务、重复编号、模型、prompt/scale 版本及影响输入语义的上下文摘要。`(a,b)` 与 `(b,a)` 是不同实验条件。失败调用产生的 0.5/0.5 只服务当前运行，持久缓存保持成功观测集合。[作者实现 L757–811、L826–910](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L757-L811)

## 5. Probabilistic Pivot Tournament

### 5.1 三阶段

1. **Ring pass**：随机生成包含全部 N 个候选的有向 Hamiltonian cycle，评分 N 条相邻边。对每条边 `(i,j)`，先取得连续奖励 \(R_i,R_j\)，再用 Bradley–Terry 转成 soft win：

   \[
   P(\tau_i \succ \tau_j\mid x)=
   \frac{1}{1+\exp(-(R_i-R_j))}.
   \]

   将 \(p\) 累加给 \(i\)，将 \(1-p\) 累加给 \(j\)，并分别增加比较计数。[论文 Eq. (3.2) 与 §3.2，p. 6](https://arxiv.org/pdf/2607.05391#page=6)

2. **Pivot selection**：按 ring 阶段的平均偏好 \(w_i/c_i\) 排序，选择 top-k 作为 pivots。[论文 §3.2，p. 6](https://arxiv.org/pdf/2607.05391#page=6)

3. **Pivot rounds 与最终选择**：先生成每个 non-pivot–pivot pair 以及所有 pivot–pivot pair，再按 Algorithm 1 扣除已在有向 ring 中出现的边。剩余边的 soft wins 继续累加到同一组 \(w,c\)，最终返回 \(\arg\max_i w_i/c_i\)。[论文 §3.2，pp. 6–7](https://arxiv.org/pdf/2607.05391#page=6)；[Appendix B.2，Algorithm 1，pp. 26–28](https://arxiv.org/pdf/2607.05391#page=26)

### 5.2 复杂度与调用计数

论文正文给出以下 \(O(Nk)\) 上界：

\[
L_{max}=N+k(N-k)+\binom{k}{2}=O(Nk),\quad k\ll N.
\]

Appendix B.2 Algorithm 1 定义
\(E_{piv}=(E_{nonpivot,pivot}\cup E_{pivot,pivot})\setminus E_{ring}\)，因此精确逻辑比较数为：

\[
L=N+k(N-k)+\binom{k}{2}-|E_{ring}\cap E^*_{piv}|.
\]

每个逻辑比较需要 \(C\times K\) 个 criterion/repetition 评分观测；一次 pairwise verifier 回复同时给出 A、B 两个奖励。[论文 §3.2，p. 7](https://arxiv.org/pdf/2607.05391#page=7)；[作者 PPT 官方文档 “Cost model”](https://llm-as-a-verifier.com/docs/advanced_features/pivot_tournament.html)

需要分别统计三种成本：

- `logical comparisons`：PPT 聚合器累计的唯一有向边数 L。
- `unique directed score keys`：方向、criterion、repeat 等身份去重后的观测键数。
- `provider calls`：缓存命中、并发重试和失败后实际发出的远程请求数。

### 5.3 N=3、k=1 的来源歧义

| 来源 | 推导结果 | 说明 |
| --- | --- | --- |
| [论文 §3.2，p. 7](https://arxiv.org/pdf/2607.05391#page=7) | 5 | 正文公式给出 `3 + 1×2 + 0 = 5`。 |
| [论文 Appendix B.2，Algorithm 1，p. 27](https://arxiv.org/pdf/2607.05391#page=27) | 4 | 伪代码把已在 `E_ring` 中的 pivot edge 从第二阶段集合移除；三节点环会重叠一条同向边。 |
| [作者 `pivot_tournament.py` L64–92](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/pivot_tournament.py#L64-L92) | 5 | 可执行实现生成全部 non-pivot–pivot pairs，并将重叠逻辑边再次聚合。评分缓存可复用已有远程结果。 |
| [作者 `run_bo3.py` L1–7](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/scripts/run_bo3.py#L1-L7) | 6 | 脚本文档写成 3 ring + 3 pivot-round；该描述与同仓库 PPT 代码不一致。 |

当前插件冻结 **4 次逻辑比较** 作为论文形式化算法基线。作者可执行实现对应的 5 次累计作为独立兼容 fixture：它会复用缓存中的 ring 观测并再次加权一条边。遥测同时报告 logical comparisons、unique keys 与 provider calls，使两种语义可以直接比较。

### 5.4 高影响检查点的 N=3、k=2 在线选择配置

TurboAgent 的参考在线配置使用 N=3、k=2、K=1、C=1，并开启精确动作多数决。本插件在高影响 PRM 检查点保持 k/K/C 默认值，并将 N 暴露为 `candidateCount`（2–8，默认 3）；经审计的观测步骤采用一个 proposal。[TurboAgent 配置](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo-agent.yaml)；[多数决与 PPT 选择](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/verifier/verifier.py)

对 N=3、k=2，正文上界为 `3 + 2×1 + 1 = 6`。Algorithm 1 的有向集合差会移除 1–2 条 ring 重叠，因此本插件每次冷启动且无多数时执行 4–5 个逻辑比较。每个比较在在线默认下对应一个 verifier 请求；评分缓存可进一步减少 provider calls。精确动作多数决对应 0 个 verifier 请求。

## 6. 在线 coding-agent 与离线 self-verification 配置

作者公开的 Terminal-Bench 2.1 self-verification 证明同一模型可承担候选生成和验证。其 Bo3 脚本固定 N=3、k=1、K=2，并使用独立分数缓存。[作者 self-verification](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)；[作者 Bo3 脚本 L1–35](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/scripts/run_bo3.py#L1-L35)

实现边界：

- 在线 coding-agent 路径采用 effect-gated PRM 动作预算：经审计的观测为 1，高影响检查点为可配置 N（2–8，默认 3），检查点选择器使用 TurboAgent 的 PPT k=2、K=1、C=1；离线 self-verification 使用 Bo3 的 N=3、PPT k=1、K=2 与三项 coding-agent criteria。两组配置拥有独立来源和延迟预算。
- 独立 verifier model 是支持项；同模型 self-verification 也是支持项。遥测需记录实际 generator/verifier 身份。
- 高影响检查点的候选采样相互独立，并共享相同任务、前置证据、工具 schema 和缓存身份。候选请求关闭 server-side turn chaining，保留 session/prompt cache affinity，并使用 TurboAgent 的 temperature=1 默认采样温度。
- 每个步骤先完成 proposal。OMP 参数级 approval tier 与 effect adapter 只将已知无状态的 `read` 调用归为观测步骤；终态、read-tier 状态/控制提交、未分类扩展工具、`write`、`exec`、未知或解析异常调用归为高影响检查点。混合工具批次按最高影响 effect 处理。
- 同一在线 PPT 的候选处于同一个 PRM 检查点；工具调用、代码相关动作文本和终态回复统一进入选择。胜出响应保留原始 tool-call ID 与 provider payload，agent loop 只看到这一条响应。
- 精确动作多数决使用序列化后的可见文本、工具名和工具参数；provider 生成的调用 ID 不影响一致性判断。多数决采用独立 `path=majority`，PPT 的 `paperEquivalent` 指标保持独立。
- no-logprobs 电路断路：同一选择内所有评分 job 共享同一请求形态，前 2 个独立 job 均以 logprobs 不支持失败（各自已耗尽客户端内部重试）后，未启动的 job 直接进入运行期 0.5 平局、不再发起 provider 调用；平局不持久化，`paperEquivalent` 保持 false。属于成本护栏，不改动 Eq. (3.1) 期望、PPT pair 集或失败语义。
- 缓存写入节流：一个阶段内部最多约 5 次中间 checkpoint 落盘加末尾一次，避免把每次评分完成都变成同步锁+fsync 重写；崩溃最多丢失最近一小批分数。
- 插件把 K（`nEvaluations`，1–16）与 k（`pivots`，1–8）暴露为在线配置项，默认 K=1、k=2。K 是论文 §4.2 的质量/成本轴，k 是论文 §3.2 的 PPT 参数；上调后按论文语义增加验证计算。`k` 在运行时按候选数 `min(k,N)` 收敛。
- 候选生成 transient 重试（工程护栏）：每个候选在 provider 瞬时故障（可重试状态或等价错误）时重试一次，等待时间按候选索引错开并顺从外部分流 `signal` 的 abort；重试是同一候选索引的新独立采样，采样分布与关闭 server-side turn-chaining 的 side-channel 语义不变，`successfulCandidates` 仍按成功候选数上报。护栏不掉 `temperature=1`、不改变 Eq. (3.1) 期望与 PPT pair 集，只扩充失败容错路径。
- 能力探测时序（生命周期护栏）：启动期 logprobs 能力探测使用 `CAPABILITY_PROBE_TIMEOUT_MS=10_000` 超时，为 OMP 30 秒扩展 handler 期限留下充裕余量；探测只作绑定判断，不影响评分请求形态或 Eq. (3.1) 期望。
- proposal 前缀预热（工程延迟/成本护栏）：proposal 完成后再并发启动 N-1 个高影响候选。所有调用保留相同 prompt-cache identity，因此 proposal 的已完成请求可预热公共 coding-context 前缀；候选总数和后续多数决/PPT 语义保持一致。
- 候选多数即决（工程延迟/成本护栏）：额外候选 fan-out 采用竞速收集，并把已完成 proposal 纳入计数；某动作的严格多数不可逆（`count > N/2`）时，立即结束等待并通过私有信号取消剩余在途候选，随后按多数决回放胜者。剩余候选的生成结果被丢弃、不进入 `successfulCandidates`，决策以 `discardedCandidates` 遥测报告取消数量。多数行动作唯一，剩余候选无法改变该判定，因此胜者与等待全部 N 个候选的结果一致；外部 `signal` 的 abort 语义与 PPT 路径保持不变。

## 7. 图片输入

图片是任务上下文的一部分。参与比较的两个候选必须看到同一组图片，verifier prompt 需要携带相同图片、顺序和语义说明；图片能力缺失应在评分前显式失败或进入可观测降级路径。作者实现把图片附加到 pairwise verifier 请求，并在 prompt 中记录图片数量。[作者实现 `fine_grained_reward.py` L403–447、L696–745](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L403-L447)

图片输入属于任务证据完整性约束。插件把当前用户任务及其后续共享消息中的 user、assistant、tool-result 图片按时间顺序传给 verifier，再附加带有 Trajectory A/B 归属的候选专属图片；图像压缩、丢帧、顺序变化或候选与 verifier 的图片集合差异都会改变验证条件。缓存身份包含共享图片和两个候选图片的内容摘要、顺序、槽位归属及预处理版本。

## 8. 可实现的理论不变量

截至 2026-08-21，`INV-01` 至 `INV-15` 的最小自动门禁已全部落地：`test/core.test.ts` 覆盖期望值/映射/扫描差分/缓存方向与环比与版本/准则顺序无关/K 槽位互换与候选身份回映/PPT pair 集与复杂度，`test/theory.test.ts` 固化离线 Bo3 与在线选择器配置，`test/verification.test.ts` 覆盖 PRM 检查点分类、观测单样本、read-tier 状态提交、未分类工具、proposal 前缀预热、额外候选并发、工具调用选择、精确多数决、缓存身份、副作用保护、证据图片与降级来源。

| ID | 必须保持的实现性质 | 最小自动门禁 | 一手依据 |
| --- | --- | --- | --- |
| INV-01 | 连续奖励等于有序评分 token 概率的数值期望，并线性归一化到 `[0,1]`。 | 构造概率分布，断言期望值、端点和单调性；覆盖非归一输入。 | [论文 §3.2、Eq. (3.1)，p. 5](https://arxiv.org/pdf/2607.05391#page=5)；[作者实现 L621–668](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L621-L668) |
| INV-02 | 论文兼容尺度为 G=20、A→20…T→1，标签字母应具备稳定单 token 表示或明确兼容处理。 | 固定 20 个 token、顺序、大小写映射和尺度版本快照。 | [论文 §4.1，pp. 7–8](https://arxiv.org/pdf/2607.05391#page=7)；[作者实现 L69–89](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L69-L89) |
| INV-03 | 一个 pairwise prompt 同时包含任务、A/B 轨迹、统一尺度和单一 criterion；同一回复产生两个分数。 | prompt snapshot；断言 criterion 位于可缓存公共前缀之后，两个 tag 均出现且尾部无附加内容。 | [论文 §3.2，p. 5](https://arxiv.org/pdf/2607.05391#page=5)；[作者实现 L696–745](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L696-L745) |
| INV-04 | 概率提取绑定最后一组 tag，并兼容 `>A` 融合 token；标签后的空白 token 不重复覆盖首个位置分布；有效评分 token 质量完成归一化。 | 覆盖引用旧 tag、多组 tag、融合 token、空白 token、大小写、空支持与返回数量不足。 | [作者最新实现 L632–679](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py#L632-L679) |
| INV-05 | C 个简单 criterion 独立评分并显式等权平均；任何权重变化均形成新算法版本。 | 单 criterion 隔离测试、顺序无关性、聚合权重和 prompt version 快照。 | [论文 §4.3，pp. 9–10](https://arxiv.org/pdf/2607.05391#page=9)；[作者官方 scaling 文档](https://llm-as-a-verifier.com/docs/advanced_features/verification_scaling.html) |
| INV-06 | K 表示独立重复观测；重复编号进入缓存身份，K≥2 时交替 A/B slot 并映射回候选。 | 断言 rep 0/1 的 slot 互换、候选身份稳定、缓存键不同。 | [论文 §4.2，p. 9](https://arxiv.org/pdf/2607.05391#page=9)；[作者实现 L748–811](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L748-L811) |
| INV-07 | Ring 为随机有向 Hamiltonian cycle，每个候选恰好一次处于 A、一次处于 B。 | 对多个 N/seed 验证入度=出度=1、覆盖全部节点、可复现。 | [论文 §3.2，pp. 6–7](https://arxiv.org/pdf/2607.05391#page=6)；[作者 PPT 实现 L28–38](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/pivot_tournament.py#L28-L38) |
| INV-08 | 每条边使用 Bradley–Terry `sigmoid(Ra-Rb)` 生成互补 soft wins。 | 断言相等分数为 0.5、交换输入概率互补、分差单调。 | [论文 Eq. (3.2)，p. 6](https://arxiv.org/pdf/2607.05391#page=6) |
| INV-09 | Pivots 由 ring 阶段 `w/c` 的 top-k 产生；第二阶段覆盖新的 non-pivot–pivot 与 pivot–pivot 有向边；最终按全阶段 `w/c` 选择。 | 对固定 ring/score 验证 Algorithm 1 pair 集、tie-break 与 k 边界，并保留作者实现兼容 fixture。 | [论文 §3.2，pp. 6–7](https://arxiv.org/pdf/2607.05391#page=6)；[Appendix B.2，Algorithm 1，pp. 26–28](https://arxiv.org/pdf/2607.05391#page=26)；[作者实现 L40–92](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/pivot_tournament.py#L40-L92) |
| INV-10 | `N+k(N-k)+C(k,2)` 是比较数上界；精确计数扣除 ring 与原始 pivot edge 集的有向交集。N=3、k=1 固定为 4，N=3、k=2 为 4–5。 | 多组 N/k/ring 验证集合差与上界；离线与在线配置均固定调用计数；另报 unique keys/provider calls。 | [论文 §3.2，p. 7](https://arxiv.org/pdf/2607.05391#page=7)；[Appendix B.2，Algorithm 1，p. 27](https://arxiv.org/pdf/2607.05391#page=27) |
| INV-11 | 真实 verifier 路径必须取得 token logprobs；文本字母回退、空支持和解析失败进入显式来源统计。论文等价决策要求全部 score tag 来自 logprob 期望，并记录有效 A–T 支持数与返回概率质量。 | Chat/Responses/DeepSeek/Vertex 响应契约 fixture；断言 `scoreSources`、`scoreDistribution` 与 `paperEquivalent`，使文本回退和中性平局排除在论文等价统计之外。 | [Appendix B.6，pp. 30–31](https://arxiv.org/pdf/2607.05391#page=30)；[OpenAI Chat](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)；[DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion/)；[Vertex GenerationConfig](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GenerationConfig) |
| INV-12 | pair、criterion、rep、模型、prompt/scale、任务证据、图片摘要和方向共同决定缓存身份；缓存项携带评分来源，失败平局保持运行期作用域。 | 缓存碰撞测试、方向测试、版本失效测试、来源持久化测试、失败后重试测试。 | [作者实现 L748–811、L862–910](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L748-L811) |
| INV-13 | 候选携带可判断任务完成度的证据；图片输入在候选与 verifier 条件中保持一致。 | 证据截断标记、图片顺序/摘要测试、能力前置校验。 | [论文 §3.1–§3.2，pp. 4–5](https://arxiv.org/pdf/2607.05391#page=4)；[作者实现 L403–447、L696–745](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L403-L447) |
| INV-14 | 每个 coding-agent 步骤先采样一个 proposal；只有全部工具同时满足 OMP `read` tier 与受审计 observation effect 时使用 PRM 单样本路径。终态、read-tier 状态/控制提交、未分类工具、`write`/`exec`/未知或解析异常动作扩展至配置 N。 | 静态与参数级动态 approval fixture；本地 read、SSH read、ask、未分类 extension tool、approval error、未知工具、终态、混合批次；断言观测路径 1 次 generator、0 次 verifier、无 degraded warning与插件 working 文案，并记录 `path=single`。 | [论文 §2](https://arxiv.org/html/2607.05391v2#S2)；[论文 §4](https://arxiv.org/html/2607.05391v2#S4)；[Appendix B.3 PRM](https://arxiv.org/html/2607.05391v2)；[OMP effect 映射研究](granularity-selection.md#42-哪些操作进入验证边界) |
| INV-15 | 高影响检查点的首个 proposal 完成后并发启动 N-1 个额外候选；所有样本共享完整上下文与 prompt-cache 身份，随后统一进入严格多数决/PPT，agent loop 只接收 winner。side-channel 候选关闭 turn chaining/cache-refresh ownership；生成期原生工具执行在采样前终止。 | proposal/额外候选双屏障、总样本数 2–8、session/cache identity、temperature、cache refresh ownership、全工具候选、混合 stop reason、胜出 tool-call 回放、`execHandlers` 零调用；严格多数提前结束仍选同一 winner。 | [论文 Appendix B.3](https://arxiv.org/html/2607.05391v2)；[论文 §6 Coding Agent Extension](https://arxiv.org/html/2607.05391v2)；[TurboAgent `_gather_completions`](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/proxy/backend.py#L146-L176) |

## 9. 常见偏离风险

| 风险 | 偏离机制与控制要求 | 一手依据 |
| --- | --- | --- |
| 使用采样字母、argmax 或离散分数替代期望 | 估计量从分布期望变为单点观测，连续性、校准和 tie 行为随之改变。保留 Eq. (3.1) 路径并标记文本回退。 | [论文摘要、§3.2、§4.1，pp. 1、5、7–8](https://arxiv.org/pdf/2607.05391#page=5) |
| 把 `top_logprobs=20` 当作完整 A–T 分布保证 | 服务端可返回更少候选，非评分 token 也占用名额。记录有效 token 数、质量和降级原因。 | [OpenAI Chat 官方规范](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)；[DeepSeek 官方规范](https://api-docs.deepseek.com/api/create-chat-completion/) |
| 从第一个 tag 或错误 token 位置读取概率 | verifier 分析可能引用格式，tokenizer 也可能融合 `>` 与字母。采用最后 tag 与融合 token 测试。 | [作者实现 L621–668](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L621-L668) |
| 将多个 criterion 合并为一个复合问题或引入隐式权重 | 复杂性降低收益和 C 轴语义发生变化。每个 criterion 单独调用，聚合权重进入版本身份。 | [论文 §4.3，pp. 9–10](https://arxiv.org/pdf/2607.05391#page=9) |
| K 次重复被缓存折叠成同一观测 | Monte Carlo 方差缩减前提失效。重复编号与随机调用保持独立。 | [论文 §4.2，p. 9](https://arxiv.org/pdf/2607.05391#page=9) |
| 固定 A/B 位置或把有向 pair 规范化为无向键 | slot bias 会进入 pivot 选择和最终排名。保留有向 ring 与重复内 A/B 交替。 | [论文 §3.2，pp. 6–7](https://arxiv.org/pdf/2607.05391#page=6)；[作者实现 L748–811](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L748-L811) |
| 用硬胜负、原始 reward 排名或未归一累计胜场替换 `sigmoid` 与 `w/c` | PPT 的概率聚合和不同参赛次数归一化被改变。固定 Bradley–Terry、soft wins 与 `w/c`。 | [论文 Eq. (3.2)、§3.2，pp. 6–7](https://arxiv.org/pdf/2607.05391#page=6) |
| 缺少 ring 边、pivot pair、重复累计 ring overlap 或把阶段结果分开排名 | 三阶段 pair 集与统计权重发生变化。对 Algorithm 1 集合差和累计器做形式化 fixture，并单列作者实现兼容结果。 | [Appendix B.2，Algorithm 1，pp. 26–28](https://arxiv.org/pdf/2607.05391#page=26)；[作者实现 L28–92](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/pivot_tournament.py#L28-L92) |
| 混淆逻辑比较、唯一评分键和服务商调用 | 成本估算、缓存收益和理论复杂度会产生互相矛盾的数字。三项分别计量。 | [作者 PPT 官方文档 “Cost model”](https://llm-as-a-verifier.com/docs/advanced_features/pivot_tournament.html)；[作者实现 L64–92](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/pivot_tournament.py#L64-L92) |
| 混用在线与离线默认值 | Bo3 self-verification 使用 N=3、k=1、K=2；本插件在线路径使用 N=2–8（默认 3）、k=2、K=1、C=1。配置、criteria 与遥测分别标记来源。 | [作者 Bo3 脚本](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/scripts/run_bo3.py)；[TurboAgent 配置](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo-agent.yaml) |
| 丢失完整轨迹、终端输出或图片证据 | verifier 的条件信息发生变化，候选正确性可能无法识别。截断策略应保留最终验证证据并输出截断遥测。 | [论文 §3.1–§3.2，pp. 4–5](https://arxiv.org/pdf/2607.05391#page=4)；[作者 loader L155–184](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/loaders.py#L155-L184) |
| 持久化失败调用生成的 0.5 平局 | 暂时性接口故障会固化为未来评分证据。失败 tie 仅用于当前运行，并允许后续重试。 | [作者实现 L796–910](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/115de305f23ed89bc42e86e010853c40059f3f7d/llm_verifier/fine_grained_reward.py#L796-L910) |
| 对近似文本或语义相似动作使用多数决 | TurboAgent 只接受序列化动作的严格字符串多数；扩大等价关系会改变捷径的错误边界。动作身份固定为可见文本、工具名与参数，调用 ID 作为传输元数据。 | [TurboAgent `_try_majority_voting`](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/verifier/verifier.py#L163-L185)；[TurboAgent `format_action`](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/proxy/backend.py#L314-L325) |
| 对观测与高影响动作统一使用 N | 文件读取、搜索等可恢复观测承担完整 BoN+PPT 延迟，在线体验和调用成本随工具步数线性放大。PRM 调度将经审计的 observation tools 固定为一个样本，并把 N 集中到终态、状态/控制提交、`write`/`exec` 与未知检查点。 | [论文 §4 的延迟预算调节](https://arxiv.org/html/2607.05391v2#S4)；[Appendix B.3 的 sampled actions per step](https://arxiv.org/html/2607.05391v2) |
| 只在 session 或终态回答结束时触发在线验证 | 中间 `write`/`exec` 动作直接越过选择边界，coding-agent 修改质量失去过程级改进信号。每个高影响 PRM 检查点统一扩展与选择。 | [论文 Appendix B.3 PRM](https://arxiv.org/html/2607.05391v2)；[论文 §6 Coding Agent Extension](https://arxiv.org/html/2607.05391v2) |
| 把插件检查点选择结果直接外推为论文 benchmark 提升 | 候选生成、检查点策略、轨迹长度、证据截断、模型和任务分布均影响外部效度。发布指标需使用本插件固定协议复测。 | [论文 §5 与 Appendix A，pp. 10–22、26](https://arxiv.org/pdf/2607.05391#page=10) |

## 10. 自回归优化门禁

任何涉及 scale、score tag、logprob 解析、prompt、criteria、K、A/B 位置、PPT pair 集、聚合、缓存身份、图片或 fallback 的改动，都按以下顺序推进：

1. 在变更说明中列出受影响的不变量 ID、理论来源和产品化假设。
2. 先增加属性测试与作者实现 parity fixture，再修改实现。
3. 同时评估选择质量、降级率、token/调用成本和延迟；PPT 成本分开报告三种计数。
4. 遥测至少记录配置 N、实际 `sampledCandidates`、`checkpointReason`、PPT k、C、K、G、seed、criteria、prompt/scale version、generator/verifier model、有效评分 token 支持与概率质量、图片摘要、截断、缓存命中、比较数、实际调用数、decision path 和错误。
5. 精确多数决、最早成功候选 fallback 与 capability failure 使用独立 decision path。文本字母回退和 0.5 tie 进入 `scoreSources`，并令 `paperEquivalent=false`。论文等价指标只聚合真实 logprob 期望经过 PPT 产生的决策。
6. 来源歧义继续保留；形式化算法、正文上界与作者实现分别形成可追溯 fixture，pair 调度变更同时报告三类计数和选择结果差异。

## 11. 一手来源覆盖

| 类别 | 已核验来源 | 覆盖内容 |
| --- | --- | --- |
| 论文原文 | [arXiv 摘要页 v2](https://arxiv.org/abs/2607.05391)、[PDF v2](https://arxiv.org/pdf/2607.05391) | 目标、Eq. (3.1)/(3.2)、C/K/G、PPT、复杂度、实验配置、限制、Algorithm 1、logit-restricted 两阶段流程。 |
| 作者固定仓库 | [最新 commit `8db8a114...`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770)、[兼容快照 `115de305...`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/115de305f23ed89bc42e86e010853c40059f3f7d) | A–T 映射、概率提取、空白 token 边界修复、pairwise prompt、A/B 交替、缓存/失败策略、PPT 可执行行为、Bo3 self-verification、criteria、轨迹截断。 |
| 作者 TurboAgent | [固定 commit `eeb61be9...`](https://github.com/llm-as-a-verifier/TurboAgent/tree/eeb61be9cb618ea9c52262cebf15092e7c185146) | coding-agent 每请求并发候选、动作格式、精确多数决、在线 N/k/K/C 配置、PPT 调用与进度监控边界。 |
| 作者官方文档 | [文档首页](https://llm-as-a-verifier.com/docs/)、[Best-of-N](https://llm-as-a-verifier.com/docs/basic_usage/best_of_n_selection.html)、[fine-grained reward](https://llm-as-a-verifier.com/docs/advanced_features/fine_grained_reward.html)、[PPT](https://llm-as-a-verifier.com/docs/advanced_features/pivot_tournament.html)、[scaling](https://llm-as-a-verifier.com/docs/advanced_features/verification_scaling.html)、[logit-restricted models](https://llm-as-a-verifier.com/docs/advanced_features/logit_restricted_models.html) | 公共 API 语义、成本模型、参数默认值与调节方向、三条扩展轴、能力边界。 |
| 服务商官方规范 | [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[OpenAI Responses](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create)、[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[Vertex GenerationConfig](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GenerationConfig)、[Vertex GenerateContentResponse](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse) | `logprobs`/`top_logprobs` 请求、返回结构、数量上限、返回不足情形及 Vertex chosen/top candidates。 |

本文未采用博客、媒体转述、第三方教程或聚合摘要作为理论依据。
