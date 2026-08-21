# LLM-as-a-Verifier 轨迹与验证粒度选择

> 研究快照：2026-08-21。本仓库基线为 `438ccb714c6c9cf1a30fe3900e33c7c037eb7d1f`，研究期间存在其他协作者的未提交改动；论文采用 arXiv v2；作者参考实现采用本仓库 `_ref` 的固定提交 `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`；TurboAgent 采用官方仓库固定提交 `31ea8dadc00211b96544d3edd02ded1ef64e9856`；OMP 采用本地官方源码提交 `de5ffc4201c4941992402daea355adc1aad3a8db`。

## 结论

本插件推荐采用**按影响面触发的稀疏 PRM 检查点**：纯本地观测步骤使用单候选 `N_t=1`；代码/工作区写入、命令执行、外部副作用、控制流提交和终态回复使用用户配置的 `N` 个候选（当前产品约束为 `2..8`）并在执行前完成精确多数决或 PPT。验证对象必须是共同的已选轨迹前缀 `h_t` 加候选动作 `a_t^i`，即 `h_t ⊕ a_t^i`。读文件、搜索和检查输出留在 `h_t` 中，后续检查点据此判断写入或结论是否得到真实证据支持。

这个方案的奖励范围属于论文的 PRM。操作类型路由属于工程侧计算分配策略；论文 Appendix B.3 直接验证了逐步 PRM 以及每步候选数 `1/3/5/9`，作者代码直接支持选择轨迹 checkpoint，论文尚未给出“按 OMP 工具影响面动态选择 `N_t`”的消融结果。实现和文档应使用“PRM checkpoint scheduling”描述它，避免创造新的奖励模型粒度。

## 1. 术语校正：论文中的两组“三/粒度”

论文包含两套相互正交的概念。

### 1.1 三种奖励范围：ORM、PRM、TRM

论文引言把“完整交互轨迹”与 PRM 的中间步骤、ORM 的最终结果并列，并把主方法称为 trajectory reward model；§5 以 TRM 身份运行主实验；Appendix B.3 另外评估 PRM 与 ORM。[论文 §1、§5、Appendix B.3](https://arxiv.org/html/2607.05391v2)；关键表述位于 HTML 第 94、211–213、492–500 行。

| 奖励范围 | 论文中的最小可证定义 | 调用时机 | 选择对象 |
|---|---|---|---|
| ORM，Outcome Reward Model | 最终答案/最终结果级奖励；Appendix B.3 用于 coding 与 math 的 Best-of-`N` | 候选最终结果全部生成后 | `N` 个最终结果 |
| PRM，Process Reward Model | 中间步骤级奖励；Appendix B.3 明确写作 per-step verification，并改变每步 sampled actions | 每个被选中的 agent step 执行前，或对该步结束后的 prefix 评分 | 同一历史状态下的候选动作，或轨迹 prefix |
| TRM，Trajectory Reward Model | 包含动作、工具调用和观测结果的完整交互轨迹；论文 §5 的主测试时扩展协议 | `N` 条候选轨迹全部结束后 | `N` 条完整轨迹 |

论文 §2 把 LLM agent 的状态定义为先前交互历史，把自然语言回复、代码编辑和工具调用都定义为 action，并定义 `τ=(s₁,a₁,…,s_H,a_H)`。[论文 §2](https://arxiv.org/html/2607.05391v2)；HTML 第 116–119 行。这个定义允许 OMP 的一次模型动作映射为 PRM step，也要求轨迹级验证保留工具观测形成的状态变化。

### 1.2 三条验证扩展轴：G、K、C

论文明确称为 “three independent axes” 的对象是：

1. `G`：评分 token 数量，即 score-token granularity；
2. `K`：独立重复验证次数；
3. `C`：评价标准数量。

三者进入论文公式 (3.1)：

```text
R(x, τ) = 1/(C K) · Σ_c Σ_k Σ_g pθ(v_g | x,c,τ) φ(v_g)
```

`G` 改善分数分离度，`K` 平均单次判断方差，`C` 分解复杂判断；它们决定单个 pair 的评分质量和成本，也不决定“何时触发验证”。来源：[论文 §3.2 公式 (3.1)、§4](https://arxiv.org/html/2607.05391v2)；HTML 第 148–151、170–174 行；作者 README `_ref/README.md:339-360`。

### 1.3 TurboAgent 的 per-request seam

论文 §6 的 Coding Agent Extension 为每个 API request 并行发送 `N` 个候选，再用 PPT 选出一个回复；它是 coding-agent 的部署 seam。作者把它建立在同一 TRM/PRM/ORM 评分与 PPT 之上，“per request” 保持为集成调用边界。[论文 §6](https://arxiv.org/html/2607.05391v2)；HTML 第 282–284 行；`_ref/README.md:281-304`。

TurboAgent 官方代码对每个模型配置读取 `num_candidates`，据此建立任意数量的候选 entry，再以 `asyncio.gather` 并发请求；示例中的 `3` 是配置值。[`backend.py:107-119`](https://github.com/llm-as-a-verifier/TurboAgent/blob/31ea8dadc00211b96544d3edd02ded1ef64e9856/turbo_agent/proxy/backend.py#L107-L119)、[`backend.py:165-198`](https://github.com/llm-as-a-verifier/TurboAgent/blob/31ea8dadc00211b96544d3edd02ded1ef64e9856/turbo_agent/proxy/backend.py#L165-L198)、[`turbo-agent.yaml:3-10`](https://github.com/llm-as-a-verifier/TurboAgent/blob/31ea8dadc00211b96544d3edd02ded1ef64e9856/turbo-agent.yaml#L3-L10)。本插件的用户配置范围 `2..8` 见 `src/auto.ts:32-34`；后文统一使用变量 `N`。

## 2. 三种奖励范围的成本、延迟、缓存和 coding-agent 适配性

记：

- `N`：本次候选数；
- `p`：PPT pivot 数，避免与论文 Appendix B.3 的 sampled-actions `k` 混淆；
- `C`、`K`：公式 (3.1) 的标准数和重复数；
- `T`：一条 coding-agent 轨迹的模型步骤数；
- `Q(N,p)=N+p(N-p)+p(p-1)/2`：论文给出的 PPT pair 数。

PPT pair 数来源于论文 §3.2，完整流程见 Appendix B.2 Algorithm 1；每个 pair 的每个 criterion/repeat 在作者实现中独立调用 verifier，因此 score cache 之前的 verifier 请求量为 `Q(N,p)·C·K`。[论文 §3.2、Appendix B.2 Algorithm 1](https://arxiv.org/html/2607.05391v2)；HTML 第 155–168、434–476 行；`_ref/llm_verifier/fine_grained_reward.py:812-829`、`:838-856`。`Q` 只统计 verifier pair，候选生成另行计费。

### 2.1 ORM：最终结果级 Best-of-N

**定义与时机。** 每个候选先产生最终结果，验证器在结束点比较 `N` 个 outcome。论文 Appendix B.3 明确把 ORM 与 coding/math Best-of-`N` 关联；该小节给出结果与验证预算，公开证据覆盖“最终结果级选择”。[论文 Appendix B.3](https://arxiv.org/html/2607.05391v2)，HTML 第 492–506 行。

**成本与延迟。** 生成成本约为 `N` 个完整结果；验证成本为一次 `Q(N,p)·C·K`。并发生成时，候选阶段墙钟时间受最慢结果控制，随后再支付 PPT 延迟。

**前缀缓存。** `N` 个生成请求共享任务输入前缀，输出从首个分歧 token 开始各自生成。verifier pair 的 task/outcome A/outcome B/rating-scale 前缀在同一 pair 的 criteria/repeats 间可以复用，条件见第 5 节。

**coding-agent 适配。** 它适合单轮补丁、代码生成和最终答案选择。交互式 coding agent 的质量证据大量来自命令、测试和工具输出，最终结果级输入提供的过程证据最少，且错误动作已经影响工作区后才得到分数。OMP 在线默认边界更适合 PRM。

### 2.2 PRM：逐步/轨迹前缀级验证

**定义与时机。** 在状态 `h_t` 处采样候选动作并验证当前 step 或对应 prefix。Appendix B.3 报告 sampled actions per step 从 `1` 增至 `9` 时，TauBench pass@1 从 `48.7` 增至 `55.7`，Terminal-Bench 从 `49.8` 增至 `54.3`。[论文 Appendix B.3 Table 10](https://arxiv.org/html/2607.05391v2)，HTML 第 492–500 行。

作者进度实现进一步提供两种 prefix 评分方式：离线 `track` 用一次 verifier 调用同时评分显式 `checkpoint_steps`，每个 repeat 的调用成本与轨迹长度无关；在线 `ProgressTracker.update` 每次只看当前 prefix，每个 update/repeat 一次调用，可用于决定何时 resample。`_ref/llm_verifier/progress.py:236-311`、`:314-320`、`:360-400`；`_ref/README.md:228-246`。

**成本与延迟。** 固定每步都用 `N` 个候选时，生成成本约为 `T·N`，PPT 上界约为 `T·Q(N,p)·C·K`；选择延迟在每一步重复出现。稀疏检查点将 `T` 替换为检查点数 `B`，普通步骤保留一次生成。

**前缀缓存。** 同一步的所有候选共享完整的 `h_t`，这是三种范围中最强的候选生成前缀复用条件。下一步只保留胜出历史，新的候选再次共享该历史。每一步新增消息和工具结果必须保持 append-only 和稳定序列化；候选之间改变 tool schema/order、`tool_choice` 或 cache identity 会破坏复用。

**coding-agent 适配。** 它能在写入或执行前拦截低质量动作，也能让后续判断利用已经观察到的文件、搜索和测试结果。固定逐步 BoN 对大量观测动作重复付费；稀疏 PRM 检查点在质量、速度和成本之间最符合 OMP 的交互结构。

### 2.3 TRM：完整交互轨迹级选择

**定义与时机。** 生成 policy 为每个任务产生 `N` 条完整轨迹，verifier 用 PPT 比较完整轨迹，再提交最高分轨迹。论文 §5 明确把主实验称为 TRM，并给出统一协议；Algorithm 1 第 3–5 行先采样 `N` 条 `τ_i`，第 6–21 行进行 PPT。[论文 §5、Appendix B.2 Algorithm 1](https://arxiv.org/html/2607.05391v2)，HTML 第 211–214、438–476 行。

作者 loader 把每个 agent step 的消息、命令和 observed output 写入 Terminal-Bench 轨迹；SWE formatter 同样保留 assistant 和 tool/user output。`_ref/llm_verifier/loaders.py:29-53`、`:155-185`。这说明 TRM 的“轨迹”包含环境证据。

**成本与延迟。** 需要 `N` 套完整 agent rollout。coding agent 还需要 `N` 个隔离 worktree/容器以及网络副作用隔离。验证只在结尾运行一次 PPT，但 prompt 很长；作者报告 Terminal-Bench 的一个 pair prompt 约含两条完整轨迹、约 `80k` tokens。`_ref/README.md:440-449`。

**前缀缓存。** 候选生成只在首次分歧前共享同一输入；分歧后的工具结果和状态不同，跨轨迹生成缓存复用下降。每条轨迹内部仍可复用自己的 append-only 历史。verifier 对同一 pair 的 criteria/repeats具有很高复用价值，作者的 criterion-at-tail 和 warm-up 将缓存命中率从 `5.2%` 提升到 `78.4%`，uncached input 约降低 `3.4×`。`_ref/README.md:440-449`。

**coding-agent 适配。** 它提供最完整的终局证据，适合离线评测、昂贵任务复跑和具备严格环境克隆的 harness。常规本地会话含不可安全复制的外部操作与用户交互，部署复杂度和总生成成本最高，因此适合作为评测/高保障模式，在线默认采用 PRM。

## 3. OMP 的实际调用边界

OMP agent loop 在工具调用后把 tool result 追加到当前 context，再进入下一次模型调用：

- 内外循环定义：`oh-my-pi/packages/agent/src/agent-loop.ts:1050-1055`；
- 每轮调用模型：同文件 `:1174-1191`；
- 解析可运行 tool calls：`:1283-1307`；
- 执行并追加结果：`:1358-1374`；
- 有工具调用时继续下一轮：`:1419-1445`；
- provider context 含当前历史与 tools：`:1514-1545`；
- 最终调用 `streamFunction(model,llmContext,options)`：`:1567-1585`、`:1631-1684`。

因此，注册在 provider `streamSimple` seam 的插件会在**每个 OMP 模型步骤**运行；一次读文件及其结果会促成下一次模型步骤。任务所指的每-request/action 基线位于本仓库 commit `438ccb714c6c9cf1a30fe3900e33c7c037eb7d1f` 的 `src/auto.ts:315-325`（可用 `git show 438ccb7:src/auto.ts` 复核）：每轮直接 fan-out 配置 `N`。这解释了连续读/搜场景的重复 BoN+PPT 成本。

本次实现先生成 proposal，仅当参数级 approval 为 `read` 且工具命中受审计 observation effect 时直接单轨回放；状态/控制提交、未分类 read-tier 工具及更高 tier 动作扩展至配置 `N`。实现同时落实第 4–7 节的执行前选择、winner-only、缓存身份与遥测合同。

OMP 的 `tool_call` extension hook 在并发调度、`tool_execution_start` 和审批 gate 之前运行，并允许 block/revise；`oh-my-pi/packages/coding-agent/src/session/agent-session.ts:1431-1434`、`:3422-3466`，事件类型位于 `.../extensibility/extensions/types.ts:916-929`、`:1271-1275`。provider seam 负责候选生成/选择，pre-tool seam 负责最终效果分级或执行前防线，二者可以共同保证“只执行 winner”。

## 4. 推荐边界：按影响面触发的稀疏 PRM

### 4.1 调度算法

对每个 OMP 模型步骤 `t`：

1. 构造唯一、稳定的已选轨迹前缀 `h_t`，包含用户任务、已执行的胜出动作及其观测结果。
2. 完成一个 proposal `a_t^1`。proposal 只产生声明式 tool call/回复，不执行工具。
3. proposal 是纯本地观测时，令 `N_t=1`，回放并执行它，把 output 追加到 `h_{t+1}`。
4. proposal 命中检查点时，再从完全相同的 `h_t` 生成 `N-1` 个候选；比较 `h_t ⊕ a_t^i`，先用严格多数决，未形成严格多数时运行论文 PPT。
5. 只向 OMP agent loop 回放 winner；工具执行完成后，把 winner 和真实 tool result 追加到轨迹。
6. 无 tool call 的终态回复始终是检查点；其 PPT 输入包含本轮所有此前观测证据。

proposal-first 让 consequential step 产生“一个 proposal 完成时间 + `N-1` 并行生成 + PPT”的延迟。纯观测步骤只支付一个生成请求；proposal 先完成还会为其余 `N-1` 个候选预热共同生成前缀。该实现优先保证缓存预热与统一调度语义。

执行前 PRM 判断的是“共同历史 + 候选动作”；该工具的真实 output 会在执行 winner 后出现，并进入下一检查点或终态判断。这样同时覆盖动作选择质量与后续结果证据。

### 4.2 哪些操作进入验证边界

OMP 已定义参数感知的 `ToolTier = read | write | exec`，缺少 approval 声明时调用方按 `exec` 处理；`resolveToolTier` 会执行动态 approval 函数。`oh-my-pi/packages/agent/src/types.ts:705-742`；`oh-my-pi/packages/coding-agent/src/tools/approval.ts:52-98`、`:104-114`。

该 tier 是审批能力等级。验证边界还需要一层 effect semantics，因为若干状态/控制工具声明为 `read`：`ask` 会请求用户决策（`.../tools/ask.ts:766-779`），`yield` 会提交结果（`.../tools/yield.ts:262-270`），`checkpoint/rewind` 创建或恢复会话检查点（`.../tools/checkpoint.ts:53-98`），`memory_edit` 会更新/遗忘长期记忆（`.../tools/memory-edit.ts:6-24`、`:34-48`），`todo` 可写任务状态（`.../tools/todo.ts:795-806`）。

| effect class | 典型操作 | `N_t` | 原因 |
|---|---|---:|---|
| `observation` | 本地 `read`/`grep`、`glob`、`ast_grep`、`lsp` 查询、只读 debug/GitHub/computer、`inspect_image`、recall、worker/job list/logs/wait | `1` | 获取证据且不提交外部状态；结果进入后续 prefix |
| `workspace_mutation` | write/edit/apply patch、删除/移动、依赖或配置变更 | 配置 `N` | 代码状态发生变化，适合执行前选择 |
| `execution` | bash、build、test、lint、install、git 命令、debug/eval | 配置 `N` | 任意命令可产生文件、进程或网络副作用；输出随后成为轨迹证据 |
| `external_effect` | deploy、数据库写入、GitHub/浏览器/computer 写操作、MCP 外部写、worker spawn/send/kill | 配置 `N` | 影响工作区外状态，回滚成本高 |
| `control_commit` | ask、yield、终态回复、checkpoint/rewind、计划/goal/todo 的变更、memory retain/edit/learn、worker cancel/send、审批或 handoff | 配置 `N` | 提交用户可见结论、任务控制流或长期状态 |
| `unknown_or_mixed` | 未注册工具、approval 解析失败、一个回复内混合 read 与高等级操作 | 配置 `N` | 使用批次中最高 effect；未知项保守进入检查点 |

参数感知必须保留：OMP 本地 read 为 `read`，SSH read 与特定 PDF image 读取为 `exec`；本地 grep 为 `read`，SSH grep 为 `exec`；glob 为 `read`。`oh-my-pi/packages/coding-agent/src/tools/read.ts:614-622`、`.../grep.ts:909-914`、`.../glob.ts:105-108`。bash 默认 `exec`，仅显式规则改变决策；`.../tools/bash.ts:551-579`。

`effect class` 应由工具注册元数据或稳定 adapter 提供。工具名 allowlist 只适合作为受测试的语义覆盖层，核心分级继续调用 OMP 的 `resolveToolTier(tool,args)`。纯观测条件是“批次内每个 call 都得到 `ToolTier=read`，且每个 call 的 effect class 都是 `observation`”。

### 4.3 读文件/搜索等观测操作的处理

纯观测操作直接执行一个候选，并把以下内容原序加入 `h_{t+1}`：

- assistant 可见动作或 tool call；
- 实际 tool output、错误标记、退出状态；
- 与该 step 关联的图像/附件；
- 任务要求与必要的先前上下文。

作者 Terminal-Bench formatter 每个 agent step 都保留 message、command 和 output；SWE formatter 保留 assistant 与 tool/user output。`_ref/llm_verifier/loaders.py:29-53`、`:155-185`。进度 prompt 明确要求 “Trust observed output”，并把 step 定义为 action + observed output；`_ref/llm_verifier/progress.py:79-116`、`:247-263`，`_ref/scripts/terminal_bench_progress.py:42-58`。多模态 per-step frame 会留在所有后续 prefix；`_ref/README.md:258-275`。

这样处理有三个直接效果：

1. 读/搜阶段保持单一 canonical history，下一次 consequential fan-out 的 `N` 个候选共享最长输入前缀；
2. verifier 在写入、执行后的下一检查点或终态回复处仍能看到读取内容与真实输出；
3. 总体成本从“为每次观测生成 `N` 个候选并运行 PPT”降为“一次生成并延后利用证据”。

搜索路径选择本身也可能影响质量。`all_steps_prm` 可作为论文 Appendix B.3 的固定逐步协议保留，用于高保障模式和对照实验；默认的 effect-gated 调度需要单独评测。论文 Appendix A 把由 verifier uncertainty 驱动的 adaptive compute allocation 列为未来工作，所以产品文档应把 operation-conditioned routing 标为工程策略及实验假设。[论文 Appendix A](https://arxiv.org/html/2607.05391v2)，HTML 第 419–422 行。

## 5. 前缀缓存复用条件

### 5.1 候选生成缓存

同一检查点的 pilot 与后续候选应满足以下完全相同的输入前缀条件：

- provider、model 和 credential/cache namespace 相同；
- `systemPrompt` 相同；
- messages 的内容、顺序和序列化相同；
- tools 的 schema、描述、顺序和 wire name 相同；
- `toolChoice` 相同；
- `sessionId`、`promptCacheKey` 和 provider session/cache state 保持同一会话身份；
- cache retention 允许缓存。

OMP 明确记录改变 `tool_choice` 会使 provider message cache 失效：`oh-my-pi/packages/agent/src/types.ts:79-87`。`sessionId` 用于 session-based caching：同文件 `:157-164`。AI 层将 `promptCacheKey` 定义为 OpenAI-family 的缓存身份并回退到 `sessionId`，同时把 `providerSessionState` 定义为会话级 provider 状态：`oh-my-pi/packages/ai/src/types.ts:468-501`；OpenAI cache key 的实际回退位于 `.../providers/openai-shared.ts:462-466`。

当前候选请求通过 `{...streamOptions}` 保留这些身份，并关闭 server-side turn chaining；`src/auto.ts:741-777`。保留 cache identity 与关闭候选间 response chaining可以同时成立：前者复用输入前缀，后者保证候选输出相互独立。

### 5.2 verifier prompt 缓存

作者实现把 task、trajectory A、trajectory B 和 rating scale 放在 criterion 之前，只让 criterion 在尾部变化；`_ref/llm_verifier/fine_grained_reward.py:714-748`。它以 `(task_name, slot-A, slot-B)` 标识 distinct prefix，先让每个 prefix 的一个请求完成，再并发其余 criteria/repeats；`_ref/llm_verifier/fine_grained_reward.py:838-876`、`:895-939`。

可执行条件：

- 同一个 `(A,B)` slot 顺序、task/history 文本、rating scale 和图像布局形成同一 prefix；
- A/B 交换形成另一个 prefix，需要单独 warm；
- criterion 放在尾部；
- 同一 prefix 的首个请求完成后再 fan-out；
- score cache 只复用完全相同的 model/provider、prompt version、criteria、`K`、task identity、candidate content/图像及 slot/repeat identity；
- cache hit 的调用不计 verifier token/call，作者 README 明确这样统计：`_ref/README.md:451-480`。

跨检查点时，新增的轨迹证据会改变 verifier prompt；复用主要来自共同历史前缀和检查点内部的 criteria/repeat fan-out。TRM 的长 pair prompt最能受益于 verifier prefix cache，PRM 的同一步候选生成最能受益于 generator prefix cache。

## 6. 成本模型与预期延迟

令纯观测步骤数为 `R`，consequential checkpoint 数为 `B`，`T=R+B`。忽略重试和严格多数提前结束：

| 协议 | generator 请求 | verifier 请求上界 | 主要墙钟结构 |
|---|---:|---:|---|
| 每步 request/action BoN | `T·N` | `T·Q(N,p)·C·K` | 每步 `parallel(N) + PPT` |
| 推荐的稀疏 PRM | `R + B·N` | `B·Q(N,p)·C·K` | 观测为 `single`；未知 checkpoint 为 `pilot + parallel(N-1) + PPT` |
| 完整 TRM | `N` 条完整 agent rollout | `Q(N,p)·C·K` | `N` 条完整轨迹并行/串行完成后再 PPT |

相对每步 BoN，稀疏 PRM 在 generation 侧节省 `(N-1)·R` 个请求，在 verifier 侧节省 `R·Q(N,p)·C·K` 个请求。严格 action majority 可以将某些检查点的 verifier 请求降为 `0`；TurboAgent 官方默认启用 majority shortcut，未达多数才进入 pivot tournament。[`turbo-agent.yaml:29-35`](https://github.com/llm-as-a-verifier/TurboAgent/blob/31ea8dadc00211b96544d3edd02ded1ef64e9856/turbo-agent.yaml#L29-L35)。

质量取舍集中在观测路径选择：固定逐步 PRM 对每次“下一步读什么”也做搜索；稀疏 PRM 把计算集中到写入、执行、控制与终态。coding agent 通常包含大量读/搜步骤，因而总体速度收益显著；具体 pass rate、端到端时延和 cached-token 比例需要在目标模型与 OMP harness 上实测。

## 7. 可执行不变量

以下不变量应作为实现、测试和日志的验收合同：

1. **GS-01 / taxonomy**：运行模式只使用 `orm`、`prm`、`trm` 三种奖励范围；`request`、`checkpoint`、`tool tier` 作为 scheduling/boundary 字段记录。
2. **GS-02 / configurable N**：consequential checkpoint 使用用户配置 `N∈[2,8]`；代码、日志、UI 和测试均不得把 `3` 当作协议常量。
3. **GS-03 / single observation**：只有“所有 tool calls 均由 OMP 参数感知解析为 `read`，且 effect class 全部为 `observation`”时允许 `N_t=1`。
4. **GS-04 / semantic override**：ask、yield、终态、checkpoint/rewind、状态变更、长期记忆变更、worker 控制、外部写入始终进入 checkpoint，即使 approval tier 为 `read`。
5. **GS-05 / fail closed**：未知工具、缺失工具、approval/effect 解析异常、malformed tool-use 和混合批次使用配置 `N`。
6. **GS-06 / pre-execution**：候选生成和 PPT 完成前，任何候选工具均不得执行；provider-native `execHandlers` 路径必须拒绝或转成声明式调用。
7. **GS-07 / winner only**：agent loop 只能看到并执行一条 winner；loser 的 tool-call ID、输出和副作用均不得进入会话。
8. **GS-08 / prefix-scored candidate**：PPT 比较输入为共同的 selected history/observations 加候选动作 `h_t ⊕ a_t^i`，不能只比较脱离历史的 tool call 文本。
9. **GS-09 / observation retention**：read/search/test output、错误、退出状态和图像按时间顺序保留在后续 verifier 可见轨迹中；隐藏 reasoning、transport metadata 和 tool schema 不作为轨迹证据。
10. **GS-10 / evidence budget**：TRM/PRM 默认保留作者 formatter 所需的完整可见证据；任何截断必须绑定公开 formatter 规则和 telemetry。作者 Terminal-Bench 使用完整 command/output，SWE 使用每块 `2000` 字符规则；论文的缓存方案以两条约 `80k` token 完整轨迹为设计目标。来源：`_ref/llm_verifier/loaders.py:29-53`、`:155-185`，`_ref/README.md:440-449`。
11. **GS-11 / cache identity**：同一 fan-out 的 model、context、tools/order、toolChoice、sessionId、promptCacheKey 和 provider cache state 保持一致；每次 decision 记录 cache identity hash 和 cached/uncached token usage。
12. **GS-12 / warm then fan-out**：未知边界 pilot 完成后再发 `N-1`；verifier 每个 distinct `(task,A,B)` prefix 先 warm 一个请求再并发 criteria/repeats。
13. **GS-13 / paper reward**：评分继续使用公式 (3.1) 的 token-logprob expectation、公式 (3.2) 的 Bradley–Terry preference 和 Algorithm 1 的 PPT；fallback/离散评分必须在 telemetry 标为 degraded。
14. **GS-14 / final boundary**：无 tool call 的用户可见终态、子任务 yield、handoff 和向用户提问始终验证；session end 只负责收尾记录，不能成为唯一验证时点。
15. **GS-15 / no arbitrary windows**：固定“每 M 次工具调用”或“每 X 秒”只具备工程启发式属性；默认边界由 effect semantics 决定，实验模式单独命名和记录。
16. **GS-16 / isolated TRM**：完整 TRM 只在候选 worktree/容器、进程、端口和外部副作用均可隔离时启用；最终只合入 winner 的变更。
17. **GS-17 / progress separation**：progress score 是监控/重采样信号；BoN/PPT selection 才决定 winner。TurboAgent 官方 progress monitor 在回复选择后后台运行且不改变响应：[`backend.py:272-316`](https://github.com/llm-as-a-verifier/TurboAgent/blob/31ea8dadc00211b96544d3edd02ded1ef64e9856/turbo_agent/proxy/backend.py#L272-L316)、[`monitor.py:1-12`](https://github.com/llm-as-a-verifier/TurboAgent/blob/31ea8dadc00211b96544d3edd02ded1ef64e9856/turbo_agent/progress_monitor/monitor.py#L1-L12)。
18. **GS-18 / evidence label**：operation-conditioned `N_t` 路由在文档与实验中标为 paper-bounded scheduling policy；固定每步 `N` PRM、最终结果 ORM 和完整轨迹 TRM 作为理论对照组。

## 8. 验收实验

实现阶段至少比较以下三组，全部使用同一模型、reasoning effort、任务集和 `N/p/C/K`：

1. `all_steps_prm`：每个 OMP 模型步骤均使用 `N`，对应当前 request/action 基线；
2. `effect_gated_prm`：本文推荐协议；
3. `trm_isolated`：在可隔离任务子集上生成 `N` 条完整轨迹。

每组记录：任务通过率、首次有效工具动作时间、总墙钟时间、generator/verifier calls、input/output/cached/uncached tokens、PPT/majority/single 比例、按 effect class 的 checkpoint 数、fallback/degraded 次数、错误工具执行数以及 winner 以外副作用数。质量门槛使用真实 coding-agent 任务的测试/hidden grader；缓存结论使用 provider usage 中的 cached token 计数，避免用请求数量推测命中率。

## 9. 来源索引与证据边界

### 一手来源

- 论文：[LLM-as-a-Verifier: A General-Purpose Verification Framework, arXiv:2607.05391v2](https://arxiv.org/html/2607.05391v2)。核心位置：§1；§2；§3.2 公式 (3.1)、(3.2)；§4；§5；§6；Appendix A；Appendix B.2 Algorithm 1；Appendix B.3 Tables 10–11。
- 作者参考实现：`_ref`，commit `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`，remote `https://github.com/llm-as-a-verifier/llm-as-a-verifier.git`。
- 作者 coding-agent 实现：[TurboAgent](https://github.com/llm-as-a-verifier/TurboAgent/tree/31ea8dadc00211b96544d3edd02ded1ef64e9856)，commit `31ea8dadc00211b96544d3edd02ded1ef64e9856`。
- OMP 官方源码：[can1357/oh-my-pi](https://github.com/can1357/oh-my-pi/tree/de5ffc4201c4941992402daea355adc1aad3a8db)，commit `de5ffc4201c4941992402daea355adc1aad3a8db`。

### 证据边界

- 论文直接命名并评估 TRM、PRM、ORM 三种奖励范围。
- 论文直接评估固定每步 sampled-actions 数的 PRM；作者代码直接支持显式 trajectory checkpoints 和在线 prefix progress。
- TurboAgent 直接实现每 request 的并行 `N` 候选与 PPT；该 seam 属于 coding-agent 集成选择。
- effect-gated PRM 组合了论文支持的 PRM、`N_t=1/N` 计算档位、显式 checkpoint 和 OMP 一手 effect 信息。按操作影响面分配 `N_t` 的收益需要第 8 节实验确认。
- 完整 TRM 是论文主测试时扩展协议；把任意多个工具动作拼成“微轨迹”并在共享工作区并行执行缺少论文实验与安全隔离保证，因此本文推荐使用“共同已选 prefix + 候选下一动作”的 PRM 形式扩大验证跨度。
