# 论文与作者实现核验（2026-09-05）

本次重新访问一手来源并读取固定提交源码；不是复述旧理论基线。审计只覆盖方法契约及能力边界，测试和真实服务商运行结果由本轮实现验收另行记录。

## 来源版本与证据

| 来源 | 本次直接读取的位置 | 结论 |
| --- | --- | --- |
| 论文 | [arXiv:2607.05391v2，HTML](https://arxiv.org/html/2607.05391v2)，页面标记 07 Jul 2026；§3.2、§4、Appendix B.2/B.3/B.6 | 成功访问，公式和 Algorithm 1 可读；未用未固定版本链接代替 v2。 |
| 作者主库 | [8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770) 的 raw Python 源码 | 本地 `git -C _ref rev-parse HEAD` 返回同一 SHA，`git -C _ref status --porcelain` 为空。没有追踪或更新上游 HEAD。 |
| TurboAgent | [eeb61be9cb618ea9c52262cebf15092e7c185146](https://github.com/llm-as-a-verifier/TurboAgent/tree/eeb61be9cb618ea9c52262cebf15092e7c185146) 的 `verifier.py`、`backend.py`、`turbo-agent.yaml` | 成功访问固定源码；多数条件与动作格式均直接核验。 |
| 本插件 | CodeGraph 返回的 `src/ppt.ts`、`src/scale.ts`、`src/auto.ts` 相关源码，以及 `README.md`、`package.json` | 仅将实际读到的实现标记为已支持；本次并行工作中的改动不等同测试已通过。 |

网页提取行号会随工具的空行处理变化，因此下面引用固定文件和符号，不将网页行号当作 GitHub 源码行号。

## 方法契约与可执行不变量

论文核对结果：Eq. (3.1) 是跨 C 个准则、K 次评估平均的评分 token 数值期望，先线性归一化奖励，再用 Eq. (3.2) 的 `sigmoid(Ra-Rb)`。Algorithm 1 对有向 ring 做集合差；pivot–pivot 方向按候选下标升序。PRM 附录的 k 是每步动作样本数，区别于 PPT 的 pivot 数。B.6 是分析模型与概率评分模型串联。[论文 §3.2、Appendix B.2/B.3/B.6](https://arxiv.org/html/2607.05391v2)

以下是据此构造的本项目验收条件，不是论文实验结果：

| 检查项 | 可运行断言 |
| --- | --- |
| 奖励与聚合顺序 | 构造 A/T 概率 0.75/0.25，归一奖励应为 0.75；跨 C/K 先平均奖励再 sigmoid，使用非对称多次分数区分“平均 sigmoid”的错误版本。 |
| 评分尺度 | 固定 A→20、T→1、大小写兼容；缺失有效概率分布不得计入纯 logprob 路径。 |
| K 独立性 | rep=0/1 缓存键不同；奇数轮 A/B 互换后必须映射回原候选身份。 |
| C 独立性 | 每准则独立 prompt 和评分；准则重排不改变平均值；不能把多个准则拼成一次复合评估冒充 C 次。 |
| 有向 ring | 多个 N/seed 下每个候选恰有一次入边和出边；反向边不能被视为同一边。 |
| PPT 两阶段 | pivots 使用 ring 的 w/c 排名；第二阶段边等于 nonpivot→pivot 与升序 pivot→pivot 的并集减去 ring；两个阶段共享 w/c。 |
| 调用预算 | N=3、k=1 为 4 个逻辑比较；N=3、k=2 为 4–5；分别统计逻辑边、唯一 C/K 评分键与实际 provider calls。 |
| 多数提前结束 | 必须 `count > 配置总样本数/2` 才能在剩余样本未结束时锁定；返回匹配动作中最早的候选。等于一半不可短路。 |
| 失败与证据 | 失败平局不能持久缓存为成功评分；副作用只执行 winner；共享上下文、工具 schema 和图像顺序在采样与验证之间一致。 |

作者固定评分代码使用 A–T、同 raw score 的概率取最大值、对保留质量归一化；最后 score tag 优先，空白 token 不覆盖前一位置分布；奇数 rep 换位后恢复候选身份，失败平局只在本轮结果中存在。[作者 `fine_grained_reward.py`](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py)

## 必须保留的歧义和兼容选择

1. **论文 Algorithm 1 与作者代码不同。** 正文给出的未扣重计数可作为 Algorithm 1 上界；固定作者 `select_best` 把全部 pivot pair 再次累计，未减去 ring。本插件 `pivotRoundPairs` 已按有向集合差实现，且先排序 pivots，符合形式化伪代码；不能为了 Python parity 把重复加权重新引入。[作者 PPT 源码](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/pivot_tournament.py)
2. **作者概率兼容值不等于完整词表精确期望。** 同值别名取 max 而非对不同 token 概率求和，且只对返回的有效质量归一化。现有兼容选择可以保留，但 `paperEquivalent` 只能表示采用该 logprob 兼容算法，不能保证全部 20 级概率均已获得。应同时看 support 和 probabilityMass。[作者评分源码](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py)
3. **精确动作等价不等于语义等价。** TurboAgent 对 `format_action` 字符串计数，内容包括可见文本、顺序排列的工具名和原始参数字符串，不含 tool-call ID；JSON 空白或键顺序不会自动标准化。OMP 若只能取得解析后的参数，需要单独说明序列化映射，不能称为原始字符串完全 parity。[动作格式](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/proxy/backend.py)、[严格多数](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo_agent/verifier/verifier.py)
4. **在线配置不是论文主 benchmark 配置。** TurboAgent 固定配置是 N=3、pivots=2、K=1、一个 Task Success criterion，多数决开启；这是本插件在线默认值来源。[固定配置](https://github.com/llm-as-a-verifier/TurboAgent/blob/eeb61be9cb618ea9c52262cebf15092e7c185146/turbo-agent.yaml)

## 能力边界与完整闭环标准

本次可确认的本地实现：PPT 有向扣重、soft win 与累计均值选择；A–T 固定尺度；在线配置提供候选数、K 和 pivots。README 声明的观测单样本与高影响检查点扩展，是本项目的 PRM 调度策略，不能据此推导论文的质量提升。

以下能力没有在本次源码范围内得到完整实现或实测证据，不能写作“已完整复现论文”：

- B.6 的闭源分析→独立开放概率评分串联；仅更换 `verifierModel` 不等于该流程。
- 论文 RL 训练、轨迹进度相关性实验、机器人和医学任务 harness，以及完整隔离轨迹的 TRM benchmark。
- 真实 OMP 会话和 provider 的本轮成功率、选择质量收益、取消后的实际计费减少；fixture 只能证明本地契约。
- `top_logprobs` 请求配置之外的实际分布完整性，以及 effect-gated 调度相对全步采样的无偏性。

本轮可形成的工程闭环是：对应不变量回归 → 全量测试与类型检查 → 独立审查 → 真实服务商/OMP 验证（具备运行条件时）。若最后一项没有执行，结论应为“算法与本地集成验证通过，真实运行未验证”，并保留原因；不以论文原始 benchmark 数字代替本插件测量。
