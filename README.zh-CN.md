# omp-llm-verifier 中文说明

## 引用

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 参考实现](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

插件采用论文的 20 级 token logprob 评分和 Probabilistic Pivot Tournament。每个 OMP 终局回答默认扩展为 3 个完整候选，使用已配置的验证器模型评分（留空时采用会话默认模型），并回放胜出响应的文本、推理、图片和终止状态。工具调用轮次直接延续当前轨迹。

版本化来源、形式化不变量、产品边界和防漂移门禁记录在 [docs/theory-baseline.md](docs/theory-baseline.md)。

## OMP 安装与使用

```bash
omp plugin install https://github.com/takltc/pi-llm-as-a-verifier.git
omp plugin config set omp-llm-verifier enabled true
omp plugin config set omp-llm-verifier candidateCount 3
omp plugin doctor
```

`enabled` 控制自动验证，`candidateCount` 控制候选数量，范围为 2-8，默认值为 3。配置在新 OMP 会话中生效。验证器默认继承 OMP `modelRoles.default` 的模型、推理等级、凭据、请求头和兼容配置。

按照论文的 智能体≠验证器 架构（参考实现用 Gemini 或 DeepSeek 给 GPT/Claude 的轨迹打分），可以通过 OMP 模型选择器指定独立的验证器模型：

```bash
omp plugin config set omp-llm-verifier verifierModel deepseek/deepseek-v4-flash
```

`verifierModel` 留空时使用会话默认模型做自我验证。kimi-code（Kimi for Coding）会拒绝 logprobs 请求，会话模型是 kimi-code 时必须配置 `verifierModel`。

仅关闭自动验证并保留插件安装状态：

```bash
omp plugin config set omp-llm-verifier enabled false
```

停用或恢复插件加载：

```bash
omp plugin disable omp-llm-verifier
omp plugin enable omp-llm-verifier
```

验证器模型必须能提供 OpenAI Chat Completions 或 Responses 的 token logprobs——论文的细粒度奖励从评分 token 分布读取。解析出的验证器模型缺少 logprobs 能力时，插件会拒绝包装会话模型并显示能力警告。任务包含视觉证据时，验证器模型还需支持图片输入。每次成对评分请求先携带共享的用户、assistant 和工具结果图片，再携带带标签的 Trajectory A 与 Trajectory B 图片，从而保持参考实现的多模态证据路径。

已验证的比较结果会缓存在项目根目录的 `.omp-llm-verifier-cache.json`，缓存键覆盖任务、有序共享图片、候选专属图片、两个候选轨迹、评分标准、模型和 prompt 版本的内容指纹，因此相同内容的重复验证不消耗验证器 token。该文件可以直接删除，建议加入 gitignore。

每个被验证的回答都是可观测的：OMP 控制台上，包装器会针对每个最终答案输出一行
`event:decision` JSON，包含 `path`（获胜者如何被选出——`verifier` 表示走了论文的
PPT 锦标赛，`fallback` 表示验证未能执行，`aborted` 与 `error` 表示终止状态）、
获胜候选的索引与平均得分、有向比较次数、参与打分的验证器模型与 prompt
契约版本，以及本次请求的验证器 token 用量。`scoreSources` 分别统计 logprob
期望、文本字母回退、运行期中性平局和旧版/未知缓存项；全部评分标签均来自 token
logprobs 时，`paperEquivalent` 为 `true`。`scoreDistribution` 记录这些标签的有效
A–T 支持数与返回概率质量的最小值和平均值。
扩展程序也可以通过包装 provider 状态上的 `onDecision` 回调读取同样的数据。

终态选择只要拥有至少两个成功的终态候选，就会执行论文的 PPT 路径，候选文本重复时
同样执行。工具调用轮次会直接延续当前智能体轨迹；替代采样提出的新工具动作保持在
终态锦标赛之外，并通过 `nonterminalCandidates` 记录。

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
