# 真实 OMP 运行验证（2026-09-05）

最终真实闭环已通过：自动 OMP 路径完成 3 候选、4 次比较、8 个 logprobs 评分且无回退；独立已知答案测试正确选出 323，暖缓存为 0 次调用。下面保留修复前失败、根因诊断和修复后证据；这不等同于论文完整基准复现。

## 范围与环境

- OMP 18.1.10；Bun 1.4.0；插件 0.3.0。
- `omp plugin doctor`：5 项通过，0 警告，0 错误。
- `--no-tools` 禁用内置工具；`--no-extensions` 禁用扩展发现，只显式加载本项目扩展。
- `--no-session` 不保存会话；会话限制 240 秒。
- 仅使用随机六位数字请求，没有执行工作区操作。进程日志保留在 OMP 自有日志目录，本文件只记录白名单字段，没有复制原始请求、上下文、凭据或私密配置。

## 可复核命令

```sh
rtk proxy omp -p --no-session --no-extensions \
  -e /Users/tak/unicom/dev/pi-llm-as-a-verifier/src/index.ts \
  --no-tools --max-time 240 --llm-verifier \
  '请只回答一个随机六位数字。不要解释。'
```

进程 PID：34481。OMP 自有日志：`~/.omp/logs/omp.2026-09-05.34481.log`。决策时间：`2026-09-05T20:28:19.324+08:00`。

## 实际结果

| 字段 | 实测值 |
| --- | --- |
| 进程退出码 | 0 |
| 最终输出 | `583204` |
| 生成器供应商 | `crofai`（日志没有记录可确认的生成器模型 ID） |
| 验证器 | `openrouter/deepseek/deepseek-v4-flash-0731` |
| 决策路径 | `verifier` |
| 检查点 | `terminal_response` |
| 配置 / 采样 / 成功候选 | 3 / 3 / 3 |
| 工具候选 / 终结候选 | 0 / 3 |
| 比较次数 | 4 |
| 决策耗时 | 81,719 ms |
| scoreSources | logprobs=0, textFallback=8, neutralTie=0, unknown=0 |
| paperEquivalent | **false** |
| promptVersion | `pairwise-granularity20-v5` |
| 验证器调用数 | 4 |
| 验证器输入 / 缓存输入 / 输出 token | 2,024 / 0 / 1,392 |
| 其中 reasoning token | 1,210 |

OMP 同时记录了 `degraded` 警告。`scoreDistribution.logprobScores=0`，支持集和概率质量统计均为 0。这表示当前运行没有可用于论文概率期望计算的评分分布，不代表供应商在任何请求下都不支持 logprobs；具体回退原因仍需单独诊断。

本次证实真实 OMP 插件加载、候选生成、PPT 比较、降级可观测性及最终返回完整跑通；未证实概率评分闭环，也不包含生产任务质量或论文基准指标复现。

## 回退根因的独立诊断

通过临时诊断扩展进行两层观测，未改生产代码。进程 37179 包装 scoreReply；进程 38151 只包装 fetch 的 verifier JSON 响应。另一个独立扩展实例包装未命中生产模块的运行（36799）不作为解析诊断证据。

- 评分字母 token ` A` 有有效概率，例如 A=-0.0000011920922、S=-16.576727、T=-16.868456；并非供应商完全不返回评分概率。
- 原始 message.content 含 `<score_A> A `，但原始 token 流是 `.\n\n` → `score` → `_A` → `>` → ` A`；开标签 `<` 缺失，B 同样。
- `score` 的原始 bytes 和 codepoints 均为 `[115,99,111,114,101]`，前序 token 为 `[46,10,10]`，均不含 `<` 的 60；闭标签 ` </` 的 `[32,60,47]` 正常。字节恢复无法补回缺失内容。
- 各位置有 20 个 top alternatives，数字类型正常。排除客户端概率类型校验、日志尖括号渲染及单纯供应商无概率的解释。
- 38151 于 20:34:45.269 完成，exit 0，输出 `483920`，4 次比较 / 8 个 textFallback / paperEquivalent=false，耗时 40,362 ms；输入 2,064、输出 3,956、reasoning 3,722 token。

结论：当前供应商路由的 token/bytes 与 message.content 不一致，使精确标签定位失败。没有用模糊匹配掩盖该数据缺陷，诊断时该路由的真实论文概率评分尚未验收通过；修复后结果见文末。详细假设与排除链见 DEBUG.md 本轮追加段。


### E3：完整文本等式验证

进程 38919 使用新的随机请求，只记录整串等式布尔值与长度/标签计数，未保存原始完整文本。4 次比较的 tokenCount 为 52、64、72、54；全部满足：

- `message.content === tokens.join("")` 为 false。
- `message.content.replace(/<(?=score_[AB]>)/g, "") === tokens.join("")` 为 true。
- lengthDelta=2、openingTags=2。

能力探测单字母回答 exact=true、lengthDelta=0。最终输出 736482，exit 0，4 次比较仍是 8 个 textFallback；生产代码未变。

这进一步将差异严格限定为两个开评分标签缺失的 `<`，排除了本次响应存在其他未观察到的正文差异。可以研究以完整文本等式为硬门槛、保留每个概率位置不动的精确边界恢复；不允许模糊后缀定位或在其他差异上强行恢复。


## 严格边界恢复后的已知答案与缓存闭环

临时 OMP command 通过 getPluginSettings/resolvePluginSettings 读取已有 verifier 选择器，通过 createDefaultVerifierClient 复用 OMP 的模型解析和认证，再调用生产 select。没有修改模型设置，没有输出凭据；禁用所有内置工具和其他扩展。缓存位于单独 mkdtemp 目录，冷/暖两次完成后删除。

- 问题：`Compute 17 × 19. Return the exact integer product.`
- 候选：`Final answer: 322`、`Final answer: 323`、`Final answer: 324`，名称分别为数字字符串。
- 准则：`correctness: The final integer must equal the exact product 17 × 19.`
- 选项：pivots=2、nEvaluations=1、seed=0、maxWorkers=4、onError=raise。
- 进程 43769，2026-09-05 20:41:59.965–20:41:59.966，exit 0。

| 运行 | winner/index | comparisons | 实际调用 | scoreSources | paperEquivalent | 输入/输出/reasoning token |
| --- | --- | --- | --- | --- | --- | --- |
| 冷缓存 | 323 / 1 | 4 | 4 | logprobs=8，其余=0 | true | 1676 / 903 / 696 |
| 暖缓存 | 323 / 1 | 4 | 0 | logprobs=8，其余=0 | true | 0 / 0 / 0 |

两次 scores 完全相同：`[0.38447077309315353, 0.7310585041111404, 0.345960980493424]`。对应验证器都是 `openrouter/deepseek/deepseek-v4-flash-0731`。命令回调同时断言 winner=323、paperEquivalent=true、textFallback=0、冷调用数大于0、暖调用数等于0。

准备 harness 时曾因准则误写成含text字段的数组触发输入校验，该次未进入推理；改为生产支持的准则字典后再执行以上记录。


## 修复后最终真实自动验证闭环

client 层现在采用严格恢复：只有完整 message.content 删除评分开标签的 `<` 后与 token 拼接逐字符完全相等，才补回缺失边界；保留原 token 数量与每个 positionLogprobs 索引，任何其他正文差异都不恢复。Chat Completions 和 Responses 均有回归测试。

单独六位数字请求在进程 42931 得到 3 个完全相同动作，走 majority/0 comparisons，因此只算多数捷径验证，不算概率评分验证。为确保候选不同，最终请求附加随机 nonce：

```sh
rtk proxy omp -p --no-session --no-extensions \
  -e /Users/tak/unicom/dev/pi-llm-as-a-verifier/src/index.ts \
  --no-tools --max-time 240 --llm-verifier \
  '最终验证轮次20260905H：请输出JSON对象，number是随机六位数字字符串，nonce是随意生成的20个小写英文字母。每个字段只生成一次。不解释。'
```

- 进程 43676，决策时间 2026-09-05T20:43:41.292+08:00；exit 0，耗时 122,566 ms。
- 最终输出：`{"number":"731946","nonce":"xqmwplzrkfctvhyabjdg"}`。
- 验证器：openrouter/deepseek/deepseek-v4-flash-0731；promptVersion=pairwise-granularity20-v5。
- path=verifier；sampledCandidates=3、successfulCandidates=3、terminalCandidates=3、toolUseCandidates=0。
- nComparisons=4；scoreSources：logprobs=8、textFallback=0、neutralTie=0、unknown=0；**paperEquivalent=true**。
- 分布质量：minSupport=13、meanSupport=15.125、minProbabilityMass=0.9999976647876558、meanProbabilityMass=0.9999996923878773。
- 验证器 usage：calls=4、inputTokens=2319、cachedInputTokens=256、uncachedInputTokens=2063、outputTokens=20958、reasoningTokens=20603。

结论：真实候选生成→PPT→概率评分→赢家返回、已知答案正确性和缓存复用均通过。尚未执行论文完整 benchmark，也不能由单次算术正确性推导所有生产任务质量。没有修改用户配置或凭据，临时诊断/已知答案 harness 与临时缓存已清理；OMP自身日志按宿主策略保留。
