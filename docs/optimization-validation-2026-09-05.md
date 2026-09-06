# 整体优化验收（2026-09-05）

本轮核验范围是 OMP Coding Agent 插件及离线选择 API：可见证据 → 候选采样 → 评分分布 → C/K 聚合 → PPT → 胜者回放 → 缓存与遥测。不是论文全部研究任务的复现。

## 来源与粒度选择

论文固定为 [arXiv v2](https://arxiv.org/html/2607.05391v2)，作者参考固定为 `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`，TurboAgent 固定为 `eeb61be9cb618ea9c52262cebf15092e7c185146`。一手来源核对、算法不变量和兼容差异见 [论文审计](paper-audit-2026-09-05.md) 与 [理论基线](theory-baseline.md)。

推荐默认为**关键动作批次 PRM**：观测 `N=1`；一个模型响应中的整组声明式动作只选择一次，再执行胜出批次；默认 `N=3、pivots=2、C=K=1、G=20`。依赖工具结果的后续决策保留边界。该批次机制原已存在，本轮明确默认语义并增加可执行对照，没有凭空改变论文评分或宣称实现隔离的多步 rollout。

| 固定回放场景 | Generator 请求 | Verifier 请求 |
| --- | ---: | ---: |
| 六次读取、三次独立写入、一次终态 | 18 | 20 |
| 六次读取、三次写入合批、一次终态 | 12 | 10 |

此 fixture 的每个非多数检查点需要 5 次比较，通用默认范围为 4–5。观测每次严格 1 次生成、0 次验证。以上仅证明调度调用量降低，不是实际时延或任务质量提升；实际代理是否能合批取决于动作依赖关系。粒度选择及阶段式方案的限制见 [粒度研究](granularity-selection.md)。

## 已修复问题与回归

| 问题 | 最终实现 | 可运行证据 |
| --- | --- | --- |
| 长任务挤掉已观察工具结果、候选内容被固定字符预算截断 | 完整保留可见历史与候选，图像独立保全 | `runtime-audit`、`verification`、`paper-parity` |
| 文本伪装工具标记或跨块拼接导致错误多数 | 带类型、块边界、结束状态的结构身份；忽略传输 ID、规范对象键顺序 | `runtime-audit` |
| `null`、布尔、字符串、正 logprob 被当作概率；继承属性混入 A–T | 只接纳有效数值 logprob 与真正评分值 | `client-audit` |
| 全空概率位置仍通过能力探测 | 至少一处有效分布，否则判 unsupported；保留部分空位置索引 | Chat/Responses `client-audit` |
| 实际供应商 token/bytes 缺失评分标签 `<` | 仅当完整正文去掉这些字符后与 token 流完全相等，才恢复开标签，概率索引不变 | Chat/Responses `client-audit`；真实 OMP |
| 并发选择用量串扰，失败重试/无 usage 响应漏计调用 | 每次选择独立计量，HTTP 尝试与返回 token 分开统计 | 并发本地 HTTP `client-audit` |
| 读取响应正文时超时被包装成 JSON 错误 | 保留取消原因，超时按现有预算重试 | 流式 HTTP `client-audit` |
| 原地修改图片后复用旧指纹 | 引用缓存附带内容快照，变化后重新哈希 | `cache-concurrency` |
| 自制过期锁回收 TOCTOU 删除其他进程新锁 | 复用已有 OMP 原生 OS 锁；锁内合并与原子替换；旧目录拒绝自动删除 | 六进程写入、活锁排斥、SIGKILL 恢复测试 |
| 大量跳过任务同步递归导致栈溢出 | 调度循环防重入；后继逐项入队 | `scheduler-audit` 的 50,000 个任务 |
| 干净 checkpoint 重复落盘、统计数组展开上限 | checkpoint 保存后清脏；统计单遍 O(1) 额外空间 | 现有调度缓存回归、百万评分 `cache` 测试 |

缓存版本更新为 7，排除旧数值解析的缓存观测。复用原生依赖，没有安装新的第三方实现；锁协议迁移要求见 README。

## 验证层级

最终本地结果：`bun test` 为 **147 pass、0 fail、10 个文件**（14.93 秒）；类型检查与 `git diff --check` 通过。冻结依赖检查为 151 个安装、184 个包，无变更。

- 本地算法：实际执行固定 Python 源码提取函数，60 个回复、120 个分值与 TypeScript 一致；C/K 先平均再 Bradley–Terry、环位置平衡、有向 pivot 集合差均有回归。
- 工程集成：真实本地 HTTP、取消/超时、并发选择、真实多进程缓存、进程异常退出恢复，以及整个 wrapper 的动作批次回放。
- 真实 OMP：3/3 候选、4 次比较、8 个 logprob 评分、全部回退为 0、`paperEquivalent=true`。
- 已知答案：同一真实验证器在 `322/323/324` 中正确选择 `17×19=323`；冷缓存 4 次调用，暖缓存 0 次调用，winner 和分数一致。
- 独立审查：Standards、Spec、thermo-nuclear 与 ponytail 审查已完成；全空概率探测问题经复现修复后复审关闭。缓存实现由另一代理独立审查，没有把作者自审算独立审查。

真实模型、时间、精确进程、token 及失败→修复的证据见 [运行验收](runtime-validation-2026-09-05.md)。`paperEquivalent` 表示当前兼容 logprob/PPT 路径，并不保证完整词表概率或复现论文 benchmark 效果。

## 可复跑命令与范围

```sh
rtk bun install --frozen-lockfile --ignore-scripts
rtk bun test
rtk bun run typecheck
rtk git diff --check
```

Python 差分需要 `python3` 和固定 `_ref` 副本，缺失时测试明确 skip；本机两者可用并实际执行。进程锁的运行结果仅覆盖当前 macOS，不宣称 Linux/Windows 已运行通过。

当前没有进行多任务 Coding Agent 成功率/P95 延迟的真实消融，也没有实现论文 RL、B.6 双阶段分析/评分或隔离 TRM 执行环境。高推理预算下单次真实自动检查点仍耗时 122.566 秒；本轮没有降低用户显式推理等级来掩盖该成本。

工作区原有未提交修改均保留，基准 HEAD 为 `a8adea90fa282bf59521af4abdb427c2ed475198`。本轮没有暂存、提交、推送或部署；整体 diff 包含本轮开始前已有的调度/重试改动，不能将全部差异归为本轮新增。
