# omp-llm-verifier 中文说明

## 引用

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 参考实现](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

插件采用论文的 20 级 token logprob 评分和 Probabilistic Pivot Tournament。每次普通 OMP 请求会生成 3 个完整候选，使用当前 OMP 默认模型进行验证，并回放胜出响应的文本、推理、图片、工具调用和终止状态。

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

验证器模型必须能提供 OpenAI Chat Completions 或 Responses 的 token logprobs——论文的细粒度奖励是从评分 token 分布上读取的，没有 logprobs 的模型无法运行论文的验证器。当解析出的验证器模型不能提供 logprobs 时，插件会拒绝包装会话模型并显示能力警告，而不是让每次请求静默退化。

已验证的比较结果会缓存在项目根目录的 `.omp-llm-verifier-cache.json`，缓存键覆盖任务、两个候选、评分标准、模型和 prompt 版本的内容指纹，因此相同内容的重复验证不消耗验证器 token。该文件可以直接删除，建议加入 gitignore。

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
