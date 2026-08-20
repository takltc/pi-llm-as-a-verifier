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

`enabled` 控制自动验证，`candidateCount` 控制候选数量，范围为 2-8，默认值为 3。配置在新 OMP 会话中生效。验证器自动继承 OMP `modelRoles.default` 的模型、推理等级、凭据、请求头和兼容配置。

仅关闭自动验证并保留插件安装状态：

```bash
omp plugin config set omp-llm-verifier enabled false
```

停用或恢复插件加载：

```bash
omp plugin disable omp-llm-verifier
omp plugin enable omp-llm-verifier
```

当前默认模型需要提供 OpenAI Chat Completions 或 Responses 的 token logprobs。模型缺少 token logprobs 时，插件会在 OMP 中显示能力警告。

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
