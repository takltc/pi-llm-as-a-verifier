# omp-llm-verifier

## 引用

- [LLM-as-a-Verifier: A General-Purpose Verification Framework](https://arxiv.org/abs/2607.05391)
- [Self-Verification / Terminal-Bench 2.1 参考实现](https://github.com/llm-as-a-verifier/llm-as-a-verifier#self-verification-terminal-bench-21)

插件采用论文的细粒度 20 级 logprob 评分与 Probabilistic Pivot Tournament。请求级默认配置为 3 个候选、1 个 pivot、2 次重复验证、固定 seed 0；候选和验证统一继承 OMP 的 modelRoles.default、推理等级、凭据、请求头与兼容配置。

## OMP 安装与使用

安装并开启插件：

~~~bash
omp plugin install https://github.com/takltc/pi-llm-as-a-verifier.git
omp plugin config set omp-llm-verifier enabled true
omp plugin doctor
~~~

之后照常启动 OMP 并发送任务。插件会在每次普通模型请求中并发生成完整候选，使用同一个 OMP 默认模型验证候选，再把胜出响应的文本、推理、图片、工具调用和终止状态回放给 Agent。Agent 继续按原生 OMP 流程执行胜出的工具调用。
评审上下文包含当前对话、图片数据和完整工具契约；评审服务异常或候选不足时，OMP 会显示 warning 并明确标记本次降级。

插件默认关闭；单次会话可用 --llm-verifier 临时开启。验证器需要当前 OMP 默认模型提供 OpenAI Chat Completions 或 Responses API 的 token logprobs。

## 本地开发

~~~bash
git clone https://github.com/takltc/pi-llm-as-a-verifier.git omp-llm-verifier
cd omp-llm-verifier
bun install --frozen-lockfile
bun test
bun run typecheck
omp plugin link "$PWD"
omp plugin doctor
~~~

本地修改后重新执行 bun test、bun run typecheck 和 omp plugin doctor，再启动 OMP 验证普通请求链路。
