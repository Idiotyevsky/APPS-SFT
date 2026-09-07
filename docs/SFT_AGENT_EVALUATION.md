# SFT Agent 评测（tool-agent-v2）

默认入口 `scripts/eval.sh` 现在运行真实多轮工具评测。旧的单次代码生成基线需显式传 `--protocol code`，两种成绩不能混为 pass@1。

## 生成与调用预算

本地 Qwen2.5-Coder-7B-Instruct 和合并 SFT 模型配置均为 32768 上下文。脚本读取模型配置，不再写死 8192；不隐式启用 RoPE 扩展。

每轮使用同一份 token IDs 计算预算并送入 vLLM：

```text
本轮 max_new_tokens = min(模型窗口 - 完整历史 token 数, 剩余总生成预算, 可选单轮上限)
```

默认单轮上限为空，即使用全部剩余上下文；默认总生成预算为 `max_actions * max_model_len`。模型可以正常提前输出 EOS，无须生成到上限。完整历史不截断；窗口用尽记为 `context_exhausted`。`finish_reason=length` 记为 `truncated`，该轮不执行工具，即便截断前的文字看起来像完整 JSON。

默认最多 4 个动作、3 次 submit、3 次 run。三个预算独立计数，候选初始化的 grader 验证不算模型调用。可用 `--max-actions`、`--max-submits`、`--max-runs` 和 `--max-total-new-tokens` 显式调整。固定预算后，基座与 SFT 使用相同参数比较。

## 三种初始状态

- `--mode problem-only`（默认）：仅题目，无候选；user 内容与训练 problem-only 状态逐字一致（无附加提示），失败后可 run 和修复。
- `--mode candidate`：题目和固定错误候选，无 grader 反馈，评测主动验证。
- `--mode repair`：题目、固定错误候选及真实 grader 的公开失败反馈，评测修复。

后两种模式必须提供 `--candidates fixed_candidates.jsonl`，每个选中题目一条，字段为 `id`、`code`、`eval_row_sha256`。摘要由 `synthesis.agent_eval.digest(eval_row)` 对原评测行计算，用来防止 APPS train/test 题号重名导致错配。候选必须来自对应独立评测题，基座和 SFT 共享同一文件；不要混入训练集候选。脚本不会自动生成或替换候选，每题启动时使用相同真实 grader 验证候选未通过；若已经 accepted，则记为 `candidate_not_wrong` 并使整次命令返回非零。验证结果只在 repair 模式送入模型，candidate 模式不泄露。

目前未随本修复生成新的错误候选集；纯题目评测可直接使用现有 `data/eval/toolapps_eval.jsonl`。

## 小规模对照

显式选择空闲 GPU；脚本不会自动抢占 GPU。`EVAL_PYTHON` 默认指向已有 opd 环境，亦可指定安装了 vLLM/transformers 的环境。

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval.sh \
  --model /home/zfs02/model/Qwen2.5-Coder-7B-Instruct \
  --per-difficulty 10 --save-completions \
  --output-dir data/eval/results/agent/base_30_v2

CUDA_VISIBLE_DEVICES=0 scripts/eval.sh \
  --model sft/exports/sft_merged \
  --per-difficulty 10 --save-completions \
  --output-dir data/eval/results/agent/sft_30_v2
```

这会从每个难度确定性地各取 10 题，而不是前 30 题。全量评测去掉 `--per-difficulty 10`。可用 `--tensor-parallel-size 2` 配合两张可见 GPU。不要为缩小显存压力悄悄降低上下文；若资源不足，先明确选择更多 GPU 或另设一个标明窗口限制的对照。

## 工具协议与错误

每轮只允许一个 JSON 对象，可直接输出，或由一对 `<tool_call>` 标签包裹。不接受自然语言前后缀、重复 JSON 字段、多个调用、多余参数或模型自造的工具观察。工具名只能是 `submit` / `run_candidate`。执行结果必须来自真实 grader，并以结构化 assistant tool_calls 和 tool 消息进入下一轮。

解析失败、输出截断、上下文耗尽、调用预算耗尽、运行基础设施异常分别记录。`NoCandidate` 会作为真实工具错误送回模型并独立计数。题目级异常不会删除已经完成的记录；启动阶段模型加载失败仍然直接报错。当前执行器沿用 `backend=local`，3 秒单次执行限制、512 MB 设置；本修复没有改变 grader 的隔离实现。

## 产物、续跑与指标

每个输出目录包含：

- `run_manifest.json`：协议、完整模型权重/tokenizer 哈希、评测文件和候选文件哈希、选择的题号、预算、软件版本和评测源码哈希。首次及续跑都读取模型文件计算摘要，可能需要额外 I/O 时间。
- `progress.jsonl`：每题立即追加并 fsync，保留终止原因、每轮 token 数、工具观察和计数。`--save-completions` 额外保留每轮完整原始输出和解析动作。
- `results.json`：每题完成后原子更新的汇总及逐题记录；明确 `complete` / `expected_total`，中间结果不会伪称全量完成。

续跑使用完全相同命令加 `--resume`。不允许模型、数据、候选、预算或代码变更后混入旧成绩；变更需新输出目录。仅恢复写入时被截断的最后一行，完整行损坏或重复题号直接报错。每个输出目录只应有一个写进程。已记录的基础设施错误同样视为完成，不自动重试并挑选更好结果；需要重测时建立新的一次评测。

主要指标为 `budgeted_agent_success_rate`，严格要求一次 submit 的全部测试通过。额外报告首次提交成功率（分母为已完成题目，未提交记未成功）、修复成功率（分母为发生失败提交或初始化错误候选的题目）、工具调用格式正确率、平均调用次数、NoCandidate/工具超时次数，以及各终止原因和难度分组。基础设施异常和模型能力失败分列，整体成功率仍保留全部已完成题目的固定分母。

该修复只验证评测接线，不预先保证模型解题效果。基座和 SFT 应分别完成相同模式、同一题集、同一调用预算的实际推理后再比较。
