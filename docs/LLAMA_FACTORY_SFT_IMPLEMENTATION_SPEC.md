# Coding Agent：基于 LLaMA-Factory 的 SFT 技术方案

本文面向已有 Coding Agent 合成数据，说明如何完成格式转换、监督位置检查、LoRA 训练、执行评测，以及向 veRL 交付初始模型。

基座沿用 `Qwen/Qwen2.5-Coder-7B-Instruct`。方案于 **2026-09-06** 核对实际数据和 LLaMA-Factory 官方源码。本文的转换函数已在两份实际文件上做内存验证：分别生成 659、20 个目标，目标内容与历史前缀保持完整，masked assistant 未被选为目标。尚未进行模型 tokenization 或 GPU 训练。代码与配置中的 Linux 路径是部署示例，配套脚本按本文接口实现。

## 1. 推荐方案与实施顺序

采用 **ShareGPT 工具格式 + 每个可训练动作一条前缀样本 + `mask_history: true` + LoRA SFT**。

这样，每条训练样本保留目标动作之前的全部历史，只监督最后一个动作。原始数据中 `trainable=false` 的错误提交可以继续作为历史出现，但永远不会被选成训练目标。这条路径适用于当前数据及已出现的多轮修复数据，不依赖修改 LLaMA-Factory 的训练器。

| 步骤 | 工作 | 产物或通过条件 |
|---|---|---|
| 1 | 固定输入文件、基座与框架版本 | 数据哈希、版本清单 |
| 2 | 审计轨迹并按题目划分 train/dev | 题目级 split manifest，无交叉 |
| 3 | 转为 ShareGPT 前缀样本 | train/dev JSON、dataset_info.json、样本映射 |
| 4 | 使用实际模板检查长度和 labels | 每条只监督目标动作，无截断、无错误历史监督 |
| 5 | 小样本训练验证 | loss 有限、adapter 更新、工具输出可解析 |
| 6 | 正式 SFT 与开发集评测 | checkpoint、训练曲线、执行指标 |
| 7 | 导出并交付 RL | 合并 SFT 模型、tokenizer、协议及题目排除清单 |

SFT 阶段不训练 critic，也不使用 GRPO/PPO reward。训练时读取已记录的工具观察；在线执行器用于数据复核和模型评测。

## 2. 已有数据的实际情况

### 2.1 主数据：383 条轨迹

已读取完整文件：

```text
D:/研究生阶段/研二/coding-agentic-rl/sft_messages.jsonl
```

| 项目 | 实测值 |
|---|---:|
| 原始轨迹 | 383 |
| 不重复记录 ID | 383 |
| 抽取的不同题面文本 | 383 |
| 可训练 assistant 消息 | 659 |
| `trainable=false` 的 assistant 消息 | 0 |
| `run_candidate → submit` 轨迹 | 276，72.1% |
| 仅 `submit` 的轨迹 | 107，27.9% |
| `run_candidate` 目标 | 276 |
| `submit` 目标 | 383 |
| user 中带 `Current candidate` | 323 |
| user 中带 `Previous submit result` | 263 |
| 函数调用 / stdin 模式 | 248 / 135 |
| 最后消息为 tool | 383 |
| 最后工具记录标记 accepted | 383 |
| 最后 grader 的 `total=1` | 163 |

所有 assistant 都只包含工具调用，没有单独的自然语言或候选代码声明；所有消息都有 `trainable` 字段。每次 assistant 只调用一个工具。

文件 SHA-256：

```text
088039E5D79A90FAF056A910E7C1986E98081AAC65AD3FA9086CF2336C50C02A
```

这里的 accepted 是已有日志标记，不是本次重新执行代码后的结论。不同题面文本也不等于已经完成近重复或 APPS 来源审计。

### 2.2 同目录另有一份多轮数据

`sft_messagesv2.jsonl` 有 13 条轨迹、21 个 assistant 消息，其中 20 个可训练、1 个不可训练。ID 中包含 9 个不同的 APPS 题号。

其中一条真实轨迹为：

```text
apps-1604-post_submit_failure_replay-11af9ed0d798

run_candidate  trainable=true
submit         trainable=false
run_candidate  trainable=true
submit         trainable=true
```

这证明逐消息 mask 已是实际需求。该文件不因名字含 v2 就自动替换主数据，也不默认与主数据合并；首先检查两文件的题目重叠、来源与质量，再通过输入清单显式选择。本文两份文件共用同一种转换方法。

### 2.3 这些数据能教会什么

直接监督：工具名与参数格式、自主给出 run 输入、读取执行结果后提交完整代码、在给定错误候选和反馈时修复。

没有直接覆盖：assistant 声明 `<candidate>...</candidate>`、从纯题目出发先生成候选再 run 的完整协议、独立的解释性回复。不能因为 SFT loss 降低，就认定模型已经掌握这些未出现在目标中的行为。进入 RL 前，应分别检查已有候选和纯题目两种初始状态。

## 3. 为什么原始 JSONL 不能直接喂给框架

原数据采用自定义消息结构：

```json
{
  "role": "assistant",
  "tool_calls": [
    {"name": "run_candidate", "arguments": {"input": "..."}}
  ],
  "trainable": true
}
```

主要适配点有三个：

1. 这里的工具调用是扁平 `name/arguments`，并不是标准 OpenAI 的 `tool_calls[].function` 结构。
2. 当前核对的 ShareGPT converter 不读取自定义 `trainable` 字段。仅保留字段，不能让它生效。
3. 原始轨迹以 tool 结束，而 ShareGPT 监督样本应以 assistant/function 动作结束。最后一次 submit 的反馈用于审计，不是下一 token 的监督目标。

因此显式转换为 ShareGPT，不依赖格式名称自动猜测。[LLaMA-Factory converter](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/converter.py)

### 3.1 字段映射

| 原字段 | ShareGPT 字段 | 处理 |
|---|---|---|
| system.content | 顶层 `system` | 原文保留 |
| tools 数组 | 顶层 `tools` | 序列化为 JSON 字符串 |
| user.content | `from=human, value=原文` | 保留题面、候选和允许的历史反馈 |
| assistant.tool_calls | `from=function_call` | value 为 name/arguments 的 JSON 字符串 |
| tool.content | `from=observation` | 原执行观察作为字符串 |
| assistant.trainable | 转换器控制项 | 决定是否生成一个以该动作结尾的样本 |
| id / metadata_ref | sidecar manifest | 追踪来源，不写入 prompt |

工具 `arguments` 在内部保持对象；代码与输入都是对象中的字符串。应使用 JSON 库序列化，不手工转义换行，也不重复将 arguments 编码成字符串。

### 3.2 按目标动作生成前缀

原始消息为：

```text
system → user → run → run_result → submit → accepted_result
```

输出两条：

```text
样本 1：system → user → [run]
样本 2：system → user → run → run_result → [submit]
```

方括号表示唯一监督目标。样本 2 中先前的 run 仍然可见，但不再次计算 loss。accepted_result 不进入这两条样本，因为它发生在目标之后。

遇到第 2.2 节的多轮数据，输出三个目标：第一次 run、第二次 run、最终 submit。失败 submit 保留在后两个样本的历史中，但不为它生成目标样本。

未过滤、未划分时，主文件应产生 **659 条前缀样本**；单独转换 v2 应产生 **20 条**。这些数是格式转换的结构预期，不是声称长度检查和质量复核后仍全部可用。

### 3.3 转换核心代码

下列函数处理当前“单工具调用、无 assistant 正文”的格式。混合正文、并行调用或其他协议应明确报错后扩展，不能丢弃内容继续训练。题目划分与质量审计先于该函数执行。

```python
import copy
import json

def export_prefix_samples(record):
    messages = record["messages"]
    if not messages or messages[0]["role"] != "system":
        raise ValueError("expected leading system message")
    system = messages[0]["content"]
    tools = json.dumps(record["tools"], ensure_ascii=False)
    history, samples = [], []

    for index, message in enumerate(messages[1:], start=1):
        role = message["role"]
        if type(message.get("trainable")) is not bool:
            raise ValueError("missing boolean trainable")
        if role in {"user", "tool"}:
            if message["trainable"]:
                raise ValueError("context must not be trainable")
            converted = {
                "from": "human" if role == "user" else "observation",
                "value": message["content"],
            }
        elif role == "assistant":
            calls = message.get("tool_calls", [])
            if message.get("content") or len(calls) != 1:
                raise ValueError("unsupported assistant shape")
            call = calls[0]
            name, arguments = call["name"], call["arguments"]
            key = {"run_candidate": "input", "submit": "code"}.get(name)
            if key is None or not isinstance(arguments, dict):
                raise ValueError("invalid tool")
            if set(arguments) != {key} or not isinstance(arguments[key], str):
                raise ValueError("invalid tool arguments")
            converted = {
                "from": "function_call",
                "value": json.dumps(
                    {"name": name, "arguments": arguments}, ensure_ascii=False
                ),
            }
        else:
            raise ValueError(f"unsupported role: {role}")

        expected = {"human", "observation"} if len(history) % 2 == 0 else {"function_call"}
        if converted["from"] not in expected:
            raise ValueError("invalid role sequence")
        history.append(converted)

        if role == "assistant" and message["trainable"]:
            samples.append({
                "sample_id": f"{record['id']}:m{index}",
                "conversations": copy.deepcopy(history),
                "system": system,
                "tools": tools,
            })
    return samples
```

上游还需检查 system 的 `trainable=false`、tool 名与前一调用匹配、submit 非空、合法工具 schema，以及原始最终提交状态。不能以转换函数成功代替轨迹质量验证。

### 3.4 转换示例的可读外形

下面仅缩略长题面和 schema；转换产物必须写入完整内容。`function_call.value` 是一个 JSON 字符串，模板再把它转为模型输出格式。

```json
{
  "sample_id": "101:m4",
  "system": "<原 system 全文>",
  "tools": "<两个完整工具 schema 的 JSON 字符串>",
  "conversations": [
    {"from": "human", "value": "<题面、原 candidate 和 Previous submit result 全文>"},
    {"from": "function_call", "value": "{\"name\":\"run_candidate\",\"arguments\":{\"input\":\"<原始输入>\"}}"},
    {"from": "observation", "value": "<原始 run_result JSON>"},
    {"from": "function_call", "value": "{\"name\":\"submit\",\"arguments\":{\"code\":\"<原始完整正确代码>\"}}"}
  ]
}
```

这里的 `101:m4` 对应已读取的真实第 101 号记录中的 submit 位置；尖括号部分仅用于文档展示，不可写入训练集。

## 4. Loss mask 的具体实现

### 4.1 使用原生末轮监督

```yaml
train_on_prompt: false
mask_history: true
packing: false
```

当前 supervised processor 会将 prompt 部分置为 `-100`，并在开启 `mask_history` 时只监督最后一轮 target。我们通过前缀转换，使最后一轮始终恰好是原始 `trainable=true` 的目标动作。[监督预处理源码](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/processor/supervised.py)

**不能对完整原轨迹只设置 `mask_history=true`。** 这样会只学习最终 submit，漏掉前面的 run。也不能在前缀样本上关闭 `mask_history`，否则先前动作会被重复监督，失败历史也可能参与 loss。

| 内容 | attention_mask | labels |
|---|---:|---|
| system、tools、题目、错误 candidate | 1 | -100 |
| 目标之前的所有 assistant/function | 1 | -100 |
| 所有 tool/observation | 1 | -100 |
| 最后目标的工具名、参数、代码与结束标记 | 1 | 对应 token ID |
| padding | 0 | -100 |

不需要手工将 labels 平移一位；因果语言模型的训练实现负责 next-token shift。目标第一个 token 由前缀末端预测。

### 4.2 优化目标

令 m 为最终 labels 中的有效监督位置，训练最小化交叉熵：

$$
\mathcal L_{SFT}=-\frac{\sum_{i,t}m_{i,t}\log\pi_\theta(x_{i,t}\mid x_{i,<t})}
{\sum_{i,t}m_{i,t}}.
$$

每个原始可训练动作只成为一次目标；前缀复制增加的是上下文计算量。训练时的 batch 切分和梯度累积仍会影响样本加权，不能声称拆分前后优化过程逐步完全相同。固定框架的归一化行为并记录有效监督 token 数；首轮采用原生交叉熵，不加入工具类别权重。

run 参数通常比 submit 代码短，即使两种动作记录数接近，submit 仍可能贡献更多监督 token。报告两种目标各自的 token 数和验证 loss，再决定是否需要单独的采样实验。

## 5. Qwen 模板与后续 RL 协议

使用 `template: qwen`、`tool_format: qwen`。Qwen2.5-Coder 不套用 Qwen3 的 thinking 模板。所核对模板会将函数动作编码为 assistant 输出，将工具结果包装在 user 段的 `<tool_response>` 中。[Qwen 模板](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py)

工具动作的可读形式为：

```text
<|im_start|>assistant
<tool_call>
{"name": "run_candidate", "arguments": {"input": "..."}}
</tool_call><|im_end|>
```

工具参数对象由 formatter 序列化，训练数据中不用预先手写 `<tool_call>` 标签。[Qwen 工具格式化实现](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/tool_utils.py)

执行评测和 veRL 必须复用经过核对的相同渲染协议。仅仅使用同一个 tokenizer 文件名，不保证 system 中 tools 的排列、空格、观察角色和结束符相同。保存三个固定前缀作为一致性样例：纯题目、已有候选、已有 run 结果；比较训练端与推理端的完整 token IDs。

实际 JSONL 中 submit 反馈还含 `status/passed/total`，RL 方案的最小反馈只含 `pass_rate/failing_input`。实施时选定共同的反馈投影函数，并记录版本；若统一为最小格式，同时处理 tool 消息和 user 中嵌入的历史反馈，保留原始日志供审核。不能只改一处，也不能依靠宽泛文本替换误删题面内容。

本轮 SFT 默认保留已有动作监督。若后续要加入 candidate 文本声明，先明确新增格式及监督样本；当前 659 个目标里没有这种声明，不能把“写进 system prompt”视为已经监督训练了该能力。

## 6. 数据划分与长度检查

### 6.1 先划分题目，再展开动作

使用已有 APPS problem manifest 为题目提供稳定键。缺失时先以完整题面哈希辅助回溯，结合来源复核；不能只把记录 ID 当题号。

按题目将约 90% 用于 SFT train、10% 用于 SFT dev，seed=42。主文件若最终确认 383 个独立可用题组，可计划 345/38 题；这是划分目标，不是本次已生成的实际 split。不同输入文件中同一题、多 bug 版本和所有前缀样本必须进入同一个集合。

不要对展开后的 659 行设置 `val_size: 0.1` 随机分割：同一题的 run 和 submit 很容易落到两边。先生成两个文件，再用 `dataset` 与 `eval_dataset` 显式加载。

SFT dev 题不得在后续 RL train 中使用。交付 `sft_train_problem_keys`、`sft_dev_problem_keys` 和二者并集，RL 从其训练题池排除这一并集。难度分层需要 APPS 元数据；当前文件的顶层字段不足以完整恢复难度，不猜测比例。

### 6.2 长度用真实模板计算

初始 `cutoff_len=8192`，统计每条转换样本的总 token 与目标 token 的 P50/P95/P99/max。此处 8192 包含题面、历史、tools 和目标代码，含义与 RL 的“模型动作预算”不同。

必须使用 LLaMA-Factory 的实际 template/processor，不只对原始 `content` 单独 tokenize。`mask_history=true` 在长度不足时可能优先保留末轮并舍弃早期历史，因此先检查未截断长度，再进入正式预处理。

超长时优先提高统一 cutoff 并确认显存；确实无法容纳则过滤并报告对应题目/目标。不要截断 submit 代码、JSON 参数或原题，也不要删除 mask 的历史后假定任务仍相同。首轮关闭 packing，避免同时引入跨样本注意力和位置问题。

对最终 labels 做全量检查：只有最后目标 span 有效、目标内容完整、所有历史为 -100、至少一个有效 label。抽样解码人工核对不能替代数量和边界断言。

## 7. 注册数据集

建议文件结构：

```text
project/
  data/coding_sft/
    dataset_info.json
    train.json
    dev.json
    smoke.json
    split_manifest.json
    sample_manifest.jsonl
  configs/
    sft_lora.yaml
    sft_smoke.yaml
    export_sft.yaml
  scripts/
    prepare_sft.py
    audit_sft_labels.py
    evaluate_sft_agent.py
  outputs/sft/
  exports/sft_merged/
```

上述是待实现的训练项目结构。`train.json/dev.json/smoke.json` 使用 JSON 数组；每个元素为第 3 节的 ShareGPT 样本。原始文件继续保留 JSONL，不覆盖。

`dataset_info.json`：

```json
{
  "coding_agent_train": {
    "file_name": "train.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "observation_tag": "observation", "function_tag": "function_call"}
  },
  "coding_agent_dev": {
    "file_name": "dev.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "observation_tag": "observation", "function_tag": "function_call"}
  },
  "coding_agent_smoke": {
    "file_name": "smoke.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "observation_tag": "observation", "function_tag": "function_call"}
  }
}
```

`sample_manifest.jsonl` 至少含：sample_id、source_file_hash、source_record_id、problem_key、target_message_index、target_tool、split、完整长度、监督长度、原始 target 哈希和反馈投影版本。先保存这个映射，框架预处理可能移除非训练字段。

## 8. 环境与 LoRA 配置

### 8.1 安装与版本固定

在 Linux GPU 环境中，为 SFT 建立独立环境。Python 可从 3.11 起步，但最终采用所选框架提交支持的依赖组合。先安装匹配驱动的 PyTorch，再按官方说明安装 LLaMA-Factory；FlashAttention 非必需，首轮可使用 SDPA。[官方安装说明](https://llamafactory.readthedocs.io/en/latest/getting_started/installation.html)

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout <选定的完整提交SHA>
pip install -e ".[torch,metrics]"
llamafactory-cli version
```

这里的 SHA 必须替换；`main`/latest 不作为实验版本。记录框架 SHA、Python/Torch/Transformers/PEFT/Accelerate 版本、CUDA、模型 revision、tokenizer 哈希、数据哈希和解析后的完整配置。SFT 与 RL 可以使用不同 Python 环境，通过标准模型目录交接。

### 8.2 主配置：`configs/sft_lora.yaml`

这是本项目的训练起点，不代表已经找到最优超参数。路径按项目实际部署调整；dataset 文件和 labels 审计通过后，使用标准 CLI 启动。

```yaml
model_name_or_path: /models/Qwen2.5-Coder-7B-Instruct
trust_remote_code: false
stage: sft
do_train: true
do_eval: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 32
lora_dropout: 0.0
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

dataset_dir: /project/data/coding_sft
dataset: coding_agent_train
eval_dataset: coding_agent_dev
template: qwen
tool_format: qwen
cutoff_len: 8192
train_on_prompt: false
mask_history: true
packing: false
overwrite_cache: true
preprocessing_num_workers: 4
dataloader_num_workers: 2

output_dir: /project/outputs/sft/run_001
overwrite_output_dir: false
logging_steps: 1
save_strategy: epoch
eval_strategy: epoch
save_total_limit: 3
save_only_model: false
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
plot_loss: true
report_to: none

per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.0
max_grad_norm: 1.0
optim: adamw_torch
bf16: true
fp16: false
flash_attn: sdpa
disable_gradient_checkpointing: false
ddp_find_unused_parameters: false
seed: 42
data_seed: 42
resume_from_checkpoint: null
```

LoRA 参数名按官方 finetuning arguments 核对。初始化日志必须列出匹配的模块和 trainable 参数数目；不能只确认 `finetuning_type=lora`。[LoRA 参数源码](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/finetuning_args.py)

有效 batch 为 `GPU数 × 每卡batch × 梯度累积步数`。上面单卡对应 16 条前缀样本；4 卡若保持有效 batch=16，则将累积改为 4。按实际展开后的 train 样本数估算更新量，不能仍用 383 条原始轨迹计算。

当前数据较小，先训练 3 epochs 并逐 epoch 比较。学习率可在 `2e-5 / 5e-5 / 1e-4` 中做小范围开发集选择。没有提升时先检查任务覆盖和监督位置，再增加轮数；不因 train loss 很低而认定训练成功。

BF16 要求硬件支持。显存不足时先保持 batch=1、开启 gradient checkpointing，再考虑多卡切分或 QLoRA；不能靠截坏代码节约显存。GPU 型号与可用显存尚未给出，本文不承诺具体单卡容量。

### 8.3 QLoRA 备选

若基座权重显存成为主要瓶颈，可另建实验配置，增加：

```yaml
quantization_bit: 4
quantization_method: bitsandbytes
quantization_type: nf4
double_quantization: true
```

保留同样的数据和监督规则，安装并验证相容的 bitsandbytes。4-bit 减少底座权重内存，但不消除长序列 activation。不要同时修改长度、数据和量化后把结果只归因于 QLoRA。

向 veRL 交付时重新加载原始未量化基座并合并 SFT adapter，单独验证数值与执行结果；不能把训练时量化底座直接视为标准 BF16 SFT 模型。

## 9. 运行流程与配套脚本

### 9.1 数据与 labels 审计

需要实现的项目命令：

```bash
python scripts/prepare_sft.py \
  --input /data/raw/sft_messages.jsonl \
  --problem-manifest /data/raw/problem_manifest.json \
  --split-mode problem --dev-ratio 0.1 --seed 42 \
  --export-mode target_prefix --output-dir data/coding_sft

python scripts/audit_sft_labels.py \
  --config configs/sft_lora.yaml \
  --report-dir outputs/sft/data_audit
```

`prepare_sft.py` 负责 schema/题目分组/前缀转换/映射保存。默认不读取 v2；需要使用时通过多个 `--input` 显式指定，并重新执行题目去重。没有 problem manifest 时，应支持回溯报告，但不能给出已经完成 APPS 隔离的假结论。

`audit_sft_labels.py` 使用锁定版本的 converter、template 与 supervised processor；只加载 tokenizer，不需要加载 7B 权重。输出原始/转换/过滤数量、长度分布、按工具的监督 token 数、每个样本的实际 labels 对齐结果。随机展示 10 条解码结果，并固定检查那条含 masked submit 的多轮记录。

当前两份输入都需要校验 code 的原始字符串保持不变、参数 JSON 可逆、所有非目标 labels=-100、最后目标完整、同题 split 不交叉。数据缓存必须绑定文件哈希、模板、cutoff 和 mask 配置；已有 `tokenized_path` 可能使框架直接加载旧缓存并忽略其他数据参数，不要换配置后沿用旧目录。[数据加载与缓存源码](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/loader.py)

### 9.2 小规模训练

从 train 题目中挑选 8—16 条前缀构成 `smoke.json`，覆盖 run、submit、stdin、函数调用；masked 历史另用结构测试覆盖，不从 dev 拿来做梯度更新。

复制主配置为 smoke 配置，至少改为：

```yaml
dataset: coding_agent_smoke
output_dir: /project/outputs/sft/smoke
do_eval: false
eval_strategy: "no"
save_strategy: "no"
load_best_model_at_end: false
gradient_accumulation_steps: 1
max_steps: 20
```

```bash
llamafactory-cli train configs/sft_smoke.yaml
```

观察有效 loss、可训练参数梯度、adapter 权重变化和少量生成结果。20 steps 用于排除接线问题，不作为效果结论。完成后正式训练从原始 Instruct 基座重新开始，不默认承接 smoke 更新。

### 9.3 正式训练与恢复

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/sft_lora.yaml
```

多卡示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 FORCE_TORCHRUN=1 \
  llamafactory-cli train configs/sft_lora.yaml \
  gradient_accumulation_steps=4
```

DDP 每卡仍持有模型副本；需要切分显存时使用选定版本的 DeepSpeed/FSDP 配置，不能把 GPU 数增加等同于自动切分权重。

中断恢复使用 `resume_from_checkpoint=/project/outputs/sft/run_001/checkpoint-<实际步数>`，保留 optimizer/scheduler 和训练状态。仅设置 `adapter_name_or_path` 是从权重继续训练，不等于恢复原训练步数与学习率计划。新实验使用新 output_dir。

## 10. 如何验证 SFT 学会了工具使用

### 10.1 离线指标

记录 train/dev loss、run/submit 各自的监督 token 数与 loss、有效标签比例、梯度范数和长度截断数。

对 dev 前缀进行生成，检查工具名是否合法、JSON 是否可解析、参数键/类型是否正确，以及 submit 是否给出完整代码。工具动作 exact match 只能辅助观察，因为不同代码、不同测试输入都可能有效。

`eval_loss` 最低的 checkpoint 是自动保存候选。最后还需通过执行评测选择供 RL 使用的模型；仅凭文本 loss 或 BLEU/ROUGE 不能判断解题质量。

### 10.2 在线执行指标

用基座与 SFT checkpoint 在同一组 dev 题、相同预算和相同执行器下比较：

| 指标 | 含义 |
|---|---|
| 工具调用有效率 | 能解析且满足两工具 schema 的调用占比 |
| 完整提交率 | 能在预算内输出合法完整 submit 的 episode 占比 |
| 解题成功率 / 平均通过率 | 固定 grader 下最终代码表现 |
| 平均调用次数 | 与成功率一起衡量，不能将提前放弃当作高效率 |
| 执行反馈后修复成功率 | 从失败候选出发是否能利用反馈完成修复 |

评测分两种初始状态报告：与 SFT 相同的“题目+候选/反馈”，以及 RL 将使用的“只有题目”。前者验证已监督的能力，后者检测分布变化。一次 submit accepted 就结束；工具观察由真实执行产生，不继续喂参考轨迹的未来观察。

评测器需复用已确定的 `run_candidate(input)` 与 `submit(code)` 环境，支持 fn_name/类方法及 stdin。LLaMA-Factory 训练命令不会自动运行这两个自定义工具，因此需要 `evaluate_sft_agent.py` 或共用现有 agent runtime。

APPS test 不用于调学习率或选择 SFT checkpoint。主文件有 163 条记录只包含一个 grader 测试项，结构上无法提供细粒度通过率；复核结果时报告覆盖情况，避免将日志 accepted 当作充分正确性证据。

## 11. 导出模型并交给 veRL

先保留原始 adapter 与完整训练 checkpoint，再合并一个用于 RL 初始化的完整模型目录。

`configs/export_sft.yaml`：

```yaml
model_name_or_path: /models/Qwen2.5-Coder-7B-Instruct
adapter_name_or_path: /project/outputs/sft/run_001/checkpoint-<选定步数>
template: qwen
tool_format: qwen
finetuning_type: lora
export_dir: /project/exports/sft_merged
export_size: 5
export_device: cpu
export_legacy_format: false
```

```bash
llamafactory-cli export configs/export_sft.yaml
```

路径中的 checkpoint 必须替换为实际选定目录。合并时加载未量化基座，不启用 `quantization_bit` 或导出量化。导出参数遵循官方说明。[保存与 LoRA 合并](https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html)

比较合并前“基座+adapter”与合并后模型在固定前缀上的 logprob 和生成行为；允许明确记录的浮点差异。检查导出目录包含完整权重、config、tokenizer 文件，并在 veRL 环境实际加载一次。

交付清单：

```text
sft_merged/                       完整 SFT 权重与 tokenizer
sft_adapter/                      可独立加载的 SFT adapter
sft_training_checkpoint/          含 optimizer 等恢复状态
sft_train_problem_keys.json
sft_dev_problem_keys.json
sft_all_problem_keys.json         供 RL train 排除
protocol_config.json              两工具、反馈投影、模板与停止规则
template_fixtures.json            三个固定前缀及 token IDs
environment.lock.json
data_audit.json
sft_eval_report.json
```

veRL actor 和固定 reference 都以 `sft_merged` 为起点，RL 新建自己的 adapter。关闭这个新 adapter 才应回到 SFT policy；不能把关闭 SFT adapter 后的原始基座误作 reference。

## 12. 验收标准与常见问题

| 现象 | 优先检查 |
|---|---|
| Cannot find valid samples | function_call 映射、消息交替、末尾是否仍是 tool |
| loss 降低但模型只会 submit | 是否只对完整轨迹设置 mask_history=true，遗漏 run 前缀 |
| 学会复述错误代码或伪造工具结果 | 错误历史/observation 是否被 label 监督 |
| JSON 内出现多余转义 | arguments 是否被重复序列化 |
| 生成只有半段代码 | 目标是否在预处理或生成阶段被截断 |
| dev 指标异常好 | 同题前缀、变体或两份输入是否跨 split |
| SFT 正常、veRL 工具解析失败 | 训练/推理模板、role、结束符与工具格式是否一致 |
| loss 不变且 adapter 不更新 | LoRA 模块匹配、有效 labels、梯度与学习率 |

完成本阶段需要同时满足：

1. 数据版本与实际统计可复现；未过滤时主文件产生 659 个目标，v2 单独产生 20 个目标。
2. 同题及其所有前缀不跨 train/dev；后续 RL 排除清单已交付。
3. 全部保留样本的目标完整，历史与工具 labels 全为 -100，错误 submit 从未成为 target。
4. 使用真实模板的 token/label 审计通过；训练端与推理端固定前缀一致。
5. 小规模训练确认参数更新；正式训练有可恢复 checkpoint 和执行评测结果。
6. 合并模型可由 veRL 加载，SFT reference 身份明确，已有候选与纯题目状态均经过评测。

本方案保留此前 SFT 的监督原则，并通过前缀转换把逐消息 `trainable` 精确落实到框架支持的末轮监督。新增工作主要是数据适配、labels 审计和执行评测；SFT 优化本身使用 LLaMA-Factory 的标准流程。
