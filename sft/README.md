# ToolAPPS SFT 实施记录（uv + LLaMA-Factory + wandb）

> 依据 `docs/LLAMA_FACTORY_SFT_IMPLEMENTATION_SPEC.md` 实施；本记录以
> 2026-09-06 实际跑通的结果为准。文档中的“383 轨迹/659 目标”是旧数据口径，
> 现以 `data/sft_final` 1481 条终版为准。

## 1. 数据口径（阶段 0）
- 修复一致性：`episodes.jsonl / sft_messages.jsonl / metadata.jsonl` 均 **1481 行** 一一对应（移除残留行 2835）。
- 目标动作：**2371 个** = `submit` 1481 + `run_candidate` 890；含多目标（run→submit）轨迹 890 行。
- 一题一条，题目 key = 原始题号，可直接做题目级划分。

## 2. 环境（uv，A6000-6/本机）
- uv venv（NFS 共享，A6000-3 也可用）：`sft-venv/`（Python 3.12.7）
- LLaMA-Factory **v0.9.5 @ 7af909522a951e3ad9f022ea6f88b6755257eaa5**（`sft/llama-factory/`，editable 安装）
- 关键版本（`sft/requirements.lock`，共 134 包）：
  `torch==2.11.0+cu128`（pytorch cu128 index）、`transformers==5.6.0`、
  `peft==0.18.1`、`accelerate==1.11.0`、`datasets==4.0.0`、`wandb==0.29.0`
- 兼容性修复记录：torchvision==0.26.0+cu128、torchaudio==2.11.0+cu128 必须与 torch cu128 同源安装，否则 `_C`/`libcudart` 符号错误。

## 3. 数据转换（阶段 2）
```bash
python3 sft/scripts/prepare_sft.py \
  --episodes data/sft_final/episodes.jsonl \
  --metadata data/sft_final/metadata.jsonl \
  --out data/coding_sft \
  --extra-episodes data/sft_debug_rule_handoff/episodes.jsonl
```
产出（`data/coding_sft/`）：
| 项目 | 值 |
|---|---:|
| train 题目 / 样本 | 1333 / **2126** |
| dev 题目 / 样本 | 148 / **245** |
| smoke 样本 | 16（run 8 + submit 8） |
| structure 样本（masked 审计用） | 20 |
| 样本总数（train+dev） | 2371 ✓（= 目标数） |

格式：ShareGPT（`conversations/system/tools` 顶层字段）+ `dataset_info.json`（`mask_history: true` 逐样本末轮监督）。

## 4. labels 审计（阶段 3，真实模板）
```bash
./sft-venv/bin/python sft/scripts/audit_sft_labels.py \
  --config sft/configs/sft_lora.yaml --dataset <coding_agent_train|dev|smoke|structure> \
  --report-dir sft/outputs/data_audit
```
全部 **0 问题**：监督 span 连续且在序列末尾、每样本≥1 label、历史与工具观察全 -100、
失败 submit 代码未泄漏进 labels（structure 20 条验证通过）。

长度（真实 Qwen 模板，cutoff 8192 内）：
| 集 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| train (2126) | ~800 | ~1600 | ~2200 | ~2550 |
| dev (245) | 813 | 1583 | 2168 | 2550 |
| smoke (16) | 958 | 2045 | 2045 | 2286 |

按工具监督 token：train 中 submit 贡献显著多于 run（run 均值≈35/样本，submit≈120/样本）。

## 5. Smoke 训练（阶段 4，A6000-6 GPU0）
```bash
CUDA_VISIBLE_DEVICES=0 ./sft-venv/bin/llamafactory-cli train sft/configs/sft_smoke.yaml
```
- 20 steps 完成，train_loss=**1.11**，grad_norm 0.2–2.8（adapter 正常更新），
  20.3s/20 steps；loss 曲线：`sft/outputs/smoke/training_loss.png`
- 接线结论：ShareGPT 前缀样本 + `mask_history:true` + `tool_format:qwen` 可正常训练。
- 正式配置：`sft/configs/sft_lora.yaml`（LoRA r32，`mask_history`，bf16，3 epochs，
  `report_to: wandb`，单卡有效 batch=16）。

## 6. 下一步（待办）
1. **wandb**：提供 `WANDB_API_KEY` + project/entity 后，正式训练时用 `sft_lora.yaml`（`report_to: wandb`）。
2. 正式训练：`CUDA_VISIBLE_DEVICES=<空闲卡> ./sft-venv/bin/llamafactory-cli train sft/configs/sft_lora.yaml`
3. dev 执行评测：`sft/scripts/evaluate_sft_agent.py`（复用两工具环境；或直接用根 `scripts/eval.sh` 对比基座/训练后模型）。
4. 合并导出：`sft/configs/export_sft.yaml`（llamafactory-cli export）→ 交付 veRL（附题目排除清单：`split_manifest.json` 的 train/dev keys 并集）。

## 7. 工具评测修复（tool-agent-v2）

当前入口 `scripts/eval.sh` 默认真正执行多轮工具调用。生成预算默认是模型配置窗口减去完整历史，当前 7B 配置为 32768；不再固定为 4096。三种初始状态、固定候选文件格式、严格单工具解析、逐题保存、续跑和基座/SFT 对照命令见 [评测说明](../docs/SFT_AGENT_EVALUATION.md)。旧单次代码生成需显式选择 `--protocol code`。
