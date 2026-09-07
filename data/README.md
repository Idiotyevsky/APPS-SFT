# data/ 目录索引

| 目录 | 用途 | 是否最终交付 |
|---|---|---|
| `raw/apps/` | APPS 官方源数据：`train.jsonl`(107MB) + `test.jsonl`(1.29GB, **封存评测用**)，`manifest.json` 记录 revision 与 SHA256 | 源数据（test 仅供最终评测，合成流程永不读取） |
| `sft_final/` | **正式 SFT 训练集（当前唯一权威交付）**：1481 条，一题一条，id=原始题号 | ✅ 交付 |
| `rl/` | RL 环境数据：`public/`(题面) + `private/`(tests/reference 仅 reward) | ✅ 交付 |
| `sft_debug_rule_handoff/` | 早期 13 条调试样例（旧 id 风格，已被 `sft_final` 取代，仅作参考） | ⚠️ 历史 |
| `eval/` | **自有固定评测集 750 条**（`toolapps_eval.jsonl` + manifest）：seed42、每难度≤250，流程对齐 RLEF-Code 规程（数据集源不同，不与 RLEF 713 逐题对比）；`results/` 存模型评测报告 | ✅ 评测 |
| `.cache/` | 内部缓存：`rule_sft_pool.jsonl`(curate 池)、`rule_handoff_pool.json`(调试池) | 内部，勿分发 |

## sft_final/（正式集）文件说明
- `episodes.jsonl` / `metadata.jsonl` / `sft_messages.jsonl`：各 1481 行，一一对应；
  `sft_messages.jsonl` 为训练直接使用的格式（含 `tools` + `messages` + `trainable`）。
- `sft_tokenized.jsonl`：Qwen 词表渲染的 `input_ids / attention_mask / labels / spans / supervised_text`（1481 行，均 ≤32768 tokens）。
- `qa_report.{json,md}`：分片并行离线 QA，0 issues；`dataset_manifest.json`：分布与配置。
- `SYNTHESIS_SHOWCASE.md`：自动抽取的样例展示。
- `cleaned/problems.jsonl` + `splits.json` + `source_manifest.json`：离线重放/审计所需内部材料。
- 属性：`label_method=rule_design`、无模型/API、无反事实声明；来源=全量 APPS train 中通过验证的题。

## 再生成命令（规则化）
```bash
# 正式集（data/sft_final；先 curate 生成 .cache/rule_sft_pool.jsonl 再 assemble）
PYTHONPATH=src python3 scripts/build_rule_sft_formal.py curate --workers 96 --caps 850,650,160 --pool all
PYTHONPATH=src python3 scripts/build_rule_sft_formal.py assemble --target 1500
# 终检/收尾（QA+剔除+tokenize+showcase）
PYTHONPATH=src python3 scripts/finalize_sft_final.py

# RL 目录（data/rl）
PYTHONPATH=src python3 scripts/build_rl_data.py

# 旧 13 条调试样例（data/sft_debug_rule_handoff；建议改用 sft_final）
PYTHONPATH=src python3 scripts/build_rule_sft_handoff.py curate
PYTHONPATH=src python3 scripts/build_rule_sft_handoff.py assemble

# 评测集（data/eval）
PYTHONPATH=src python3 scripts/build_toolapps_eval.py     # 生成 750 评测集
PYTHONPATH=src python3 scripts/qa_eval.py                # 评测集结构 QA

# 模型评测（在 GPU 机执行；根 README「评测与结果」节有基线记录）
bash scripts/eval.sh
```

> 提示：若要新起一轮正式集，请先清空 `data/sft_final/`（或换 output 路径），避免与旧产物混淆。
