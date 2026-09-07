# SFT 评测调试报告（截至 2026-09-06 晚）

## 1. 已确证结论
1. **run002（3ep / lr5e-5 / 全模块 LoRA）存在生成退化**：
   - vLLM agent 冒烟（30 题口径、cap2048）跑到 7/30，全部 `status=truncated solved=0`（见 sft/outputs/agent_smoke30_v5.log）。
   - 失败输出特征：开头是正确 submit JSON，随后无限重复 `\n`，JSON 永不闭合。
   - 单轮 codegen 评测 750 题：SFT merged=20/750，基座=135/750。
2. **exp2（1ep / lr2e-5 / 仅 q,k,v,o / dropout0.05）修复退化**：
   - HF 单题探针：输出干净完整 JSON 并闭合（490-515 字符），无重复换行。
   - 但存在“写错语言/写错解”的能力问题（如用 C++ 作答），推测纯题目直提监督样本不足（~393/2126）。
3. **训练与评测渲染一致性**：
   - system 文本一致；`# Tools` 工具块注入一致；
   - 前缀 token 级比对：LlamaFactory(训练) 与 HF apply_chat_template(评测) **逐 token 一致**；
   - 已删除评测侧附加句 “No candidate exists yet...”，测试与文档同步更新（tests/test_agent_eval.py 22 项通过）。
4. **合并/导出正确**：基座+adapter 与 merged 行为一致；层抽样合并误差 0；词表/模板文件已对齐。

## 2. 未完成：Agent 式多轮评测批量结果
- 原因：批量推理（vLLM 与 HF）在多台机器上随机挂起：
  - A6000-6（本机）与 2080ti-5（210.28.132.174，GPU2）都出现“首题 generate 后 CPU 自旋、GPU 0%”；
  - vLLM 表现为“引擎就绪后无任何进度”；
  - 单题探针（含 eager attention）可正常完成（10s 级），说明脚本/模型/评测逻辑本身能跑通；
  - 卡死位置收窄到“解析/沙箱/后续轮次”附近，但尚未稳定复现定位（疑似多租户抢占 + 共享环境问题）。
- 相关日志：sft/outputs/agent_smoke30_v5.log、agent_exp2_9.log、hf_smoke3.log、longgen_probe.log。

## 3. 建议下一步（需干净/空闲 GPU）
1. 在确认无抢占的 GPU 上，用 exp2 merged 跑 3→30→750 题 HF（eager）agent 评测；
2. 同时用 faulthandler/分段日志定位批量挂起（若复现）；
3. 依据结果决定是否补“纯题目直提”监督数据并重训。
