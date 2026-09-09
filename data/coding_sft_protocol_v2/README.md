# Minimal Protocol SFT warm-up dataset

- 单位：prefix sample（prepare_sft 导出）。目标：让 Qwen2.5-Coder-7B
  稳定学会工具协议后进入 RL，不追求提升 APPS solve rate。
- 样本：train 300 / dev 24；同 problem 至多 2 条。
- 结构：ShareGPT（conversations/system/tools），tools 按 candidate
  是否已存在动态提供（首决策仅 submit）。
- 配比：{'problem_submit': 30, 'first_failure_run': 80, 'first_failure_direct_repair': 55, 'post_run_repair': 90, 'second_failure_run': 20, 'multiround_final_submit': 25}
- 状态：{"second_failure_run": 20, "multiround_final_submit": 25, "first_failure_direct_repair": 55, "post_run_repair": 90, "problem_submit": 30, "first_failure_run": 80}
- 统计见 protocol_warmup_stats.json；token 级审计用 LLaMA-Factory 真实
  template 复核（mask_history=true），详见 README 顶层流程。
