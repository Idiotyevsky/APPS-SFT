# RL / GRPO

This directory is reserved for the next post-training phase. It is intentionally
separate from the frozen SFT pipeline.

## Initialization

Use the Base model plus the selected dynamic LoRA adapter documented in
PROJECT_STATUS.md. Do not merge the adapter.

## Planned pilot

The first separately authorized pilot is LoRA-GRPO. Rewards should reflect
execution-grounded progress:

- accepted submission;
- positive pass-rate delta;
- duplicate-submission penalty;
- invalid-action penalty;
- action-budget exhaustion penalty.

Calling run_candidate alone is not a reward. The grader, tool contract,
balanced evaluation set, and public/private information boundary remain frozen.

Subdirectories:

rl/configs/    RL experiment configurations
rl/scripts/    training/evaluation entrypoints
rl/rewards/    reward functions
rl/tests/      RL-specific tests
