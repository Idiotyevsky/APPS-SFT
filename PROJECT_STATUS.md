# Current Project Status

## Current main line

- Base model: /home/nfs05/model/Qwen2.5-14B-Instruct
- Selected SFT warm start: sft/outputs/protocol_run13_qwen25_14b_light_all_linear/checkpoint-2
- Loading mode: Base + dynamic LoRA; no merged adapter
- Current phase: RL preparation

## SFT result

The 14B protocol-light run used 144 train samples, 24 stratified dev samples,
all-linear LoRA (rank/alpha 32/32), weighted repair loss, and 12 optimizer
steps. Label audits passed with zero problems. The selected checkpoint is the
earliest checkpoint that preserved Base coding behavior while improving the
constrained 30-task gate:

| model | solved | repair | duplicates | truncated |
|---|---:|---:|---:|---:|
| Base | 4/30 | 7.1% | 4 | 2 |
| selected checkpoint-2 | 5/30 | 11.1% | 5 | 1 |

SFT is closed for the 7B and 14B lines. Do not resume hyperparameter tuning.

## Frozen contracts

- Tools: submit(code) and run_candidate(input)
- Grader/environment semantics
- Balanced 30-task evaluation set
- APPS train/dev split and data schemas
- Base coding model and selected dynamic-LoRA initialization

src/synthesis/ is intentionally left in place and should receive only
bug fixes before the first RL pilot. A package migration is deferred until
the RL environment has run successfully.

## RL direction

The next separately authorized experiment is LoRA-GRPO. Initial reward should
prioritize accepted submissions, pass-rate improvement, duplicate-submission
penalty, invalid-action penalty, and action-budget exhaustion penalty.
Calling run_candidate alone is not a reward.

## Do not do

- Do not merge the selected LoRA.
- Do not resume 7B/14B SFT tuning without a new experiment decision.
- Do not modify grader, evaluator protocol, tool contract, or eval split.
- Do not treat raw strict tool-call legality as the primary 14B gate when
  constrained evaluation is the deployment path.
