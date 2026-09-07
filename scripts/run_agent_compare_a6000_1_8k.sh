#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zfs02/jiangjr/data_syn
PY=/home/zfs01/jiangjr/envs/opd/bin/python
ENV_BIN=/home/zfs01/jiangjr/envs/opd/bin
BASE_MODEL=/home/zfs02/model/Qwen2.5-Coder-7B-Instruct
SFT_MODEL="$ROOT/sft/exports/sft_exp2_merged"
BASE_OUT="$ROOT/data/eval/results/agent/base_v5_constrained_balanced30_t8k"
SFT_OUT="$ROOT/data/eval/results/agent/sft_exp2_v5_constrained_balanced30_t8k"
BASE_LOG="$ROOT/sft/outputs/base_v5_constrained_balanced30_t8k_a6000-1.log"
SFT_LOG="$ROOT/sft/outputs/sft_exp2_v5_constrained_balanced30_t8k_a6000-1.log"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2
export PATH="$ENV_BIN:$PATH"
export LIBRARY_PATH="$ENV_BIN/../lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENV_BIN/../lib:${LD_LIBRARY_PATH:-}"
if [[ -x "$ENV_BIN/nvcc" ]]; then
  export CUDA_HOME="$(dirname "$ENV_BIN")"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

common=(
  --eval-path "$ROOT/data/eval/toolapps_eval.jsonl"
  --per-difficulty 10
  --max-model-len 32768
  --max-new-tokens 8192
  --max-actions 4
  --max-submits 3
  --max-runs 3
  --gpu-memory-utilization 0.9
  --tensor-parallel-size 1
  --save-completions
  --constrain-actions
)

"$PY" sft/scripts/evaluate_sft_agent.py \
  --model "$BASE_MODEL" --output-dir "$BASE_OUT" "${common[@]}" \
  >"$BASE_LOG" 2>&1

"$PY" sft/scripts/evaluate_sft_agent.py \
  --model "$SFT_MODEL" --output-dir "$SFT_OUT" "${common[@]}" \
  >"$SFT_LOG" 2>&1
