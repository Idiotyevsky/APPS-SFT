#!/usr/bin/env bash
# =============================================================================
# eval.sh — 默认 Agent 工具评测；--protocol code 显式运行旧单次代码基线
#
# Agent 示例：CUDA_VISIBLE_DEVICES=0 scripts/eval.sh --model /path/to/model --per-difficulty 10
# Agent 默认用模型完整上下文；--resume 续跑，--help 查看参数。
# 以下说明仅适用于 --protocol code：
# 覆盖完整流程：
#   1) 探测 GPU（按空闲显存选卡，显存不足即报错退出）
#   2) 选择评测 python 环境（优先 /home/zfs01/jiangjr/envs/opd，回退 anaconda）
#   3) 补齐 ninja / CUDA 库路径（vLLM torch.compile + flashinfer 需要）
#   4) vLLM 批量生成 750 条输出（greedy, max-new-tokens 可配）
#   5) 多进程并行判分（本地私有 grader，每进程独立沙箱）
#   6) 汇总写 data/eval/results/<model>_eval.json 并打印 pass@1 表
#
# 用法（在 GPU 机/共享盘任意一处执行均可）：
#   scripts/eval.sh --protocol code                          # 默认全流程
#   scripts/eval.sh --protocol code --model /path/to/weights --limit 20   # 快速冒烟
#   scripts/eval.sh --protocol code --grade-workers 40 --max-new-tokens 1024
#   scripts/eval.sh --protocol code --fresh                  # 清掉旧 progress 重跑
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Agent evaluation is the default. The old single-shot code baseline remains
# available explicitly via --protocol code; never mix the two metrics.
PROTOCOL=agent
EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--protocol" ]]; then
    [[ $# -ge 2 ]] || { echo "--protocol needs agent or code" >&2; exit 2; }
    PROTOCOL="$2"; shift 2
  else
    EVAL_ARGS+=("$1"); shift
  fi
done
set -- "${EVAL_ARGS[@]}"
if [[ "$PROTOCOL" == "agent" ]]; then
  AGENT_MODEL="${MODEL:-/home/zfs02/model/Qwen2.5-Coder-7B-Instruct}"
  AGENT_MODE=problem-only
  for ((i=0; i<${#EVAL_ARGS[@]}; i++)); do
    case "${EVAL_ARGS[$i]}" in
      --model) AGENT_MODEL="${EVAL_ARGS[$((i+1))]:-$AGENT_MODEL}" ;;
      --model=*) AGENT_MODEL="${EVAL_ARGS[$i]#--model=}" ;;
      --mode) AGENT_MODE="${EVAL_ARGS[$((i+1))]:-$AGENT_MODE}" ;;
      --mode=*) AGENT_MODE="${EVAL_ARGS[$i]#--mode=}" ;;
    esac
  done
  AGENT_SLUG="${AGENT_MODEL//\//__}"
  AGENT_PY="${EVAL_PYTHON:-/home/zfs01/jiangjr/envs/opd/bin/python}"
  if [[ ! -x "$AGENT_PY" ]]; then
    echo "Set EVAL_PYTHON to an environment with vLLM and transformers." >&2
    exit 1
  fi
  AGENT_ENV_BIN="$(dirname "$AGENT_PY")"
  export PATH="$AGENT_ENV_BIN:$PATH"
  export LIBRARY_PATH="$AGENT_ENV_BIN/../lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$AGENT_ENV_BIN/../lib:${LD_LIBRARY_PATH:-}"
  if [[ -x "$AGENT_ENV_BIN/nvcc" ]]; then
    export CUDA_HOME="$(dirname "$AGENT_ENV_BIN")"
  fi
  # CUDA_VISIBLE_DEVICES is caller-controlled; do not silently select a GPU.
  exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$AGENT_PY" \
    sft/scripts/evaluate_sft_agent.py --model "$AGENT_MODEL" \
    --eval-path "${EVAL_PATH:-$ROOT/data/eval/toolapps_eval.jsonl}" \
    --output-dir "data/eval/results/agent/$AGENT_SLUG/$AGENT_MODE" "$@"
fi
if [[ "$PROTOCOL" != "code" ]]; then
  echo "--protocol must be agent or code" >&2
  exit 2
fi

MODEL="${MODEL:-/home/zfs02/model/Qwen2.5-Coder-7B-Instruct}"
EVAL_PATH="${EVAL_PATH:-data/eval/toolapps_eval.jsonl}"
OUT_DIR="data/eval/results"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GRADE_WORKERS="${GRADE_WORKERS:-40}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
LIMIT="${LIMIT:-0}"
FRESH=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2 ;;
    --eval-path)    EVAL_PATH="$2"; shift 2 ;;
    --limit)        LIMIT="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --grade-workers) GRADE_WORKERS="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --fresh)        FRESH=1; shift ;;
    --resume)       RESUME=1; shift ;;
    --help|-h)
      sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

mkdir -p "$OUT_DIR"

# ---- python 环境 ----
PY=""
for cand in /home/zfs01/jiangjr/envs/opd/bin/python /home/nfs05/anaconda3/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
    PY="$cand"; break
  fi
done
if [[ -z "$PY" ]]; then echo "no suitable python found"; exit 1; fi
ENV_BIN="$(dirname "$PY")"
export PATH="$ENV_BIN:$PATH"
export LIBRARY_PATH="/usr/local/cuda/lib64:$ENV_BIN/../lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$ENV_BIN/../lib:${LD_LIBRARY_PATH:-}"
echo "python: $PY"

# ---- GPU 选择（空闲显存最大且 >= MIN_FREE_GIB）----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi missing"; exit 1
fi
readarray -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
BEST=""
BEST_FREE=0
for line in "${GPU_LINES[@]}"; do
  idx="${line%%,*}"; rest="${line#*,}"
  free="${rest%%,*}"
  if [[ "$free" -gt "$BEST_FREE" ]]; then BEST_FREE="$free"; BEST="$idx"; fi
done
echo "GPU free memory (MiB):"
printf '%s\n' "${GPU_LINES[@]}" | sed 's/^/  /'
FREE_GIB=$((BEST_FREE / 1024))
if [[ "$FREE_GIB" -lt "$MIN_FREE_GIB" ]]; then
  echo "no GPU with >= ${MIN_FREE_GIB}GiB free (best: GPU $BEST ${FREE_GIB}GiB). Aborting."
  exit 1
fi
echo "using GPU $BEST (${FREE_GIB}GiB free)"

# ---- 输出管理 ----
SLUG="${MODEL//\//__}"
if [[ "$FRESH" -eq 1 ]]; then
  rm -f "$OUT_DIR/${SLUG}"*progress.jsonl "$OUT_DIR/${SLUG}"*_eval.json "$OUT_DIR/${SLUG}"*_generations.jsonl "$OUT_DIR/eval_run.log"
fi
RESUME_FLAG=()
if [[ "$RESUME" -eq 1 ]]; then RESUME_FLAG+=(--resume); fi
LIMIT_FLAG=()
if [[ "$LIMIT" -gt 0 ]]; then LIMIT_FLAG+=(--limit "$LIMIT"); fi

# ---- 评测（vLLM 生成 → 并行判分）----
echo "==== starting evaluation: $MODEL ===="
CUDA_VISIBLE_DEVICES="$BEST" PYTHONPATH=src "$PY" \
  scripts/evaluate_model.py \
  --backend vllm \
  --model "$MODEL" \
  --eval-path "$EVAL_PATH" \
  --output-dir "$OUT_DIR" \
  --dtype bf16 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --grade-workers "$GRADE_WORKERS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --save-completions \
  "${LIMIT_FLAG[@]}" "${RESUME_FLAG[@]}"

echo "==== done: results in $OUT_DIR/${SLUG}*_eval.json ===="
