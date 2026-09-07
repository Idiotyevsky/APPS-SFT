#!/usr/bin/env bash
# =============================================================================
# download_model_modelscope.sh — 用 ModelScope 下载模型到共享模型目录
#
# 用法（任选其一，建议由本人执行）：
#   bash scripts/download_model_modelscope.sh
#   bash scripts/download_model_modelscope.sh Qwen/Qwen3.5-4B /home/zfs02/model
#
# 默认：
#   模型  : Qwen/Qwen3.5-4B   （ModelScope 仓库 id，可用 --model 覆盖）
#   目录  : /home/zfs02/model/<模型名>   （如 /home/zfs02/model/Qwen3.5-4B）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-4B}"
BASE_DIR="${2:-/home/zfs02/model}"
NAME="$(basename "${MODEL}")"
TARGET="${BASE_DIR}/${NAME}"

mkdir -p "${TARGET}"
echo "== 模型: ${MODEL}"
echo "== 目标: ${TARGET}"

# 定位 python（GPU 机常用环境优先，随后退到系统 python）
PY=""
for cand in /home/zfs01/jiangjr/envs/opd/bin/python /home/nfs05/anaconda3/bin/python3 python3; do
  if [[ -x "$cand" ]] || command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[[ -n "$PY" ]] || { echo "no python found"; exit 1; }
echo "== python: ${PY}"

# 确保 modelscope 可用
"$PY" - <<'EOF' || { echo "installing modelscope..."; "$PY" -m pip install -q modelscope; }
import modelscope  # noqa
EOF

"$PY" - <<EOF
from modelscope import snapshot_download

model_id = "${MODEL}"
local_dir = "${TARGET}"
print(f"downloading {model_id} -> {local_dir}", flush=True)
snapshot_download(
    model_id=model_id,
    local_dir=local_dir,
    ignore_file_pattern=["*.msindex"],
)
print(f"done: {local_dir}", flush=True)
EOF

echo "== 完成。目录内容:"
ls -la "${TARGET}" | head -20
