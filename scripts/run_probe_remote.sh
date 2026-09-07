#!/usr/bin/env bash
# run_probe_remote.sh <problem_id>  —— 远端后台跑单题探针并落盘日志
set -euo pipefail
PID_ARG="${1:-4004}"
cd /home/zfs02/jiangjr/data_syn
pkill -9 -f 'probe_one[.]py' 2>/dev/null || true
rm -f sft/outputs/probe_one.log
setsid nohup bash -c "cd /home/zfs02/jiangjr/data_syn && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src ./sft-venv/bin/python -u sft/scripts/probe_one.py ${PID_ARG} > sft/outputs/probe_one.log 2>&1" \
  < /dev/null > /dev/null 2>&1 &
echo "STARTED pid $!"
