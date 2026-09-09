#!/usr/bin/env python3
"""Build the repair-balanced 300-sample protocol warm-up dataset.

This reuses the audited v5 prefix pool and the original deterministic scoring
and disjoint-dev logic.  The original protocol dataset is not overwritten.
"""
from __future__ import annotations

from pathlib import Path

import build_protocol_dataset as builder


builder.OUT = builder.ROOT / "data/coding_sft_protocol_v2"
builder.POOL_DIR = builder.ROOT / "data/coding_sft_v5"
builder.QUOTAS = {
    "problem_submit": 30,
    "first_failure_run": 80,
    "first_failure_direct_repair": 55,
    "post_run_repair": 90,
    "second_failure_run": 20,
    "multiround_final_submit": 25,
}
builder.TOTAL_TARGET = sum(builder.QUOTAS.values())
builder.HARD_GATES = {
    "total_min": 300,
    "total_max": 300,
    "first_failure_run_min": 80,
    "second_failure_run_min": 20,
    "multiround_final_submit_min": 20,
    "run_ge_submit": False,
}


if __name__ == "__main__":
    raise SystemExit(builder.main())
