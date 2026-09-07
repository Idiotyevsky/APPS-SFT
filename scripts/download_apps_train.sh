#!/usr/bin/env bash
set -euo pipefail

# usage: $0 APPS_REVISION OUTPUT_PATH [EXPECTED_SHA256 [train|test]]
# default split: train
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 APPS_REVISION OUTPUT_PATH [EXPECTED_SHA256] [train|test]" >&2
  echo "example: $0 <commit> data/raw/apps/train.jsonl <sha256>" >&2
  echo "example: $0 <commit> data/raw/apps/test.jsonl <sha256> test" >&2
  exit 2
fi

revision=$1
output=$2
expected=${3:-}
split=${4:-train}
if [[ "$split" != "train" && "$split" != "test" ]]; then
  echo "split must be train or test" >&2
  exit 2
fi
if [[ "$revision" == "main" || "$revision" == "master" || -z "$revision" ]]; then
  echo "use a pinned APPS commit revision for formal runs (not main/master)" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
url="https://huggingface.co/datasets/codeparrot/apps/resolve/${revision}/${split}.jsonl?download=true"
curl --fail --location --retry 5 --continue-at - --output "$output" "$url"
actual=$(sha256sum "$output" | awk '{print $1}')
echo "downloaded split=$split revision=$revision sha256=$actual path=$output"
if [[ -n "$expected" && "$actual" != "$expected" ]]; then
  echo "SHA256 mismatch: expected $expected, got $actual" >&2
  exit 1
fi
