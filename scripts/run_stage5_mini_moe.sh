#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/mini_moe_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_stage5_mini_moe}"
METHOD="${METHOD:-pbit_load_aware}"
MAX_STEPS="${MAX_STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"

mkdir -p "${OUTPUT_DIR}"

python scripts/train_mini_moe.py \
  --config "${CONFIG_PATH}" \
  --method "${METHOD}" \
  --max_steps "${MAX_STEPS}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --output_dir "${OUTPUT_DIR}"
