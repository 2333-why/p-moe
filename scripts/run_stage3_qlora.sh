#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-models/deepseek-moe-16b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_stage3_qlora}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-2-raw-v1}"
SPLIT="${SPLIT:-train}"
MAX_STEPS="${MAX_STEPS:-100}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_qlora_config.yaml}"

mkdir -p "${OUTPUT_DIR}"

python scripts/train_qlora.py \
  --config "${CONFIG_PATH}" \
  --model_name "${MODEL_NAME}" \
  --local_files_only \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config_name "${DATASET_CONFIG_NAME}" \
  --split "${SPLIT}" \
  --max_steps "${MAX_STEPS}" \
  --output_dir "${OUTPUT_DIR}"
