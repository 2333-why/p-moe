#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-models/deepseek-moe-16b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_stage1_generate}"
PROMPT="${PROMPT:-Load-aware p-bit routing is}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-outputs/offload_stage1}"

mkdir -p "${OUTPUT_DIR}" "${OFFLOAD_FOLDER}"

python scripts/test_generate.py \
  --model_name "${MODEL_NAME}" \
  --local_files_only \
  --prompt "${PROMPT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --offload_folder "${OFFLOAD_FOLDER}" \
  --output_dir "${OUTPUT_DIR}"
