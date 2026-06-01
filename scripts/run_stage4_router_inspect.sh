#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-models/deepseek-moe-16b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_stage4_router_inspect}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-outputs/offload_stage4}"

mkdir -p "${OUTPUT_DIR}" "${OFFLOAD_FOLDER}"

python scripts/inspect_router.py \
  --model_name "${MODEL_NAME}" \
  --local_files_only \
  --offload_folder "${OFFLOAD_FOLDER}" \
  --output_dir "${OUTPUT_DIR}"
