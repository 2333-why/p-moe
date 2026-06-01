#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-models/deepseek-moe-16b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/run_stage2_wikitext_ppl}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-2-raw-v1}"
SPLIT="${SPLIT:-test}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
STRIDE="${STRIDE:-1024}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-outputs/offload_stage2}"

mkdir -p "${OUTPUT_DIR}" "${OFFLOAD_FOLDER}"

python scripts/eval_wikitext_ppl.py \
  --model_name "${MODEL_NAME}" \
  --local_files_only \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config_name "${DATASET_CONFIG_NAME}" \
  --split "${SPLIT}" \
  --block_size "${BLOCK_SIZE}" \
  --stride "${STRIDE}" \
  --offload_folder "${OFFLOAD_FOLDER}" \
  --output_dir "${OUTPUT_DIR}"
