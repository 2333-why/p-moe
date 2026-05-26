#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-moe-16b-base}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-2-raw-v1}"
SPLIT="${SPLIT:-test}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
STRIDE="${STRIDE:-1024}"
MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
STAGE2_SCRIPT="${STAGE2_SCRIPT:-scripts/eval_wikitext_ppl.py}"

echo "Stage 2 evaluation wrapper"
echo "Project root: ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}/${DATASET_CONFIG_NAME} (${SPLIT})"
echo "Local files only: ${LOCAL_FILES_ONLY}"

if [[ ! -f "${STAGE2_SCRIPT}" ]]; then
  echo "Missing ${STAGE2_SCRIPT}."
  echo "Stage 2 evaluation is not implemented in this checkout yet."
  echo "Set STAGE2_SCRIPT to the evaluation entry point when it exists."
  exit 2
fi

args=(
  "${STAGE2_SCRIPT}"
  --model_name "${MODEL_NAME}"
  --dataset_name "${DATASET_NAME}"
  --dataset_config_name "${DATASET_CONFIG_NAME}"
  --split "${SPLIT}"
  --max_length "${MAX_LENGTH}"
  --stride "${STRIDE}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${LOCAL_FILES_ONLY}" != "0" ]]; then
  args+=(--local_files_only)
fi

if [[ -n "${MAX_EVAL_TOKENS}" ]]; then
  args+=(--max_eval_tokens "${MAX_EVAL_TOKENS}")
fi

"${PYTHON_BIN}" "${args[@]}"
