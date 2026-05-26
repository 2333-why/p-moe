#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-moe-16b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
STAGE4_SCRIPT="${STAGE4_SCRIPT:-scripts/inspect_router.py}"

echo "Stage 4 router inspection wrapper"
echo "Project root: ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Model: ${MODEL_NAME}"
echo "Local files only: ${LOCAL_FILES_ONLY}"

if [[ ! -f "${STAGE4_SCRIPT}" ]]; then
  echo "Missing ${STAGE4_SCRIPT}."
  echo "Stage 4 router inspection is not implemented in this checkout yet."
  echo "Set STAGE4_SCRIPT to the router inspection entry point when it exists."
  exit 2
fi

args=(
  "${STAGE4_SCRIPT}"
  --model_name "${MODEL_NAME}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${LOCAL_FILES_ONLY}" != "0" ]]; then
  args+=(--local_files_only)
fi

"${PYTHON_BIN}" "${args[@]}"
