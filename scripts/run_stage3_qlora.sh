#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-moe-16b-base}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-wikitext-2-raw-v1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
USE_WANDB="${USE_WANDB:-0}"
MAX_STEPS="${MAX_STEPS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-}"
STAGE3_SCRIPT="${STAGE3_SCRIPT:-scripts/train_qlora.py}"

echo "Stage 3 QLoRA wrapper"
echo "Project root: ${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}/${DATASET_CONFIG_NAME}"
echo "Local files only: ${LOCAL_FILES_ONLY}"
echo "WandB enabled: ${USE_WANDB}"

if [[ ! -f "${STAGE3_SCRIPT}" ]]; then
  echo "Missing ${STAGE3_SCRIPT}."
  echo "Stage 3 QLoRA training is not implemented in this checkout yet."
  echo "Set STAGE3_SCRIPT to the training entry point when it exists."
  exit 2
fi

args=(
  "${STAGE3_SCRIPT}"
  --model_name "${MODEL_NAME}"
  --dataset_name "${DATASET_NAME}"
  --dataset_config_name "${DATASET_CONFIG_NAME}"
  --train_split "${TRAIN_SPLIT}"
  --eval_split "${EVAL_SPLIT}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ "${LOCAL_FILES_ONLY}" != "0" ]]; then
  args+=(--local_files_only)
fi

if [[ "${USE_WANDB}" != "0" ]]; then
  args+=(--use_wandb)
fi

if [[ -n "${MAX_STEPS}" ]]; then
  args+=(--max_steps "${MAX_STEPS}")
fi

if [[ -n "${LEARNING_RATE}" ]]; then
  args+=(--learning_rate "${LEARNING_RATE}")
fi

if [[ -n "${MAX_SEQ_LENGTH}" ]]; then
  args+=(--max_seq_length "${MAX_SEQ_LENGTH}")
fi

"${PYTHON_BIN}" "${args[@]}"
