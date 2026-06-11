#!/usr/bin/env bash
set -euo pipefail

# End-to-end DeepSeek QLoRA baseline vs QLoRA+p-bit experiment.
# Run manually on a GPU node after activating the intended conda environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXP_ROOT="${EXP_ROOT:-$(cd .. && pwd)}"
PMOE_OUT="${PMOE_OUT:-$EXP_ROOT/outputs}"
CONFIG="${CONFIG:-configs/train_deepseek_qlora_pbit.yaml}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/deepseek-moe-16b-base}"

MAX_STEPS="${MAX_STEPS:-1000}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
ALPHA="${ALPHA:-0.0}"
BETA="${BETA:-0.01}"
TEMPERATURE="${TEMPERATURE:-1.0}"

BASELINE_OUT="${BASELINE_OUT:-$PMOE_OUT/deepseek_qlora_baseline_${MAX_STEPS}}"
PBIT_OUT="${PBIT_OUT:-$PMOE_OUT/deepseek_qlora_pbit_${MAX_STEPS}_a${ALPHA}_b${BETA}}"

mkdir -p "$PMOE_OUT"

tlog() {
  local logfile=$1
  shift
  "$@" 2>&1 | tee -a "$logfile"
}

echo "ROOT_DIR=$ROOT_DIR"
echo "PMOE_OUT=$PMOE_OUT"
echo "CONFIG=$CONFIG"
echo "MODEL_NAME=$MODEL_NAME"
echo "MAX_STEPS=$MAX_STEPS BLOCK_SIZE=$BLOCK_SIZE GRAD_ACCUM=$GRAD_ACCUM LEARNING_RATE=$LEARNING_RATE"
echo "ALPHA=$ALPHA BETA=$BETA TEMPERATURE=$TEMPERATURE"
echo "BASELINE_OUT=$BASELINE_OUT"
echo "PBIT_OUT=$PBIT_OUT"

tlog "$PMOE_OUT/deepseek_qlora_baseline_${MAX_STEPS}.log" python scripts/train_deepseek_qlora_pbit.py \
  --config "$CONFIG" \
  --no_pbit_patch \
  --model_name "$MODEL_NAME" \
  --max_steps "$MAX_STEPS" \
  --block_size "$BLOCK_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --output_dir "$BASELINE_OUT"

tlog "$PMOE_OUT/deepseek_qlora_pbit_${MAX_STEPS}_a${ALPHA}_b${BETA}.log" python scripts/train_deepseek_qlora_pbit.py \
  --config "$CONFIG" \
  --use_pbit_patch \
  --model_name "$MODEL_NAME" \
  --alpha "$ALPHA" \
  --beta "$BETA" \
  --temperature "$TEMPERATURE" \
  --max_steps "$MAX_STEPS" \
  --block_size "$BLOCK_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --output_dir "$PBIT_OUT"

tlog "$PMOE_OUT/eval_qlora_baseline_${MAX_STEPS}.log" python scripts/eval_deepseek_adapter_ppl.py \
  --base_model "$MODEL_NAME" \
  --adapter_path "$BASELINE_OUT" \
  --output_dir "$PMOE_OUT" \
  --local_files_only

tlog "$PMOE_OUT/eval_qlora_pbit_${MAX_STEPS}_a${ALPHA}_b${BETA}.log" python scripts/eval_deepseek_adapter_ppl.py \
  --base_model "$MODEL_NAME" \
  --adapter_path "$PBIT_OUT" \
  --output_dir "$PMOE_OUT" \
  --local_files_only

python scripts/collect_results.py \
  --outputs_dir "$PMOE_OUT" \
  --csv_path "$PMOE_OUT/results_table.csv" \
  --md_path "$PMOE_OUT/results_summary.md"

grep deepseek "$PMOE_OUT/results_summary.md" || true
