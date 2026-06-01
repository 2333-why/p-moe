NEW="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/why/software/miniconda3"
ENV="/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/why/envs/deepseek_moe"

unset -f conda 2>/dev/null || true
unset -f __conda_exe 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true
unalias conda 2>/dev/null || true
hash -r

eval "$("$NEW/bin/conda" shell.bash hook)"
conda activate "$ENV"


cd /inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/why/p-moe

export EXP_ROOT=/inspire/qb-ilm/project/advanced-machine-learning/yanjunchi-24040/why
export PMOE_OUT="$EXP_ROOT/outputs"

export HF_HOME=/inspire/hdd/global_user/yanjunchi-24040/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "PMOE_OUT=$PMOE_OUT"
# p-MoE: Load-Aware p-bit Competitive Routing

This repository is a research framework for **Load-Aware p-bit Competitive Routing for Sparse Mixture-of-Experts Language Models**.

The core idea is to keep hard top-k sparse MoE routing in the forward pass, while using a p-bit surrogate in the backward pass and a load-aware bias to reduce expert collapse, load imbalance, router non-differentiability, and under-trained experts.

The project supports two complementary tracks:

- **Line A: DeepSeek-MoE-16B validation scaffold** for loading, generation, WikiText perplexity, router inspection, expert load statistics, and safe patch planning.
- **Line B: controllable Mini-MoE experiments** for algorithmic comparisons among standard top-k, noisy top-k, Gumbel/Concrete, straight-through, p-bit routing, and load-aware p-bit routing.

No script should load a model, download data, initialize WandB, train, evaluate, or patch a model at import time. Runtime work is only triggered by explicit CLI execution.

## Research Method

Traditional sparse MoE routing selects experts with a hard top-k mask:

```text
scores = router(x)
mask_hard = TopK(scores, k)
y = sum_i mask_i * gate_i * expert_i(x)
```

p-MoE keeps this hard sparse forward path, but replaces the router backward path with a p-bit surrogate:

```text
I_i = s_i - alpha * sum_{j != i} z_j + beta * (1/N - L_i)
q_i = sigmoid(I_i / T)
m = m_hard.detach() - q.detach() + q
```

The load estimate is maintained as an exponential moving average:

```text
L_i(t) = mu * L_i(t-1) + (1 - mu) * f_i(t)
```

Low-load experts receive positive bias and overused experts receive negative bias. The intended effect is better router gradients and more forward activation opportunities for under-trained experts.

## Stage Overview

| Stage | Purpose | Main scripts/configs |
| --- | --- | --- |
| 1 | DeepSeek-MoE loading, text generation, GPU memory inspection | `scripts/check_env.py`, `scripts/test_generate.py`, `configs/generation_config.yaml` |
| 1.5 | Separate online asset download from offline inference | `scripts/download_assets.py`, `configs/download_config.yaml` |
| 2 | WikiText-2 / WikiText-103 causal LM perplexity | `scripts/eval_wikitext_ppl.py`, `configs/eval_wikitext_config.yaml` |
| 3 | LoRA / QLoRA continued pretraining scaffold | `scripts/train_lora.py`, `scripts/train_qlora.py`, `configs/train_lora_config.yaml`, `configs/train_qlora_config.yaml` |
| 4 | p-bit router core and safe patch scaffold | `scripts/test_pbit_router_unit.py`, `scripts/inspect_router.py`, `configs/pbit_router_config.yaml` |
| 5 | MiniGPT / Mini-MoE controllable experiments | `scripts/train_mini_moe.py`, `scripts/eval_mini_moe.py`, `configs/mini_moe_config.yaml` |
| 6 | DeepSeek router inspection, runtime p-bit smoke patch, and patch planning | `scripts/inspect_router.py`, `scripts/train_deepseek_pbit_smoke.py`, `src/deepseek_pbit_patch.py`, `src/router_patch.py` |
| 7 | Experiment orchestration and result collection | `scripts/collect_results.py`, `configs/result_collect_config.yaml` |

## Environment Setup

The managed server startup may define project and cache variables:

```bash
export HROOT="$(cd $(dirname ${BASH_SOURCE[0]}); cd ..; pwd)"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/large/hf/cache
```

If the platform provides a mirror setup helper, source it before installing or downloading:

```bash
source mirror.sh
```

Using server helper commands:

```bash
hc-new deepseek_moe python=3.10
hc-activate deepseek_moe
```

Standard conda fallback:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe
```

Optional CUDA 12.8 PyTorch install:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Do not hard-code cache paths in source code. The scripts should respect `HF_HOME`, `HF_ENDPOINT`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE` when they are already set.

## Dependencies

Base dependencies are listed in `requirements.txt`:

```text
torch
transformers
accelerate
safetensors
sentencepiece
protobuf
datasets
pyyaml
tqdm
numpy
pandas
```

Optional packages are commented in `requirements.txt` and should only be installed when needed:

```text
# peft
# bitsandbytes
# wandb
# matplotlib
```

Stage 1 and Stage 2 should not require PEFT, bitsandbytes, WandB, or matplotlib.

## Secret Safety

Never write raw secrets to code, configs, README files, logs, JSON summaries, or generated reports.

Sensitive variables may include:

```bash
HF_TOKEN
WANDB_API_KEY
```

Scripts may report only `set` or `not set`. They must not print raw values.

WandB is disabled by default. Training scripts should import and initialize WandB only when `--use_wandb` is explicitly passed, and credentials should be read from environment variables only.

## Model Download Workflow

Stage 1.5 separates asset download from offline inference. Use `models/` as the default local model root.

```bash
python scripts/download_assets.py \
  --config configs/download_config.yaml \
  --mode local \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --local_dir models/deepseek-moe-16b-base
```

Expected output should include a `summary.json` under `outputs/` or the configured run directory. The download script may access the network and Hugging Face cache only when explicitly executed.

## Offline Inference Workflow

After assets are available locally:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_generate.py \
  --config configs/generation_config.yaml \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_new_tokens 120
```

Use `--local_files_only` for offline runs. Use `--offload_folder outputs/offload` if GPU memory is tight.

## WikiText PPL Workflow

Run WikiText evaluation manually on the GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_wikitext_ppl.py \
  --config configs/eval_wikitext_config.yaml \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --dataset_config_name wikitext-2-raw-v1 \
  --split test \
  --block_size 2048 \
  --stride 1024
```

The evaluation script should save `eval_summary.json` in the configured output directory.

## LoRA / QLoRA Workflow

LoRA continued pretraining scaffold:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --config configs/train_lora_config.yaml \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

QLoRA continued pretraining scaffold:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --config configs/train_qlora_config.yaml \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

These scripts may train only when explicitly executed. They should provide clear missing-dependency guidance if `peft` or `bitsandbytes` is unavailable.

## p-bit Router Scaffold

The p-bit router implementation is configured by `configs/pbit_router_config.yaml`.

Manual unit check:

```bash
python scripts/test_pbit_router_unit.py --config configs/pbit_router_config.yaml
```

Manual router inspection before any DeepSeek patch planning:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_router.py \
  --config configs/pbit_router_config.yaml \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

The DeepSeek patch path must remain scaffold-only by default. Inspect first, then design a patch plan from the saved inspection report.

## DeepSeek p-bit Smoke Training

After router inspection confirms DeepSeek uses `MoEGate` under `model.layers.*.mlp.gate`, the repository provides an in-memory runtime patch for a very small continued-pretraining smoke test. It does not edit Hugging Face cache files and it does not run unless you explicitly execute the script.

Use one shared output root on the server:

```bash
export EXP_ROOT="$(cd ..; pwd)"
export PMOE_OUT="$EXP_ROOT/outputs"
mkdir -p "$PMOE_OUT"
```

Baseline router smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_deepseek_pbit_smoke.py \
  --config configs/train_deepseek_pbit_smoke.yaml \
  --no_pbit_patch \
  --output_dir "$PMOE_OUT/deepseek_baseline_router_smoke"
```

p-bit router smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_deepseek_pbit_smoke.py \
  --config configs/train_deepseek_pbit_smoke.yaml \
  --use_pbit_patch \
  --output_dir "$PMOE_OUT/deepseek_pbit_router_smoke"
```

The first DeepSeek p-bit run should stay small, for example `max_steps: 20`, `block_size: 512`, and `freeze_non_router: true`. This only checks that forward/backward works, router gradients exist, and load metrics are recorded. It is not enough for a paper claim.

## DeepSeek QLoRA + p-bit Training

For a more complete DeepSeek experiment, use QLoRA continued pretraining so adapter parameters participate in training while the router can also be trained and optionally patched with p-bit backward:

```bash
python scripts/train_deepseek_qlora_pbit.py \
  --config configs/train_deepseek_qlora_pbit.yaml \
  --no_pbit_patch \
  --output_dir "$PMOE_OUT/deepseek_qlora_baseline_1000"
```

```bash
python scripts/train_deepseek_qlora_pbit.py \
  --config configs/train_deepseek_qlora_pbit.yaml \
  --use_pbit_patch \
  --alpha 0.0 \
  --beta 0.01 \
  --temperature 1.0 \
  --output_dir "$PMOE_OUT/deepseek_qlora_pbit_1000"
```

Evaluate saved adapters with:

```bash
python scripts/eval_deepseek_adapter_ppl.py \
  --base_model deepseek-ai/deepseek-moe-16b-base \
  --adapter_path "$PMOE_OUT/deepseek_qlora_pbit_1000" \
  --output_dir "$PMOE_OUT" \
  --local_files_only
```

## Mini-MoE Workflow

The Mini-MoE line is the main controllable algorithm validation path.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_mini_moe.py \
  --config configs/mini_moe_config.yaml \
  --router_type pbit_load_aware
```

Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_mini_moe.py \
  --config configs/mini_moe_config.yaml \
  --checkpoint outputs/mini_moe/latest
```

The framework should support dense, standard top-k, noisy top-k, Gumbel/Concrete, top-k straight-through, p-bit, p-bit with competition, p-bit with load-aware bias, and p-bit warmup variants.

Metrics to collect include train loss, eval PPL, expert load variance, dead expert ratio, expert entropy, router gradient norm, expert gradient norm, and routing statistics by layer.

## Result Collection

Collect results after manual experiments:

```bash
python scripts/collect_results.py \
  --config configs/result_collect_config.yaml \
  --outputs_dir outputs
```

Expected aggregate artifacts:

```text
outputs/results_table.csv
outputs/results_summary.md
```

The collector should scan:

```text
outputs/run_*/summary.json
outputs/run_*/eval_summary.json
outputs/**/train_summary.json
```

## Script Safety

Safe to run for environment or static checks:

- `python scripts/check_env.py`
- `python scripts/test_pbit_router_unit.py --config configs/pbit_router_config.yaml`
- `python scripts/collect_results.py --config configs/result_collect_config.yaml`

Scripts that may download assets:

- `python scripts/download_assets.py ...`

Scripts that may load a model:

- `python scripts/test_generate.py ...`
- `python scripts/eval_wikitext_ppl.py ...`
- `python scripts/inspect_model.py ...`
- `python scripts/inspect_router.py ...`

Scripts that may train:

- `python scripts/train_lora.py ...`
- `python scripts/train_qlora.py ...`
- `python scripts/train_mini_moe.py ...`

Scripts that may evaluate:

- `python scripts/eval_wikitext_ppl.py ...`
- `python scripts/eval_mini_moe.py ...`

## GPU Memory Advice

DeepSeek-MoE-16B can require substantial GPU memory. Prefer:

- local model directories after download;
- `--local_files_only` for offline runs;
- `device_map auto` when supported;
- lower `--max_new_tokens` during generation smoke tests;
- smaller `--block_size` for PPL if memory is tight;
- `--offload_folder outputs/offload` for CPU or disk offload;
- Mini-MoE experiments for algorithmic iteration before full-model validation.

Keep `outputs/` on storage with enough capacity for summaries, checkpoints, logs, and offload files.

## Common Errors

Missing optional PEFT dependency:

```text
Install peft or run only non-LoRA stages.
```

Missing optional bitsandbytes dependency:

```text
Install bitsandbytes for QLoRA, or use regular LoRA.
```

Hugging Face offline cache miss:

```text
Download assets first, verify local_dir, then rerun with --local_files_only.
```

CUDA out of memory:

```text
Reduce block_size, stride, batch size, or max_new_tokens; enable offload; use Mini-MoE for controllable tests.
```

Authentication needed:

```text
Set HF_TOKEN in the environment. Do not paste the token into configs or source files.
```

## Recommended Manual Experiment Order

1. Check the environment:

```bash
python scripts/check_env.py
```

2. Download model assets:

```bash
python scripts/download_assets.py \
  --mode local \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --local_dir models/deepseek-moe-16b-base
```

3. Run offline generation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_generate.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_new_tokens 120
```

4. Run WikiText PPL:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_wikitext_ppl.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --dataset_config_name wikitext-2-raw-v1 \
  --split test \
  --block_size 2048 \
  --stride 1024
```

5. Inspect routers before any patch planning:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_router.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

6. Run Mini-MoE experiments after reviewing `configs/mini_moe_config.yaml`.

7. Collect results:

```bash
python scripts/collect_results.py --outputs_dir outputs
```
