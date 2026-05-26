# DeepSeek-MoE p-bit Routing Probe

This repository contains staged scaffolds for probing `deepseek-ai/deepseek-moe-16b-base` and later experimenting with p-bit-assisted MoE routing. The code is safe-by-default: development checks must not download the model, load the large model, evaluate WikiText, or train unless the user explicitly runs those scripts.

## Layout

```text
.
|-- AGENTS.md
|-- requirements.txt
|-- configs/
|   |-- download_config.yaml
|   |-- generation_config.yaml
|   |-- eval_wikitext_config.yaml
|   |-- train_lora_config.yaml
|   |-- train_qlora_config.yaml
|   `-- pbit_router_config.yaml
|-- models/
|   `-- .gitkeep
|-- scripts/
|   |-- check_env.py
|   |-- download_assets.py
|   |-- test_generate.py
|   |-- inspect_model.py
|   |-- eval_wikitext_ppl.py
|   |-- train_lora.py
|   |-- train_qlora.py
|   |-- inspect_router.py
|   |-- test_pbit_router_unit.py
|   |-- collect_results.py
|   |-- run_stage2_eval.sh
|   |-- run_stage3_qlora.sh
|   `-- run_stage4_router_inspect.sh
|-- src/
`-- outputs/
```

## Installation

Preferred managed-server workflow:

```bash
source mirror.sh
hc-new deepseek_moe python=3.10
hc-activate deepseek_moe

pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Fallback:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe
pip install -r requirements.txt
```

Optional Stage 3 dependencies are intentionally not required for Stage 1/2:

```bash
pip install peft bitsandbytes
# Optional only when explicitly using --use_wandb:
pip install wandb
```

Never put raw `HF_TOKEN`, `WANDB_API_KEY`, Bark keys, or other secrets in source files, logs, JSON outputs, or commits.

## Stage 1: Environment And Generation Probe

Check the environment:

```bash
python scripts/check_env.py
```

After model assets are available, run generation:

```bash
python scripts/test_generate.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_new_tokens 120
```

Outputs go to `outputs/run_YYYYMMDD_HHMMSS/summary.json`, `generated.txt`, `config_used.yaml`, and `device_map.json`.

## Stage 1.5: Online Download And Offline Inference

Download assets only. This does not run inference.

Default local download:

```bash
python scripts/download_assets.py
```

Explicit local download:

```bash
python scripts/download_assets.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --mode local \
  --local_dir models/deepseek-moe-16b-base \
  --revision main \
  --resume_download
```

Cache download using `HF_HOME`:

```bash
python scripts/download_assets.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --mode cache \
  --revision main \
  --resume_download
```

Offline generation from local directory:

```bash
python scripts/test_generate.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_new_tokens 120
```

Manual model structure inspection:

```bash
python scripts/inspect_model.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

## Stage 2: WikiText Perplexity

Default config: `configs/eval_wikitext_config.yaml`.

Run only after the model and dataset are available locally:

```bash
python scripts/eval_wikitext_ppl.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --dataset_config_name wikitext-2-raw-v1 \
  --split test \
  --block_size 2048 \
  --stride 1024
```

Outputs:

```text
outputs/run_YYYYMMDD_HHMMSS/
|-- config_used.yaml
|-- device_map.json
`-- eval_summary.json
```

`eval_summary.json` records `ppl`, `total_nll`, evaluated tokens, timing, device map, and offload flags.

## Stage 3: LoRA / QLoRA Continued Pretraining

These are training frameworks only. Do not run them until Stage 1/2 show the hardware and local caches are ready.

LoRA:

```bash
python scripts/train_lora.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_steps 100
```

QLoRA:

```bash
python scripts/train_qlora.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_steps 100
```

WandB is disabled by default. Enable it only when needed:

```bash
python scripts/train_qlora.py --use_wandb --local_files_only
```

The scripts read WandB settings from environment variables only and must not print or save raw keys. Training outputs include `train_config_used.yaml`, `train_summary.json`, Trainer state files when produced, and adapter outputs under the run directory.

## Stage 4: p-bit Router Scaffold

The p-bit code is a scaffold. It does not assume DeepSeek router class names and does not patch the model automatically.

Safe tensor-only test:

```bash
python scripts/test_pbit_router_unit.py
```

Manual router inspection:

```bash
python scripts/inspect_router.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

Inspect `router_inspection.json` and `router_inspection.txt` before any real patching. Do not replace router modules until the true DeepSeek-MoE router contract is confirmed.

## Stage 5: Orchestration And Result Collection

Optional shell templates default to local-only execution:

```bash
bash scripts/run_stage2_eval.sh
bash scripts/run_stage3_qlora.sh
bash scripts/run_stage4_router_inspect.sh
```

After any runs, collect results:

```bash
python scripts/collect_results.py
```

Outputs:

```text
outputs/results_table.csv
outputs/results_summary.md
```

## Recommended Experiment Order

1. `python scripts/check_env.py`
2. `python scripts/download_assets.py --mode local --local_dir models/deepseek-moe-16b-base --resume_download`
3. `python scripts/test_generate.py --model_name models/deepseek-moe-16b-base --local_files_only`
4. `python scripts/eval_wikitext_ppl.py --model_name models/deepseek-moe-16b-base --local_files_only`
5. `python scripts/test_pbit_router_unit.py`
6. `python scripts/inspect_router.py --model_name models/deepseek-moe-16b-base --local_files_only`
7. Only after the above, decide whether to run LoRA/QLoRA or router patch experiments.
8. `python scripts/collect_results.py`

## Memory Judgment

- `has_cpu_offload: false` and `has_disk_offload: false`: best case.
- `has_cpu_offload: true`: possible but likely slow.
- `has_disk_offload: true`: not practical for later experiments.
- CUDA OOM: use more GPUs, smaller eval slices, or defer to QLoRA/quantized experiments.

## Development Checks

Allowed lightweight checks:

```bash
python -m compileall scripts src
python scripts/check_env.py
python scripts/test_pbit_router_unit.py
```

Do not run download, generation, evaluation, training, or router inspection scripts during automated development unless explicitly requested.
