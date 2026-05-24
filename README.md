# DeepSeek-MoE-16B-Base Inference Probe

This repository is Stage 1 of a p-bit-assisted MoE routing research project. Stage 1 is inference only: it checks whether `deepseek-ai/deepseek-moe-16b-base` can load on the current server, whether GPU memory is sufficient, and whether the model can complete a short text generation run.

This stage does not implement WikiText perplexity, LoRA/QLoRA training, p-bit router modification, WandB logging, distributed training, model conversion, or quantized loading.

## Project Layout

```text
.
|-- AGENTS.md
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- download_config.yaml
|   `-- generation_config.yaml
|-- models/
|   `-- .gitkeep
|-- scripts/
|   |-- check_env.py
|   |-- download_assets.py
|   |-- test_generate.py
|   `-- inspect_model.py
|-- src/
|   |-- __init__.py
|   |-- gpu_utils.py
|   |-- model_utils.py
|   |-- logging_utils.py
|   `-- generation_utils.py
`-- outputs/
    `-- .gitkeep
```

## Installation

Preferred workflow on the managed server:

```bash
source mirror.sh
hc-new deepseek_moe python=3.10
hc-activate deepseek_moe

# Optional CUDA 12.8 PyTorch wheels.
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Standard conda fallback:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe

# Optional CUDA 12.8 PyTorch wheels.
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

The code respects existing Hugging Face variables such as `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE`. It does not overwrite or hard-code cache paths.

## Configuration

The default config is `configs/generation_config.yaml`. Important fields are `model_name`, `dtype`, `device_map`, `prompt`, `max_new_tokens`, and `output_dir`.

The generation script supports these command-line overrides:

```text
--model_name
--dtype
--device_map
--prompt
--max_new_tokens
--output_dir
--offload_folder
```

## Usage

Lightweight environment check:

```bash
python scripts/check_env.py
```

## Online Download Stage

Use this stage when the server has network access. It downloads model assets only; it does not load the model for inference and does not generate text.

Download to the Hugging Face cache controlled by `HF_HOME`:

```bash
python scripts/download_assets.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --mode cache \
  --resume_download
```

Download to a project-local directory:

```bash
python scripts/download_assets.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --mode local \
  --local_dir models/deepseek-moe-16b-base \
  --resume_download
```

The download script respects existing `HF_ENDPOINT`, `HF_HOME`, and `HF_TOKEN`. It prints only whether `HF_TOKEN` is set, never the raw token.

## Offline Inference Stage

After assets are already available in the Hugging Face cache or in a local model directory, run inference without network access by adding `--local_files_only`.

Offline inference from Hugging Face cache:

```bash
python scripts/test_generate.py \
  --local_files_only \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120
```

Offline inference from a project-local model directory:

```bash
python scripts/test_generate.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120
```

Generation with overrides:

```bash
python scripts/test_generate.py \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 120
```

Generation with the optional shell logging helper:

```bash
tlog generate.log python scripts/test_generate.py \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120
```

Model structure inspection. This also loads the model:

```bash
python scripts/inspect_model.py
```

During initial code generation and review, do not run `scripts/test_generate.py` or `scripts/inspect_model.py` unless the user explicitly accepts the model download/loading.

## Output Files

Each run creates a timestamped directory:

```text
outputs/run_YYYYMMDD_HHMMSS/
|-- config_used.yaml
|-- generated.txt
|-- summary.json
|-- device_map.json
`-- model_modules.txt
```

- `config_used.yaml`: final config after command-line overrides.
- `generated.txt`: generated continuation text.
- `summary.json`: run status, timing, token counts, memory snapshots, offload flags, and non-secret environment status.
- `device_map.json`: saved `model.hf_device_map`.
- `model_modules.txt`: model module inspection output from `scripts/inspect_model.py`.

## Judging GPU Memory Sufficiency

Use `summary.json` and `device_map.json` to decide whether the server is adequate for later stages:

- `has_cpu_offload: false` and `has_disk_offload: false`: best case; the model is mainly on GPU.
- `has_cpu_offload: true`: the model may run, but inference can be slow and later experiments may be limited.
- `has_disk_offload: true`: memory is likely insufficient for practical experiments.
- CUDA out-of-memory during loading or generation means the current configuration is not sufficient.

Also compare GPU memory before loading, after loading, and after generation. Multi-GPU runs should show per-GPU allocated, reserved, peak, and total memory.

## Common Errors

CUDA unavailable:

```text
CUDA is not available. DeepSeek-MoE-16B inference is not recommended on CPU.
```

Resolution: use a CUDA-capable node and install a compatible PyTorch build.

CUDA out of memory:

```text
CUDA out of memory
```

Suggestions:

1. Use multiple GPUs with `device_map="auto"`.
2. Add `max_memory` config.
3. Use CPU offload only for loading tests.
4. Try a smaller model first.
5. Consider quantized loading in a later stage.

Model download or access failure:

1. Check network connectivity.
2. Check Hugging Face access to the model.
3. Check whether `HF_ENDPOINT` should be set for the mirror.
4. Check free disk space for the Hugging Face cache.
5. Run `huggingface-cli login` if authenticated access is required.

Disk space problem:

```text
DeepSeek-MoE-16B weights are large. Please ensure sufficient disk space for Hugging Face cache.
```

## Secrets

Never commit secrets. Do not write raw values for `HF_TOKEN`, `WANDB_API_KEY`, Bark keys, API keys, access tokens, or passwords into source files, logs, README examples, or JSON outputs.

Scripts may report only whether secrets are set:

```text
HF_TOKEN: set
WANDB_API_KEY: not set
```

Recommended review checks:

```bash
grep -R "hf_" . --exclude-dir=.git --exclude-dir=outputs || true
grep -R "wandb_" . --exclude-dir=.git --exclude-dir=outputs || true
grep -R "api.day.app" . --exclude-dir=.git --exclude-dir=outputs || true
```

These checks can produce documentation false positives. Raw real keys must not appear.

## Future Stages

- Stage 2: WikiText-2 / WikiText-103 perplexity evaluation.
- Stage 3: LoRA / QLoRA continued pretraining on WikiText.
- Stage 4: Inspect and modify MoE router for hard-forward p-bit-backward routing.
- Stage 5: Compare original router against the p-bit backward surrogate router.

Do not implement these stages in this repository phase.

## Codex Multi-Agent Note

Use the file ownership split in `AGENTS.md`. Agent A owns only `README.md`, `requirements.txt`, `configs/generation_config.yaml`, and `outputs/.gitkeep`. Other agents own the environment utilities, model loading/generation code, model inspection script, and final review.

Allowed lightweight checks during development:

```bash
python -m compileall scripts src
python scripts/check_env.py
```

Do not run commands that download or load `deepseek-ai/deepseek-moe-16b-base` unless the user explicitly requests it.
