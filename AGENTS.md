# AGENTS.md

## Project: DeepSeek-MoE-16B-Base Inference Probe

This repository started as **Stage 1** of a larger research project on p-bit-assisted MoE routing.

Stage 1 verifies whether `deepseek-ai/deepseek-moe-16b-base` can be downloaded, loaded on the current server, whether GPU memory is sufficient, and whether the model can complete a simple text generation task.

The current codebase may also contain framework scaffolds for Stage 1.5 through Stage 5. These scaffolds must be safe-by-default: they may define configs, scripts, utilities, and patch plans, but agents must not run commands that download DeepSeek-MoE-16B, load the large model, evaluate WikiText, train LoRA/QLoRA, or patch the real model unless the user explicitly asks for that runtime experiment.

This file is intended for Codex / coding agents. All agents must follow it.

---

# 0. High-Level Research Context

The long-term research goal is:

```text
Hard-forward MoE routing + p-bit backward surrogate routing
```

The long-term plan is:

1. Stage 1: DeepSeek-MoE-16B-Base loading, inference, text generation, GPU memory inspection.
2. Stage 2: WikiText-2 / WikiText-103 perplexity evaluation.
3. Stage 3: LoRA / QLoRA continued pretraining on WikiText.
4. Stage 4: Inspect and modify MoE router for hard-forward p-bit-backward routing.
5. Stage 5: Compare original router vs p-bit backward router.

Current repository task:

```text
Implement Stage 1 plus safe code/config scaffolds for Stage 1.5, Stage 2, Stage 3, Stage 4, and Stage 5.
```

Do **not** execute Stage 2/3/4/5 experiments in automated development. Implement only runnable scripts, configs, utilities, and documentation unless the user explicitly requests a real run.

---

# 1. Stage 1 Main Goals

Build a complete runnable Python project that can:

1. Load `deepseek-ai/deepseek-moe-16b-base` from Hugging Face.
2. Use `AutoTokenizer` and `AutoModelForCausalLM`.
3. Prefer BF16 inference.
4. Use `device_map="auto"` for automatic model placement.
5. Generate a short English continuation from a prompt.
6. Record GPU memory before loading, after loading, and after generation.
7. Print and save `model.hf_device_map`.
8. Detect whether CPU or disk offload occurred.
9. Measure generation speed in tokens/second.
10. Save results to `outputs/run_YYYYMMDD_HHMMSS/summary.json`.
11. Save generated text to `generated.txt`.
12. Save used config to `config_used.yaml`.
13. Save device map to `device_map.json`.
14. Provide clear error messages and suggestions if loading or generation fails.
15. Keep the code modular so later stages can add WikiText perplexity, LoRA/QLoRA, and p-bit router experiments.

---

# 2. Strict Stage 1 Limitations

Stage 1 is only for:

- environment check;
- explicit model asset download through `scripts/download_assets.py`;
- DeepSeek-MoE-16B-Base loading;
- text generation;
- GPU memory inspection;
- device map inspection;
- model module inspection.

Do not execute during automated development:

- WikiText perplexity evaluation;
- LoRA / QLoRA training;
- DeepSeek model loading for p-bit router modification;
- WandB online logging;
- automatic phone notification;
- distributed training;
- model conversion;
- quantized loading unless explicitly requested later.

During the initial code-generation pass, do **not** run commands that trigger large model download or model loading.

Allowed lightweight checks:

```bash
python -m compileall scripts src
```

Do not run:

```bash
python scripts/download_assets.py
python scripts/test_generate.py
python scripts/inspect_model.py
python scripts/eval_wikitext_ppl.py
python scripts/train_lora.py
python scripts/train_qlora.py
python scripts/inspect_router.py
```

unless the user explicitly asks to run them and accepts model download/loading.

Download-stage additions:

- `scripts/download_assets.py` is the only script that should perform network model asset download.
- It must not run inference or call `AutoModelForCausalLM.from_pretrained`.
- It must support downloading `deepseek-ai/deepseek-moe-16b-base` to Hugging Face cache or `models/deepseek-moe-16b-base`.
- It must respect `HF_ENDPOINT`, `HF_HOME`, and `HF_TOKEN`, and must never print raw tokens.
- Offline inference should use `scripts/test_generate.py --local_files_only`.

---

# 3. Server Environment Assumptions

The user runs experiments on a managed compute platform with a project startup script that defines environment variables and shell helpers.

## 3.1 Project Root

The user startup script may define:

```bash
export HROOT="$(cd $(dirname ${BASH_SOURCE[0]}); cd ..; pwd)"
```

Rules:

- Do not hard-code absolute paths.
- Place generated project files under the current working project directory.
- Use relative paths whenever possible.
- If a path needs to depend on the user's environment, read it from environment variables such as `HROOT`, `HF_HOME`, or command-line arguments.

## 3.2 Conda Environment Location

The user's startup script may store conda environments under:

```bash
$HROOT/envs
```

The user may have helper aliases/functions:

```bash
hc-new
hc-clone
hc-remove
hc-activate <env_name>
```

When writing README instructions, prefer the user's helper workflow first:

```bash
source mirror.sh
hc-new deepseek_moe python=3.10
hc-activate deepseek_moe
```

If these helpers are unavailable, provide standard conda fallback commands:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe
```

## 3.3 Hugging Face Mirror and Cache

The user's environment may define:

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=<user-defined Hugging Face cache directory>
```

Rules:

- Code must respect existing Hugging Face environment variables.
- Do not overwrite `HF_HOME`, `HF_ENDPOINT`, `HF_HUB_CACHE`, or `HF_DATASETS_CACHE` inside Python code.
- Do not hard-code the Hugging Face cache path.
- When printing environment information, report the current values of:
  - `HF_ENDPOINT`
  - `HF_HOME`
  - `HF_HUB_CACHE`
  - `HF_DATASETS_CACHE`
- If these variables are not set, print `not set`.

## 3.4 Secret Handling

Never write secrets into source code, README, logs, JSON summaries, or this file.

The shell environment may contain sensitive variables such as:

```bash
HF_TOKEN
WANDB_API_KEY
```

Rules:

- Only read these from `os.environ` if needed.
- Do not print raw values.
- Do not save raw values to `summary.json`.
- Do not include raw values in exceptions or logs.
- If logging their existence, only print:
  - `HF_TOKEN: set` or `HF_TOKEN: not set`
  - `WANDB_API_KEY: set` or `WANDB_API_KEY: not set`

Never include Hugging Face tokens, WandB keys, Bark keys, API keys, access tokens, or passwords in generated files.

Agents must check for accidental secret leakage before finalizing changes.

Recommended checks:

```bash
grep -R "hf_" . --exclude-dir=.git --exclude-dir=outputs || true
grep -R "wandb_" . --exclude-dir=.git --exclude-dir=outputs || true
grep -R "api.day.app" . --exclude-dir=.git --exclude-dir=outputs || true
```

These commands may produce false positives in documentation examples. Raw real keys must never appear.

## 3.5 WandB

Stage 1 is inference-only and does not require WandB.

Rules:

- Do not import WandB.
- Do not initialize WandB.
- Do not log to WandB.
- Do not save `WANDB_API_KEY`.

Future-stage preference:

- For Stage 2 evaluation, WandB may be used when useful for comparing repeated perplexity runs, but local JSON/CSV outputs must still be saved.
- For Stage 3 LoRA / QLoRA training, prefer using WandB to track loss, learning rate, evaluation metrics, runtime, GPU memory summaries, checkpoint names, and experiment config.
- For Stage 4/5 router experiments, prefer using WandB to compare baseline router vs p-bit surrogate-router variants across repeated runs.
- Future scripts that use WandB must read `WANDB_BASE_URL` and `WANDB_API_KEY` from the environment only.
- Future scripts must never print, save, or commit raw `WANDB_API_KEY`.
- WandB support in future stages should be optional and controlled by a flag such as `--use_wandb`; scripts must still run without WandB.

## 3.6 PyTorch / CUDA Install Preference

The user may use CUDA 12.8 PyTorch wheels:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
```

However, generated code should not assume a specific CUDA wheel.

README should include both:

Generic installation:

```bash
pip install -r requirements.txt
```

Optional CUDA 12.8 installation:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 3.7 Logging Helper

The user's shell may define:

```bash
tlog logfile command...
```

README examples may include:

```bash
tlog generate.log python scripts/test_generate.py
```

Python code must not depend on this shell function.

## 3.8 Loguru

Stage 1 does not require Loguru. Prefer the existing structured files (`summary.json`, `generated.txt`, `device_map.json`) plus shell `tlog` for Stage 1.

Future-stage preference:

- For Stage 2 evaluation, prefer Loguru for progress, per-dataset split status, recoverable data errors, and evaluation summaries.
- For Stage 3 training, prefer Loguru for startup environment summaries, checkpoint events, periodic metric snapshots, and exception traces.
- For Stage 4/5 router experiments, prefer Loguru for module discovery, router replacement decisions, routing statistics, and comparison run metadata.
- Loguru logs should be written under the current run directory, for example `outputs/run_YYYYMMDD_HHMMSS/run.log`.
- Loguru must not log raw secrets, tokens, API keys, or full environment dumps.
- Loguru support should complement structured JSON outputs, not replace them.

## 3.9 Phone Notification Helper

The user's shell may define a `bark` function for phone notifications.

Rules:

- Do not call it from Python code by default.
- Do not include the Bark key in any generated code or documentation.
- If adding notification support in a future stage, make it optional and controlled by a command-line flag such as `--notify`.

---

# 4. Codex Multi-Agent Operating Mode

The user may ask Codex to use subagents / multiple coding agents.

Use a **main-agent plus subagent** workflow.

## 4.1 Main Agent Responsibilities

The main agent must:

1. Read this `AGENTS.md`.
2. Summarize the project goal, prohibitions, and acceptance criteria before making changes if asked.
3. Divide work among subagents.
4. Ensure subagents do not overwrite each other's files.
5. Wait for all subagents to finish.
6. Merge and review results.
7. Run only lightweight checks unless the user explicitly allows model loading.
8. Provide a final summary:
   - what each agent completed;
   - modified file list;
   - tests run;
   - remaining risks;
   - commands the user should run manually.

## 4.2 Recommended Subagent Split

Use 5 subagents for Stage 1.

### Agent A: Project Skeleton and README

Allowed files:

```text
README.md
requirements.txt
configs/download_config.yaml
configs/generation_config.yaml
models/.gitkeep
outputs/.gitkeep
```

Responsibilities:

- Create or verify directory structure.
- Generate `requirements.txt`.
- Generate default YAML configs.
- Generate README with installation, usage, outputs, common errors, future stages.
- Include both `hc-new/hc-activate` workflow and standard conda fallback.
- Do not implement model loading logic.

### Agent B: Environment and GPU Utilities

Allowed files:

```text
scripts/check_env.py
src/gpu_utils.py
```

Responsibilities:

- Implement CUDA/GPU/BF16 environment checks.
- Print Python, PyTorch, Transformers, CUDA, GPU memory, disk space.
- Print `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`.
- Print only whether `HF_TOKEN` and `WANDB_API_KEY` are set.
- Never print raw secrets.

### Agent C: Model Loading and Text Generation

Allowed files:

```text
scripts/download_assets.py
scripts/test_generate.py
src/model_utils.py
src/generation_utils.py
src/logging_utils.py
```

Responsibilities:

- Load tokenizer.
- Implement download-only asset acquisition in `scripts/download_assets.py`.
- Load `AutoModelForCausalLM`.
- Support BF16, FP16, FP32.
- Support `device_map="auto"`.
- Support optional `offload_folder`.
- Support `local_files_only` for offline inference.
- Save `summary.json`, `generated.txt`, `device_map.json`, `config_used.yaml`.
- Detect CPU/disk offload.
- Catch and summarize errors.
- Do not implement training.

### Agent D: Model Structure Inspection

Allowed files:

```text
scripts/inspect_model.py
```

Responsibilities:

- Load model only when the user manually runs the script.
- Print model class name.
- Count total and trainable parameters.
- Print top-level modules.
- Search for module names containing:
  - `moe`
  - `gate`
  - `router`
  - `expert`
- Save output to `outputs/run_YYYYMMDD_HHMMSS/model_modules.txt`.
- Do not modify the model.

### Agent E: Review, Safety, and Lightweight Tests

Allowed files:

```text
No primary implementation files unless fixing small issues after review.
```

Responsibilities:

- Review all generated files for compliance with this `AGENTS.md`.
- Check for raw secret leakage.
- Check for hard-coded absolute paths.
- Check that Stage 1 does not implement Stage 2/3/4 functionality.
- Run lightweight checks only:
  - `python -m compileall scripts src`
- Do not run commands that download or load DeepSeek-MoE-16B unless the user explicitly asks.

## 4.3 File Ownership Rule

Subagents must not edit files owned by other agents unless the main agent explicitly asks them to fix a specific issue.

If two agents need the same file, the main agent must coordinate the change.

## 4.4 Large Model Execution Rule

During development, agents must not trigger large downloads or large model loading.

Do not run:

```python
AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
```

as part of automated testing unless the user explicitly instructs this run.

The final README may instruct the user to run:

```bash
python scripts/test_generate.py
```

but agents should not run it automatically in the first code-generation pass.

---

# 5. Required Directory Structure

Generate the project as:

```text
deepseek_moe_probe/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── download_config.yaml
│   └── generation_config.yaml
├── models/
│   └── .gitkeep
├── scripts/
│   ├── check_env.py
│   ├── download_assets.py
│   ├── test_generate.py
│   └── inspect_model.py
├── src/
│   ├── __init__.py
│   ├── gpu_utils.py
│   ├── model_utils.py
│   ├── logging_utils.py
│   └── generation_utils.py
└── outputs/
    └── .gitkeep
```

Rules:

- Do not hard-code absolute paths.
- All paths should be configurable or relative to the project root.
- Keep the code simple and runnable.
- Do not introduce complex frameworks.

---

# 6. Dependencies

`requirements.txt` must include:

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
```

Optional future dependencies may be commented out:

```text
# bitsandbytes
# peft
# wandb
```

Do not require optional dependencies in Stage 1.

---

# 7. Default Config

Create `configs/generation_config.yaml`:

```yaml
model_name: "deepseek-ai/deepseek-moe-16b-base"

dtype: "bf16"
device_map: "auto"
trust_remote_code: true
low_cpu_mem_usage: true

prompt: "The history of artificial intelligence can be traced back to"
max_new_tokens: 120
do_sample: true
temperature: 0.7
top_p: 0.9
repetition_penalty: 1.05

output_dir: "outputs"
```

The main script must support command-line overrides for important fields:

- `--model_name`
- `--dtype`
- `--device_map`
- `--prompt`
- `--max_new_tokens`
- `--output_dir`
- `--offload_folder`
- `--local_files_only`

---

# 8. Script Requirements

## 8.1 `scripts/check_env.py`

This script checks whether the environment is suitable for running DeepSeek-MoE-16B inference.

It must print:

1. Python version.
2. PyTorch version.
3. Transformers version.
4. CUDA availability.
5. CUDA version.
6. Number of GPUs.
7. GPU name for each GPU.
8. Total GPU memory for each GPU.
9. Allocated, reserved, and peak CUDA memory for each GPU.
10. BF16 support status.
11. Current working directory.
12. Hugging Face environment variables:
    - `HF_ENDPOINT`
    - `HF_HOME`
    - `HF_HUB_CACHE`
    - `HF_DATASETS_CACHE`
13. Whether `HF_TOKEN` is set, without printing the token.
14. Whether `WANDB_API_KEY` is set, without printing the key.
15. Free disk space for the current directory and Hugging Face cache if available.

If CUDA is unavailable, do not crash. Print:

```text
CUDA is not available. DeepSeek-MoE-16B inference is not recommended on CPU.
```

Run command:

```bash
python scripts/check_env.py
```

## 8.2 `scripts/download_assets.py`

This script is the only Stage 1 entry point that should perform network model asset downloads.

It must:

1. Read `configs/download_config.yaml` when present.
2. Support downloading `deepseek-ai/deepseek-moe-16b-base` to the Hugging Face cache.
3. Support downloading to `models/deepseek-moe-16b-base`.
4. Support command-line arguments:
   - `--model_name`
   - `--local_dir`
   - `--mode cache/local`
   - `--resume_download`
5. Respect `HF_ENDPOINT`, `HF_HOME`, and `HF_TOKEN`.
6. Never print raw `HF_TOKEN` or other secrets.
7. Never call `AutoModelForCausalLM.from_pretrained`.
8. Never run inference or text generation.

Example cache download:

```bash
python scripts/download_assets.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --mode cache \
  --resume_download
```

Example local download:

```bash
python scripts/download_assets.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --mode local \
  --local_dir models/deepseek-moe-16b-base \
  --resume_download
```

Agents must not run this command during automated development unless the user explicitly asks and accepts network download.

## 8.3 `scripts/test_generate.py`

This is the main script.

It must:

1. Read `configs/generation_config.yaml`.
2. Apply command-line overrides.
3. Create a timestamped run directory under `outputs/`.
4. Save the final config to `config_used.yaml`.
5. Print environment summary.
6. Record GPU memory before model loading.
7. Load tokenizer.
8. Load model.
9. Record GPU memory after loading.
10. Save `model.hf_device_map` to `device_map.json`.
11. Detect CPU/disk offload.
12. Generate text from the configured prompt.
13. Record GPU memory after generation.
14. Save generated text to `generated.txt`.
15. Save a full result summary to `summary.json`.
16. Print the generated text and key metrics to the terminal.
17. If any error occurs, save a failure `summary.json` with error type, error message, and suggestions.
18. Support `--local_files_only` for offline inference from Hugging Face cache or a local model directory.

Basic run:

```bash
python scripts/test_generate.py
```

Example with overrides:

```bash
python scripts/test_generate.py \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --dtype bf16 \
  --device_map auto \
  --prompt "The future of machine learning is" \
  --max_new_tokens 120
```

Example with shell logging helper:

```bash
tlog generate.log python scripts/test_generate.py \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120
```

Offline example:

```bash
python scripts/test_generate.py \
  --local_files_only \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120
```

## 8.4 `scripts/inspect_model.py`

This script loads the model and inspects its module structure.

It must print and save:

1. Model class name.
2. Total parameter count.
3. Trainable parameter count.
4. Top-level module names.
5. All module names containing any of:
   - `moe`
   - `gate`
   - `router`
   - `expert`

Save output to:

```text
outputs/run_YYYYMMDD_HHMMSS/model_modules.txt
```

Do not modify the model.

Run command:

```bash
python scripts/inspect_model.py
```

---

# 9. Source Module Requirements

## 9.1 `src/gpu_utils.py`

Implement:

```python
def get_gpu_info() -> list[dict]:
    """
    Return information for every GPU:
    name, total memory, allocated memory, reserved memory, and peak allocated memory.
    Values should be in GB.
    """

def print_gpu_memory(stage: str) -> None:
    """
    Print GPU memory usage for a named stage.
    Example stages: before_loading, after_loading, after_generation.
    """

def reset_peak_memory_stats() -> None:
    """
    Reset PyTorch CUDA peak memory statistics.
    """

def get_cuda_summary() -> dict:
    """
    Return CUDA availability, GPU count, CUDA version, BF16 support, and related information.
    """
```

Rules:

- Must not crash if CUDA is unavailable.
- Must support multiple GPUs.
- Use GB as the memory unit.

## 9.2 `src/model_utils.py`

Implement:

```python
def resolve_dtype(dtype_name: str):
    """
    Support bf16, fp16, and fp32.
    bf16 -> torch.bfloat16
    fp16 -> torch.float16
    fp32 -> torch.float32
    """

def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    """
    Load tokenizer.
    If pad_token_id is missing, try to set it to eos_token_id.
    """

def load_causal_lm(
    model_name: str,
    dtype_name: str = "bf16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
    max_memory: dict | None = None,
    offload_folder: str | None = None,
    local_files_only: bool = False,
):
    """
    Load AutoModelForCausalLM and return the model.
    """
```

Model loading should use:

```python
AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=resolved_dtype,
    device_map=device_map,
    trust_remote_code=trust_remote_code,
    low_cpu_mem_usage=low_cpu_mem_usage,
    max_memory=max_memory,
    offload_folder=offload_folder,
    local_files_only=local_files_only,
)
```

Rules:

- Catch CUDA OOM errors.
- Give clear suggestions if loading fails.
- Print `model.hf_device_map` if available.
- Warn if device map contains `cpu` or `disk`.
- Do not print or save raw tokens.

## 9.3 `src/generation_utils.py`

Implement:

```python
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> dict:
    """
    Generate text and return:
    {
        "prompt": ...,
        "generated_text": ...,
        "input_tokens": ...,
        "output_tokens": ...,
        "new_tokens": ...,
        "generation_time_sec": ...,
        "tokens_per_sec": ...
    }
    """
```

Rules:

- Use `torch.no_grad()`.
- Automatically move input tensors to the model's device.
- If `model.device` exists, use it.
- Otherwise use `next(model.parameters()).device`.
- Decode with `skip_special_tokens=True`.
- Return generation speed.

## 9.4 `src/logging_utils.py`

Implement:

```python
def create_run_dir(base_dir: str = "outputs") -> str:
    """
    Create outputs/run_YYYYMMDD_HHMMSS and return its path.
    """

def save_json(obj: dict, path: str) -> None:
    """
    Save JSON with ensure_ascii=False and indent=2.
    """

def save_text(text: str, path: str) -> None:
    """
    Save plain text.
    """
```

---

# 10. Required `summary.json`

On success, save at least:

```json
{
  "model_name": "deepseek-ai/deepseek-moe-16b-base",
  "dtype": "bf16",
  "device_map": "auto",
  "load_success": true,
  "generation_success": true,
  "prompt": "...",
  "generated_text": "...",
  "input_tokens": 10,
  "output_tokens": 130,
  "new_tokens": 120,
  "generation_time_sec": 12.3,
  "tokens_per_sec": 9.75,
  "gpu_info_before_loading": [],
  "gpu_info_after_loading": [],
  "gpu_info_after_generation": [],
  "hf_device_map": {},
  "has_cpu_offload": false,
  "has_disk_offload": false,
  "hf_endpoint": "set or not set",
  "hf_home": "set or not set",
  "hf_token_is_set": true,
  "wandb_api_key_is_set": true
}
```

Rules:

- Do not save raw `HF_TOKEN`.
- Do not save raw `WANDB_API_KEY`.
- Do not save raw Bark key or any other secret.
- It is acceptable to save whether a secret is set as a boolean.

On failure, save at least:

```json
{
  "load_success": false,
  "generation_success": false,
  "error_type": "...",
  "error_message": "...",
  "suggestion": "..."
}
```

---

# 11. Offload Detection

If any entry in `model.hf_device_map` is `"cpu"`, set:

```json
"has_cpu_offload": true
```

If any entry is `"disk"`, set:

```json
"has_disk_offload": true
```

Interpretation:

- `has_cpu_offload=false` and `has_disk_offload=false`: good, model is mainly on GPU.
- `has_cpu_offload=true`: model can run but may be slow.
- `has_disk_offload=true`: memory is insufficient for practical experiments.

---

# 12. Error Handling

Handle these cases clearly.

## 12.1 CUDA Unavailable

Print:

```text
CUDA is not available. DeepSeek-MoE-16B inference is not recommended on CPU.
```

## 12.2 CUDA Out of Memory

If an exception contains `CUDA out of memory`, save a suggestion:

```text
Suggestions:
1. Use multiple GPUs with device_map="auto".
2. Add max_memory config.
3. Use CPU offload only for loading tests.
4. Try a smaller model first.
5. Consider quantized loading in a later stage.
```

## 12.3 Model Download Failure

Suggest checking:

```text
1. Network connection.
2. Hugging Face access.
3. Whether HF_ENDPOINT should be set.
4. Disk space.
5. Whether huggingface-cli login is required.
```

## 12.4 Disk Space Problem

Print:

```text
DeepSeek-MoE-16B weights are large. Please ensure sufficient disk space for Hugging Face cache.
```

---

# 13. README Requirements

`README.md` must include:

1. Project purpose.
2. Statement that this stage is inference only.
3. Installation instructions using the user's `hc-new` / `hc-activate` workflow first.
4. Standard conda fallback instructions.
5. Optional CUDA 12.8 PyTorch installation command.
6. Environment check command.
7. Text generation command.
8. Model inspection command.
9. Explanation of output files.
10. Explanation of how to judge whether GPU memory is sufficient.
11. Common errors and solutions.
12. Future stages.
13. A reminder not to commit secrets.
14. A short note on Codex multi-agent usage.

Recommended installation section:

```bash
# Load user's environment startup script first, if available.
source mirror.sh

# Create environment using user helper.
hc-new deepseek_moe python=3.10
hc-activate deepseek_moe

# Optional: install PyTorch CUDA 12.8 wheel.
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies.
pip install -r requirements.txt
```

Fallback:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe
pip install -r requirements.txt
```

Run examples:

```bash
python scripts/check_env.py

python scripts/test_generate.py

python scripts/test_generate.py \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 120

tlog generate.log python scripts/test_generate.py \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120

python scripts/inspect_model.py
```

Expected output directory:

```text
outputs/run_YYYYMMDD_HHMMSS/
├── config_used.yaml
├── generated.txt
├── summary.json
├── device_map.json
└── model_modules.txt
```

---

# 14. Future TODO Comments

Add clear TODO comments in relevant scripts:

```python
# TODO Stage 2:
# Add WikiText-2 perplexity evaluation.

# TODO Stage 3:
# Add LoRA / QLoRA continued pretraining.

# TODO Stage 4:
# Inspect and modify MoE router for hard-forward p-bit-backward routing.
```

Do not implement these in the current stage.

---

# 15. Acceptance Criteria

The project is accepted if:

1. `python scripts/check_env.py` runs successfully.
2. `python scripts/test_generate.py` runs successfully when hardware is sufficient.
3. The generated text is printed.
4. `summary.json` is saved.
5. `generated.txt` is saved.
6. `device_map.json` is saved.
7. Failure cases also save an error summary.
8. Multi-GPU memory information is shown correctly.
9. No absolute paths are hard-coded.
10. All major functions have docstrings.
11. README is complete enough for a new user to run the project.
12. The code is complete and runnable, not pseudocode.
13. No raw secret is printed or saved.
14. Stage 1 does not include training, WikiText evaluation, or p-bit router changes.
15. Multi-agent file ownership rules were followed.

---

# 16. Recommended First Run for the User

After code generation, the user may run:

```bash
cd deepseek_moe_probe

source mirror.sh

hc-new deepseek_moe python=3.10
hc-activate deepseek_moe

pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py

tlog generate.log python scripts/test_generate.py \
  --prompt "The history of artificial intelligence can be traced back to" \
  --max_new_tokens 120

python scripts/inspect_model.py
```

If the helper functions are unavailable, use:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe
pip install -r requirements.txt
```

---

# 17. Suggested Codex Multi-Agent Prompt

The user may paste this into Codex:

```text
Please read AGENTS.md first. Use subagents for multi-agent development.

You are the main agent. Start 5 subagents and assign them as follows:

Agent A: Project Skeleton and README
- Only modify README.md, requirements.txt, configs/generation_config.yaml, outputs/.gitkeep.

Agent B: Environment and GPU Utilities
- Only modify scripts/check_env.py and src/gpu_utils.py.

Agent C: Model Loading and Text Generation
- Only modify scripts/test_generate.py, src/model_utils.py, src/generation_utils.py, src/logging_utils.py.

Agent D: Model Structure Inspection
- Only modify scripts/inspect_model.py.

Agent E: Review and Lightweight Tests
- Do not write primary functionality. Review files, check for secrets, check no hard-coded absolute paths, check Stage 1 boundaries, and run only lightweight tests.

Do not run commands that download or load deepseek-ai/deepseek-moe-16b-base.
Do not run scripts/test_generate.py.
Do not run scripts/inspect_model.py.
Allowed tests:
- python -m compileall scripts src

Wait for all subagents to complete, then summarize:
1. What each agent completed.
2. Modified files.
3. Tests run and results.
4. Any risks or manual steps.
5. Exact commands I should run next.
```

---

# 18. Experiment Conclusion Template

After a successful run, the project should provide enough information to write:

```text
已完成 DeepSeek-MoE-16B-Base 的文本生成推理测试。模型能够通过 Hugging Face Transformers 的 AutoModelForCausalLM 接口加载，并在 BF16 精度下完成文本生成。通过 device_map 与 CUDA 显存统计，确认当前硬件是否能够支持后续 WikiText perplexity 评估。若模型全部部署在 GPU 上且无 CPU/disk offload，则进入下一阶段；若出现 CPU/disk offload 或 CUDA OOM，则需要采用多卡、量化或更小模型进行替代实验。
```
