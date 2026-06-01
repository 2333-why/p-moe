# AGENTS.md

## Project: p-MoE — Load-Aware p-bit Competitive Routing for Mixture-of-Experts

This repository is a research codebase for **Load-Aware p-bit Competitive Routing for Sparse Mixture-of-Experts Language Models**.

The project originally started as a DeepSeek-MoE-16B inference probe, but the current goal is broader: build a full research framework for a top-conference-level p-MoE paper.

The central research idea is:

```text
Keep hard top-k sparse MoE routing in the forward pass,
use p-bit surrogate gradients in the backward pass,
and add load-aware expert competition to reduce router non-differentiability,
expert collapse, load imbalance, and under-trained experts.
```

This file is intended for Codex / coding agents. All agents must follow it.

---

# 0. Current Implementation and Review Rule

For implementation passes, agents should primarily write code, configs, scripts, and documentation.

The repository may not yet contain downloaded DeepSeek-MoE weights or cached
WikiText datasets. Therefore, agents must not require successful large-model
runtime execution as a condition for code review.

Allowed lightweight checks during review:

```bash
python -m compileall scripts src
python scripts/check_env.py
python scripts/test_pbit_router_unit.py
python <script>.py --help
```

Allowed static checks:

```bash
grep/rg/Select-String scans for secrets, absolute paths, and risky calls
read-only file inspection
AST/syntax inspection that does not import or execute project modules
```

Do **not** download models.
Do **not** load DeepSeek-MoE-16B.
Do **not** evaluate WikiText.
Do **not** train LoRA/QLoRA.
Do **not** train mini-MoE unless explicitly requested.
Do **not** run router inspection against DeepSeek unless explicitly requested.
Do **not** patch the real DeepSeek model.

The user will manually run large experiments later on the GPU server after
assets have been downloaded.

Never call this during development unless the user explicitly requests a runtime experiment:

```python
AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
```

---

# 1. High-Level Research Context

## 1.1 Core Method

Traditional MoE routing:

```text
scores = router(x)
mask_hard = TopK(scores, k)
y = sum_i mask_i * gate_i * expert_i(x)
```

p-MoE keeps the **hard top-k forward** for sparse computation, but replaces the non-differentiable router backward path with a p-bit surrogate.

Each expert corresponds to a p-bit:

```text
z_i in {0, 1}
```

where `z_i = 1` means expert `i` is activated.

The p-bit local field is:

```text
I_i = s_i - alpha * sum_{j != i} z_j + beta * (1/N - L_i)
```

where:

- `s_i`: router score for expert `i`;
- `alpha`: expert competition strength;
- `L_i`: historical expert load;
- `beta`: load-aware bias strength;
- `N`: number of experts.

The p-bit activation probability is:

```text
q_i = sigmoid(I_i / T)
```

The straight-through mask is:

```text
m = m_hard.detach() - q.detach() + q
```

Forward uses `m_hard`; backward uses gradients from `q`.

## 1.2 Load-Aware Expert Under-Training Fix

Under-trained experts occur because experts not selected in forward receive no task gradient.

p-MoE addresses this by maintaining an exponential moving average of expert load:

```text
L_i(t) = mu * L_i(t-1) + (1 - mu) * f_i(t)
```

and applying load-aware bias:

```text
b_i = beta * (1/N - L_i)
```

Low-load experts get positive bias and are more likely to enter the forward path; overused experts get negative bias.

The key claim:

```text
p-bit backward improves router gradient;
load-aware p-bit bias increases forward activation opportunities for under-trained experts.
```

## 1.3 Two Experimental Lines

This repository must support two complementary lines:

### Line A: Real-model validation with DeepSeek-MoE-16B

Purpose:

- model loading;
- text generation;
- WikiText PPL baseline;
- router/expert inspection;
- expert load statistics;
- safe patch scaffold only.

Do **not** blindly patch DeepSeek-MoE. Always inspect first.

### Line B: Controllable mini-MoE innovation validation

Purpose:

- implement a small MiniGPT/Mini-MoE model;
- compare standard top-k, noisy top-k, Gumbel, ST, and p-bit routing;
- measure PPL, load variance, dead expert ratio, entropy, router gradients, expert gradients;
- this is the main algorithmic validation line for a paper.

---

# 2. Project Stages

The full repository should support these stages:

```text
Stage 1:   DeepSeek-MoE-16B loading, generation, GPU memory inspection.
Stage 1.5: Separate online model asset download from offline inference.
Stage 2:   WikiText-2 / WikiText-103 causal LM perplexity evaluation.
Stage 3:   LoRA / QLoRA continued pretraining framework.
Stage 4:   p-bit router core module and safe DeepSeek router patch scaffold.
Stage 5:   controllable mini-MoE experiments for algorithmic validation.
Stage 6:   DeepSeek-MoE router inspection, expert load statistics, patch planning.
Stage 7:   experiment orchestration and result collection for paper metrics.
```

Current implementation pass:

```text
Implement code and configs for all stages, but do not run anything.
```

---

# 3. Server Environment Assumptions

The user runs experiments on a managed compute platform with a project startup script.

## 3.1 Project Root

The startup script may define:

```bash
export HROOT="$(cd $(dirname ${BASH_SOURCE[0]}); cd ..; pwd)"
```

Rules:

- Do not hard-code absolute paths.
- Use relative paths by default.
- Read `HROOT`, `HF_HOME`, `HF_ENDPOINT`, and user-provided CLI paths from environment variables or arguments when needed.
- All generated outputs should default to `outputs/`.
- All model local directories should default to `models/`.

## 3.2 Conda Environment

The user may use helper commands:

```bash
hc-new deepseek_moe python=3.10
hc-activate deepseek_moe
```

README should also include standard fallback:

```bash
conda create -n deepseek_moe python=3.10 -y
conda activate deepseek_moe
```

## 3.3 Hugging Face Mirror and Cache

The environment may define:

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=<large Hugging Face cache directory>
```

Rules:

- Respect existing `HF_ENDPOINT`, `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`.
- Do not overwrite these variables inside Python code.
- Do not hard-code Hugging Face cache path.
- Print environment values only when requested by environment-check scripts.

## 3.4 Secret Handling

Never write secrets into source code, README, logs, JSON summaries, or this file.

Sensitive variables may exist:

```bash
HF_TOKEN
WANDB_API_KEY
```

Rules:

- Do not print raw values.
- Do not save raw values.
- Only report `set` / `not set` or boolean flags.
- Never include Hugging Face tokens, WandB keys, Bark keys, API keys, passwords, or access tokens in generated files.

## 3.5 WandB

Default: disabled.

Rules:

- Do not import WandB unless `--use_wandb` is explicitly passed.
- If used in later training scripts, read keys from environment variables only.
- Do not print or save raw keys.

## 3.6 CUDA / PyTorch

README may include optional CUDA 12.8 installation:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Do not assume a specific CUDA wheel in code.

## 3.7 Shell Helpers

The user may have:

```bash
tlog logfile command...
bark "message" "title"
```

Python code must not depend on these helpers.

Do not call `bark` from Python by default.

---

# 4. Codex Multi-Agent Operating Mode

If the user asks for multi-agent development, use a main-agent plus subagent workflow.

## 4.1 Main Agent Responsibilities

The main agent must:

1. Read this `AGENTS.md` first.
2. Summarize the project goal and restrictions if asked.
3. Divide tasks among subagents.
4. Prevent subagents from overwriting each other's files.
5. Wait for all subagents to complete.
6. Merge and review results.
7. Run only allowed lightweight/static checks unless the user explicitly allows runtime experiments.
8. Final summary must include:
   - what each agent completed;
   - modified files;
   - which stages are supported;
   - what remains scaffold only;
   - whether any script might accidentally download/load/train;
   - exact manual commands for the user.

## 4.2 Recommended 8-Agent Split

### Agent A: Project architecture, README, and configs

Allowed files:

```text
README.md
requirements.txt
AGENTS.md
configs/*.yaml
outputs/.gitkeep
models/.gitkeep
```

Responsibilities:

- Define the full project structure.
- Explain all stages in README.
- Create/update all default configs:
  - `configs/generation_config.yaml`
  - `configs/download_config.yaml`
  - `configs/eval_wikitext_config.yaml`
  - `configs/train_lora_config.yaml`
  - `configs/train_qlora_config.yaml`
  - `configs/pbit_router_config.yaml`
  - `configs/mini_moe_config.yaml`
  - `configs/result_collect_config.yaml`
- Do not implement core model logic.

### Agent B: DeepSeek Stage 1 / 1.5 utilities

Allowed files:

```text
scripts/check_env.py
scripts/download_assets.py
scripts/test_generate.py
scripts/inspect_model.py
src/gpu_utils.py
src/model_utils.py
src/generation_utils.py
src/logging_utils.py
```

Responsibilities:

- Environment check.
- GPU memory utilities.
- Download assets script.
- Offline inference support.
- Model loading utilities.
- Text generation utilities.
- Model module inspection.
- Support `--local_files_only`.
- Support `--offload_folder`.
- Never print tokens.

### Agent C: WikiText PPL evaluation

Allowed files:

```text
scripts/eval_wikitext_ppl.py
src/eval_utils.py
src/data_utils.py
configs/eval_wikitext_config.yaml
```

Responsibilities:

- Implement causal LM PPL evaluation.
- Support WikiText-2 / WikiText-103.
- Support sliding-window evaluation.
- Support local model loading.
- Save `eval_summary.json`.
- Do not run evaluation.

### Agent D: LoRA / QLoRA continued pretraining framework

Allowed files:

```text
scripts/train_lora.py
scripts/train_qlora.py
src/train_utils.py
src/lora_utils.py
configs/train_lora_config.yaml
configs/train_qlora_config.yaml
requirements.txt
```

Responsibilities:

- Implement training script framework.
- Support PEFT LoRA and QLoRA.
- Optional dependencies only.
- Default WandB disabled.
- Clear missing-dependency errors.
- Do not run training.

### Agent E: p-bit router core and safe patch scaffold

Allowed files:

```text
src/pbit_router.py
src/router_metrics.py
src/router_patch.py
scripts/test_pbit_router_unit.py
scripts/inspect_router.py
configs/pbit_router_config.yaml
```

Responsibilities:

- Implement `PBitRouterConfig` dataclass.
- Implement `PBitLoadState`.
- Implement `PBitBackwardRouter`.
- Implement `top_k_mask`, `sample_top_k_mask`, `straight_through_mask`.
- Implement load-aware bias.
- Implement competition term.
- Implement temperature scheduler.
- Implement router metrics.
- Implement safe patch scaffold.
- Do not patch DeepSeek automatically.

### Agent F: Mini-MoE controllable experiment framework

Allowed files:

```text
src/mini_moe_model.py
src/mini_moe_layers.py
src/mini_moe_train.py
scripts/train_mini_moe.py
scripts/eval_mini_moe.py
configs/mini_moe_config.yaml
```

Responsibilities:

- Implement a small MiniGPT/Mini-MoE model.
- Support dense baseline.
- Support standard top-k MoE.
- Support noisy top-k.
- Support Gumbel / Concrete relaxation.
- Support top-k ST.
- Support p-bit backward.
- Support p-bit + load-aware bias.
- Support warmup strategies.
- Save metrics for paper analysis.
- Do not run training.

### Agent G: Experiment scripts and result collection

Allowed files:

```text
scripts/run_stage1_generate.sh
scripts/run_stage2_eval.sh
scripts/run_stage3_qlora.sh
scripts/run_stage4_router_inspect.sh
scripts/run_stage5_mini_moe.sh
scripts/collect_results.py
src/result_utils.py
configs/result_collect_config.yaml
```

Responsibilities:

- Implement safe shell script templates.
- Use `set -euo pipefail`.
- Do not hard-code absolute paths.
- Default to local files where appropriate.
- Implement result scanning and summary generation.
- Do not run scripts.

### Agent H: Static safety review

Allowed files:

```text
No primary implementation files unless fixing obvious small issues after review.
```

Responsibilities:

- Static and lightweight review only.
- Run only allowed lightweight/static checks from Section 0.
- Check no raw secrets.
- Check no hard-coded absolute paths.
- Check no script auto-downloads or auto-trains at import time.
- Check WandB is disabled by default.
- Check all scripts support `--help`.
- Check DeepSeek patch remains scaffold only.

---

# 5. Required Directory Structure

The repository should support this structure:

```text
p-moe/
├── AGENTS.md
├── README.md
├── requirements.txt
├── configs/
│   ├── generation_config.yaml
│   ├── download_config.yaml
│   ├── eval_wikitext_config.yaml
│   ├── train_lora_config.yaml
│   ├── train_qlora_config.yaml
│   ├── pbit_router_config.yaml
│   ├── mini_moe_config.yaml
│   └── result_collect_config.yaml
├── scripts/
│   ├── check_env.py
│   ├── download_assets.py
│   ├── test_generate.py
│   ├── inspect_model.py
│   ├── eval_wikitext_ppl.py
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── inspect_router.py
│   ├── test_pbit_router_unit.py
│   ├── train_mini_moe.py
│   ├── eval_mini_moe.py
│   ├── collect_results.py
│   ├── run_stage1_generate.sh
│   ├── run_stage2_eval.sh
│   ├── run_stage3_qlora.sh
│   ├── run_stage4_router_inspect.sh
│   └── run_stage5_mini_moe.sh
├── src/
│   ├── __init__.py
│   ├── gpu_utils.py
│   ├── model_utils.py
│   ├── generation_utils.py
│   ├── logging_utils.py
│   ├── data_utils.py
│   ├── eval_utils.py
│   ├── train_utils.py
│   ├── lora_utils.py
│   ├── pbit_router.py
│   ├── router_metrics.py
│   ├── router_patch.py
│   ├── mini_moe_layers.py
│   ├── mini_moe_model.py
│   ├── mini_moe_train.py
│   └── result_utils.py
├── models/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

Rules:

- No absolute paths.
- Keep code simple.
- All major functions need docstrings.
- All scripts should support `--help`.

---

# 6. Dependencies

Base dependencies:

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

Optional dependencies should not be required for Stage 1/2:

```text
# peft
# bitsandbytes
# wandb
# matplotlib
```

If optional dependencies are missing, scripts should print clear installation guidance.

---

# 7. Core p-bit Router Requirements

`src/pbit_router.py` must implement the core innovation.

Required components:

```python
PBitRouterConfig
PBitLoadState
PBitBackwardRouter
top_k_mask
sample_top_k_mask
straight_through_mask
```

The module must support:

- `num_experts`
- `top_k`
- `alpha`
- `beta`
- `temperature`
- `min_temperature`
- `load_ema_decay`
- `use_load_bias`
- `use_competition`
- `use_sampling`
- `eps`

Core equations:

```text
L_i(t) = mu * L_i(t-1) + (1 - mu) * f_i(t)
b_i = beta * (1/N - L_i)
I_i = s_i - alpha * sum_{j != i} z_j + b_i
q_i = sigmoid(I_i / T)
m_hard = TopK(s + b, k) or SampleTopK(q, k)
m = m_hard.detach() - q.detach() + q
```

This module should only produce masks/gates. It must not compute expert outputs directly.

---

# 8. Mini-MoE Requirements

The mini-MoE line is the main controllable algorithm validation path.

It must support:

- dense baseline;
- standard top-k MoE;
- noisy top-k;
- Gumbel-Softmax / Concrete relaxation;
- top-k straight-through;
- p-bit backward;
- p-bit + competition;
- p-bit + load-aware bias;
- p-bit + warmup.

Metrics to save:

- train loss;
- eval PPL;
- expert load variance;
- dead expert ratio;
- expert entropy;
- router gradient norm;
- expert gradient norm;
- routing statistics by layer if available.

---

# 9. DeepSeek-MoE Requirements

DeepSeek-MoE is the real-model validation line.

Stage 1:

- download assets;
- load model;
- generate text;
- inspect memory;
- save `summary.json`.

Stage 2:

- WikiText PPL baseline.

Stage 6:

- inspect router modules;
- save router inspection report;
- build patch plan.

Important:

```text
Do not blindly patch DeepSeek-MoE router.
Run inspect_router.py first, then use inspection results to design patch.
```

`router_patch.py` must be scaffold-only by default.

---

# 10. Result Metrics for Paper

Result collection should support:

- PPL;
- train loss;
- tokens/sec;
- load variance;
- dead expert ratio;
- expert entropy;
- router gradient norm;
- expert gradient norm;
- CPU/disk offload flags;
- error messages.

`collect_results.py` should aggregate:

```text
outputs/run_*/summary.json
outputs/run_*/eval_summary.json
outputs/**/train_summary.json
```

and create:

```text
outputs/results_table.csv
outputs/results_summary.md
```

---

# 11. Script Safety Rules

All scripts must be safe at import time.

Rules:

- No model loading at import time.
- No dataset download at import time.
- No training at import time.
- No WandB initialization at import time.
- All executable logic must be under `if __name__ == "__main__":`.
- `--local_files_only` should be supported for DeepSeek-related scripts where applicable.
- Default configs should prefer local model paths once downloaded.

---

# 12. README Requirements

README must include:

1. Project purpose and research idea.
2. Full stage overview.
3. Environment setup.
4. Download assets workflow.
5. Offline inference workflow.
6. WikiText PPL workflow.
7. LoRA / QLoRA workflow.
8. p-bit router scaffold explanation.
9. Mini-MoE experiment workflow.
10. Result collection workflow.
11. Secret safety warning.
12. GPU memory notes.
13. What scripts are safe to run.
14. What scripts trigger model download/loading/training.
15. Recommended manual experiment order.

---

# 13. Acceptance Criteria

The codebase is acceptable if:

1. It implements all requested files and configs.
2. It does not run anything during implementation.
3. It does not include raw secrets.
4. It does not hard-code absolute paths.
5. All scripts support `--help`.
6. Stage 1 works as DeepSeek inference scaffold.
7. Stage 2 works as WikiText PPL scaffold.
8. Stage 3 works as LoRA/QLoRA scaffold.
9. Stage 4 implements p-bit router core and safe patch scaffold.
10. Stage 5 implements mini-MoE controllable experiment framework.
11. Stage 7 implements result collection.
12. README explains manual usage clearly.

---

# 14. Suggested Prompt for Codex

The user may paste this into Codex:

```text
Please read AGENTS.md first. Implement the full p-MoE research framework described there.
Use subagents according to the 8-agent split.
Only write code, configs, scripts, and README.
Run only lightweight/static checks allowed by AGENTS.md.
Do not download models.
Do not load DeepSeek-MoE-16B.
Do not train.
Do not evaluate.
Do not inspect.
Do not patch the real model.
Wait for all subagents to complete, then summarize modified files, supported stages, remaining scaffold parts, safety risks, and manual commands I should run next.
```

---

# 15. Recommended Manual Experiment Order

After implementation, the user should manually run on the GPU server:

```bash
python scripts/check_env.py
```

Download assets:

```bash
python scripts/download_assets.py \
  --mode local \
  --model_name deepseek-ai/deepseek-moe-16b-base \
  --local_dir models/deepseek-moe-16b-base
```

Offline generation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/test_generate.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --max_new_tokens 120
```

WikiText PPL:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_wikitext_ppl.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only \
  --dataset_config_name wikitext-2-raw-v1 \
  --split test \
  --block_size 2048 \
  --stride 1024
```

Router inspection:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inspect_router.py \
  --model_name models/deepseek-moe-16b-base \
  --local_files_only
```

Mini-MoE experiments should be run separately after reviewing config.
