<div align="center">

# ⚡ FitLLM

### Train and Run 32B+ Language Models on Just 4 GB of VRAM

*No cloud bill. No multi-GPU rig. No model shrinkage.*  
*Just constraints — and a system built to honor them.*

---

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c?style=flat-square)](https://pytorch.org)
[![VRAM](https://img.shields.io/badge/VRAM-4%20GB%20budget-green?style=flat-square)](#)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--32B-purple?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What Is This?

**FitLLM** fine-tunes and serves 32-billion-parameter language models inside a self-imposed **4 GB VRAM budget** — a constraint tighter than most gaming GPUs.

This is not pruning, distillation, or a smaller model. It's the same full 32B model, trained end-to-end, with real gradients, on a single GPU.

The core insight: instead of fitting the model into memory, **invert the loop** — keep the model on disk, stream one layer at a time through a 3-tier cache, and let the GPU see only what it needs right now.

---

## The Problem We're Solving

Fine-tuning a 32B model with standard tools requires roughly:

| Component | VRAM needed |
|-----------|-------------|
| Model weights (bfloat16) | ~64 GB |
| Optimizer states (Adam) | ~128 GB |
| Activations + gradients | ~16 GB |
| **Total** | **~208 GB** |

FitLLM's actual usage: **4 GB.**

---

## How It Works

### Layer Sharding

The model is split into **64 per-layer `.safetensors` files** at 4-bit NF4 quantization (~0.24 GB each, ~16 GB total on disk). Shards live on NVMe and are streamed to the GPU one at a time.

```
NVMe Disk (16 GB)
  layer_000_weights.safetensors  ─┐
  layer_001_weights.safetensors   │  64 shards × 0.24 GB
  ...                             │
  layer_063_weights.safetensors  ─┘
         │
         ▼  3-tier cache
CPU RAM (13 GB weight cache)
         │
         ▼  one layer at a time
GPU VRAM (4 GB budget)
  [current layer weights]
  [LoRA adapters]
  [activations for this layer]
         │
         ▼  gradients
CPU RAM  →  CPU-side AdamW optimizer
```

### Shard-Stationary Training

Standard training loads each shard twice per mini-batch — once forward, once backward — times the number of gradient accumulation steps. With 64 layers and 16 grad-accum steps, that's **2,048 GPU↔RAM transfers per optimizer step**.

FitLLM inverts the loop:

```
for layer in range(64):                     ← outer: one layer at a time
    load shard from cache or NVMe
    for mini_batch in range(grad_accum):    ← inner: all batches through THIS layer
        forward() → save activations to CPU RAM
    for mini_batch in reversed(range(grad_accum)):
        backward() → accumulate LoRA grads to CPU RAM
    evict shard from GPU
    optimizer.step()  ← CPU-side AdamW, zero VRAM
```

Each layer is loaded **once** per optimizer step regardless of grad_accum size. Transfer count drops to **128** — a **16× reduction** in data movement.

### Three-Tier Shard Cache

| Tier | Location | Size | Purpose |
|------|----------|------|---------|
| L1 | GPU VRAM | 4 GB (FitLLM budget) | Active compute |
| L2 | CPU RAM | 13 GB weight cache | Fast layer lookup on backward pass |
| L3 | NVMe | Full model (16 GB) | Always-available ground truth |

After the first forward pass, all 64 layer weights sit in CPU RAM. The backward pass hits L2 (RAM) on every layer — no disk reads after warmup.

### CPU-Side Optimizer

Adam's momentum buffers (`m` and `v`) are the same size as the parameters they track. For a 32B model, that's ~64 GB of optimizer state. FitLLM runs AdamW entirely on CPU:

- Zero VRAM for optimizer states
- GPU freed immediately after each layer's backward pass
- Full-precision math on the optimizer update (no quantization error in the update step)

### Adaptive VRAM Probe

Before every layer, FitLLM probes available VRAM and tracks its own allocations against the budget. On shared GPU machines (tested with vLLM, model-serving, and UI processes co-tenanted on the same A100), other tenants cause VRAM to swing from 31 GB free to 878 MB free within minutes.

The probe waits with exponential backoff (up to 120 seconds) for other tenants to release memory before computing — rather than immediately falling to a non-functional CPU path.

### LoRA Fine-Tuning

Low-Rank Adaptation targets `q_proj` and `v_proj` with rank=16:
- **~25M trainable parameters** out of 32B total (0.08% of the model)
- Adapters and their gradients live in CPU RAM
- Only the current layer's adapter slice is on GPU during its compute window

---

## Architecture

```
fitllm/
├── model.py        ShardedModel — top-level: owns HF model ref, wires all components
├── forward.py      Shard-stationary forward pass, VRAM-aware compute device selection
├── backward.py     Reversed layer-by-layer backward through CPU-cached activations
├── trainer.py      LoRATrainer — training loop, LR schedule, grad clipping, checkpointing
├── optimizer.py    CPU-side AdamW with per-layer LoRA grad handling
├── scheduler.py    3-tier shard cache + async NVMe prefetch threads
├── probe.py        AdaptiveShardProbe — live VRAM budget tracking with try/except resilience
├── lora.py         LoRA weight injection and adapter management
├── qlora.py        4-bit NF4 quantization via bitsandbytes
├── kernels.py      FlashAttention2 + LigerKernel fusion detection and application
├── inference.py    Speculative decoding engine (draft + verifier)
├── config.py       ShardConfig, TrainingConfig, InferenceConfig dataclasses
├── env.py          .env → typed config objects
└── registry.py     Model architecture registry (maps model_type → layer accessor)
```

---

## Quick Start

### 1. Install

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate

# Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch>=2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install FitLLM
pip install -e .
```

**Optional acceleration** (install after core — FlashAttention compiles from source, takes 10–20 min):
```bash
pip install -r requirements-acceleration.txt --no-build-isolation
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env to match your hardware
```

Key settings:
```ini
FITLLM_MODEL=Qwen/Qwen2.5-32B-Instruct
FITLLM_SHARD_DIR=./shards-qwen32b
FITLLM_VRAM_LIMIT_GB=4.0          # hard cap — FitLLM enforces this against its own usage
FITLLM_WEIGHT_CACHE_GB=13.0       # CPU RAM for the layer cache
FITLLM_COMPRESSION=4bit           # NF4 4-bit quantization
```

### 3. Shard the Model

One-time step. Downloads the model from HuggingFace and splits it into per-layer files:

```bash
python -m fitllm shard \
  --model Qwen/Qwen2.5-32B-Instruct \
  --output ./shards-qwen32b
```

For a 32B model in 4-bit: 64 files × ~242 MB = ~15.5 GB on disk.

### 4. Fine-Tune

```bash
python -m fitllm train \
  --shard-dir ./shards-qwen32b \
  --dataset databricks/databricks-dolly-15k \
  --steps 1000
```

What healthy training looks like:

```
===================================================================================
  FitLLM Training  |  Qwen/Qwen2.5-32B-Instruct  |  4-bit NF4  |  LoRA r=16
  steps=1000  grad_accum=8  lr=2e-4  seq_len=512  warmup=50
===================================================================================
    Step    Loss   Smooth  GradNorm       LR   Tok/s   Elapsed      ETA
-----------------------------------------------------------------------------------
       1/1000  2.4132  2.4132    0.8821  2.0e-06    14.2   0:01:28  24:38:00
      10/1000  2.1847  2.2934    0.7103  2.0e-05    16.8   0:14:22  23:49:00
      50/1000  1.9203  2.0118    0.6244  2.0e-04    18.1   1:12:05  22:16:00
     100/1000  1.7441  1.8392    0.5902  1.9e-04    18.4   2:27:12  21:03:00
```

Log lines to watch for:
- `[Waiting on GPU VRAM (other tenants busy)]` — normal, will self-resolve in seconds
- `[GPU VRAM contention persisted for 120s]` — concerning; GPU heavily saturated
- Loss decreasing steadily after warmup (step 50) — training is healthy

### 5. Generate

```bash
# Greedy decoding
python -m fitllm generate \
  --shard-dir ./shards-qwen32b \
  --prompt "Explain gradient descent to a 10-year-old."

# Speculative decoding (faster — requires small model from same family)
python -m fitllm generate \
  --shard-dir ./shards-qwen32b \
  --small-model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "Write a Python function to merge two sorted lists."
```

### 6. Probe Hardware

Print live VRAM and RAM stats before committing to a run:

```bash
python -m fitllm probe

# === FitLLM Hardware Probe ===
#   VRAM budget      : 4.0 GB  (FITLLM_VRAM_LIMIT_GB)
#   Free GPU VRAM    : 2.84 GB
#   Free CPU RAM     : 143.7 GB
#   GPU safety margin: 0.75 GB
#   GPU parallel_n   : 13
#   Strategy         : single_shard
#   Compute device   : cuda
```

---

## CLI Reference

```
python -m fitllm <command> [options]

  shard      Download and split a HuggingFace model into per-layer shard files
  train      LoRA fine-tuning on a HuggingFace dataset
  generate   Inference with greedy or speculative decoding
  probe      Print live VRAM/RAM stats and compute strategy
  verify     Verify SHA-256 checksums of all shard files

train options:
  --shard-dir DIR      Directory containing shard files (required)
  --dataset NAME       HuggingFace dataset name (required)
  --steps N            Training steps (default: 1000)
  --lr FLOAT           Learning rate (default: 2e-4)
  --grad-accum N       Gradient accumulation steps (default: 8)
  --lora-rank N        LoRA rank (default: 16)
  --resume PATH        Resume from a saved checkpoint
  --log-wandb          Enable WandB experiment tracking
```

---

## Configuration Reference

All values can be set in `.env` or as environment variables. CLI flags override `.env`.

### VRAM & Hardware

| Variable | Default | Description |
|----------|---------|-------------|
| `FITLLM_VRAM_LIMIT_GB` | — | **Hard cap on FitLLM's own GPU usage.** Enforced against `torch.cuda.memory_reserved()` — not a snapshot, always live. |
| `FITLLM_GPU_SAFETY_MARGIN_GB` | `0.75` | VRAM headroom reserved for CUDA kernel overhead |
| `FITLLM_CPU_RAM_LIMIT_GB` | `16.0` | Total CPU RAM budget (model loading + weight cache) |
| `FITLLM_WEIGHT_CACHE_GB` | `0` | CPU RAM for layer weight cache. `0` = auto (uses available RAM minus safety margin) |
| `FITLLM_CPU_SAFETY_MARGIN_GB` | `3.0` | CPU RAM always kept free |

### Model

| Variable | Default | Description |
|----------|---------|-------------|
| `FITLLM_MODEL` | — | HuggingFace model ID or local path |
| `FITLLM_SHARD_DIR` | `./shards` | Directory for shard `.safetensors` files |
| `FITLLM_COMPRESSION` | `4bit` | Quantization: `4bit` (NF4), `8bit`, or `fp16` |
| `FITLLM_VERIFY_CHECKSUMS` | `1` | SHA-256 check on load. Set to `0` during training to skip per-step overhead. |

### Fine-Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `FITLLM_LR` | `1e-4` | Base learning rate |
| `FITLLM_GRAD_ACCUM` | `16` | Gradient accumulation steps |
| `FITLLM_MAX_STEPS` | `9380` | Total training steps (10 epochs of dolly-15k) |
| `FITLLM_MAX_SEQ_LEN` | `512` | Maximum token sequence length |
| `FITLLM_LORA_RANK` | `16` | LoRA rank |
| `FITLLM_LORA_ALPHA` | `32` | LoRA alpha scaling |
| `FITLLM_LORA_TARGETS` | `q_proj,v_proj` | Which attention projections to adapt |
| `FITLLM_WARMUP_STEPS` | `50` | Linear LR warmup steps before cosine decay |
| `FITLLM_MAX_GRAD_NORM` | `1.0` | Gradient clipping threshold |
| `FITLLM_CHECKPOINT_DIR` | `./checkpoints` | Checkpoint save directory |
| `FITLLM_SHARD_STATIONARY` | `1` | Enable shard-stationary training (recommended) |

### Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `FITLLM_DRAFT_MODEL` | — | Small draft model for speculative decoding |
| `FITLLM_SPECULATIVE_K` | `4` | Draft tokens proposed per speculative step |
| `FITLLM_MAX_NEW_TOKENS` | `512` | Max tokens to generate |
| `FITLLM_TEMPERATURE` | `0.7` | Sampling temperature |

---

## Current Training Run

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-32B-Instruct |
| Dataset | databricks/databricks-dolly-15k (15,011 examples) |
| Target | 9,380 steps = 10 epochs |
| VRAM budget | **4.0 GB** (self-imposed, enforced live) |
| LoRA | rank=16, targets: `q_proj` + `v_proj` |
| Trainable params | ~25M out of 32B (0.08%) |
| Hardware | A100 80 GB shared with other tenants |

---

## Hardware Guide

| GPU VRAM | Model size (4-bit) | Layers in parallel | Notes |
|----------|-------------------|--------------------|-------|
| 4 GB | 32B | ~13 layers | FitLLM's design target |
| 8 GB | 32B | ~28 layers | Comfortable for 32B |
| 16 GB | 70B | ~15 layers | Viable for 70B |
| 24 GB | 70B | ~22 layers | Good for 70B |
| 80 GB | 405B | ~25 layers | Full frontier models |

**Disk space:** `model_size_fp16 / 2` for 4-bit. A 32B model ≈ 16 GB.  
**System RAM:** More RAM = bigger weight cache = faster training. Minimum ~4 GB (cache off).  
**CPU-only:** FitLLM runs without a GPU — all compute on CPU. Functional but slow.

---

## Long Runs

For multi-hour or multi-day training, use `tmux` so the session survives disconnects:

```bash
tmux new -s fitllm_train
source /data/fitllm-venv/bin/activate

python -m fitllm train \
  --shard-dir ./shards-qwen32b \
  --dataset databricks/databricks-dolly-15k \
  --steps 9380 \
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# Detach:   Ctrl+B, D
# Reattach: tmux attach -t fitllm_train
```

Resume from a checkpoint after any interruption:

```bash
python -m fitllm train \
  --shard-dir ./shards-qwen32b \
  --dataset databricks/databricks-dolly-15k \
  --steps 9380 \
  --resume ./checkpoints-qwen32b/step_1200
```

---

## Tests

```bash
pytest tests/ -v
```

| Test file | What it verifies |
|-----------|-----------------|
| `test_vram_ceiling.py` | FitLLM never exceeds its VRAM budget under any codepath |
| `test_gradient_equivalence.py` | Shard-stationary grads numerically match monolithic grads |
| `test_lora_gradients.py` | LoRA adapter gradient correctness |
| `test_adaptive_probe.py` | Probe behavior under contention (multi-tenant simulation) |
| `test_forward_backward.py` | Forward/backward numerical equivalence |
| `test_optimizer_step.py` | CPU-side optimizer update correctness |
| `test_multi_shard_forward.py` | Multi-shard batched forward pass correctness |

---

## Research

This work is described in a research paper and patent filing:

- **`paper/main.pdf`** — FitLLM: Layer-Sharded Training and Inference of Large Language Models Under Extreme VRAM Constraints
- **`patent/fitllm_formatted_v3.pdf`** — Patent filing covering the shard-stationary training method and 3-tier cache architecture

---

## Why 4 GB?

Because it's the hardest constraint that's still practically meaningful. A 4 GB VRAM limit:

- Is smaller than an NVIDIA RTX 3060 (12 GB), GTX 1080 Ti (11 GB), or any modern consumer GPU
- Is less than 5% of the A100 used in development
- Rules out every standard training trick that relies on "just fit it all in memory"

If the system works at 4 GB, it works at 8 GB and 16 GB with room to spare. The constraint is a proof, not a limitation.

---

## Installation Notes

**Python version:** 3.11+ required (uses `match` statements and newer typing).

**CUDA version:** The project is tested on CUDA 12.1. Adjust the `--index-url` in install commands if you have a different CUDA version.

**bitsandbytes on Linux:** Works out of the box. On Windows, use `bitsandbytes-windows`.

**FlashAttention:** Requires a CUDA GPU with compute capability ≥ 8.0 (A100, H100, RTX 30xx/40xx). Compile time is 10–20 minutes on first install — this is normal.

**LigerKernel:** Falls back gracefully if not installed. When installed, fuses RMSNorm and rotary embedding kernels for ~10–20% per-layer speedup.

---

<div align="center">

*Built to prove that resource constraints are an engineering problem, not a stopping condition.*

</div>
