from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ShardConfig:
    compression: str = "4bit"           # "4bit" | "8bit" | "fp16"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mixed_precision: bool = True
    adaptive_shards: bool = True
    gpu_safety_margin_gb: float = 0.75
    cpu_safety_margin_gb: float = 1.5
    prefetch_depth: int = 2
    pin_memory: bool = True
    use_cuda_streams: bool = True
    reprobe_every: int = 50             # periodic VRAM re-probe
    verify_checksums: bool = True       # shard integrity verification
    shard_group_size: int = 1           # layers per shard file; 1 = one file per layer
    weight_cache_gb: float = 0.0        # CPU RAM for forward-pass weight cache (0 = off)
    prefetch_depth: int = 2             # number of batches to prefetch ahead; more = more I/O threads
    vram_limit_gb: float = 0.0          # cap on FitLLM's OWN GPU usage; 0 = no cap


@dataclass
class TrainingConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_accum: int = 8
    max_seq_len: int = 512
    max_steps: int = 1000
    save_every: int = 100
    mixed_precision: bool = True
    log_wandb: bool = False
    checkpoint_dir: str = "./checkpoints"
    keep_checkpoints: int = 3
    max_grad_norm: float = 1.0            # gradient clipping threshold (0 = disabled)
    warmup_steps: int = 50               # linear warmup steps before cosine decay
    activation_checkpoint_every: int = 1  # save every Nth activation; rest recomputed
    shard_stationary: bool = True         # invert loop: load each shard once for all grad_accum batches


@dataclass
class InferenceConfig:
    draft_model: Optional[str] = None
    speculative_k: int = 4
    dynamic_k: bool = True              # dynamic speculative K adjustment
    k_min: int = 2
    k_max: int = 12
    layer_skip_threshold: float = 0.0
    use_flash_attention: bool = True
    use_fused_kernels: bool = True
    temperature: float = 1.0
    max_new_tokens: int = 200
    reprobe_every: int = 1             # re-probe VRAM after every generated token
