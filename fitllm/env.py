"""
Load FitLLM settings from a .env file and os.environ.

Priority (highest → lowest):
  1. os.environ (already-set shell variables)
  2. .env file in the project root (or path given to load())
  3. Code defaults in config.py

Call load() once at startup (done automatically by __main__.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load(env_path: Optional[Path] = None) -> None:
    """
    Parse a .env file and inject variables into os.environ.
    Variables already present in os.environ are NOT overwritten.
    """
    if env_path is None:
        # Walk up from this file to find .env
        here = Path(__file__).resolve().parent
        for candidate in [here / ".env", here.parent / ".env"]:
            if candidate.exists():
                env_path = candidate
                break

    if env_path is None or not Path(env_path).exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip inline comments
            if " #" in value:
                value = value[: value.index(" #")].strip()
            # Only set if not already in environment
            if key and key not in os.environ:
                os.environ[key] = value


# ── Typed getters ─────────────────────────────────────────────────────────────

def get_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def get_float(key: str, default: float = 0.0) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def get_int(key: str, default: int = 0) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Convenience: build config objects from env ────────────────────────────────

def shard_config_from_env():
    """Return a ShardConfig populated from env / .env values."""
    from .config import ShardConfig

    safety = get_float("FITLLM_GPU_SAFETY_MARGIN_GB", 0.75)
    vram_limit = get_float("FITLLM_VRAM_LIMIT_GB", 0.0)

    # LoRA targets: comma-separated list, e.g. "q_proj,v_proj"
    lora_targets_str = get_str("FITLLM_LORA_TARGETS", "q_proj,v_proj")
    lora_targets = tuple(t.strip() for t in lora_targets_str.split(",") if t.strip())

    return ShardConfig(
        compression=get_str("FITLLM_COMPRESSION", "4bit"),
        lora_rank=get_int("FITLLM_LORA_RANK", 16),
        lora_alpha=get_float("FITLLM_LORA_ALPHA", 32.0),
        lora_targets=lora_targets,
        gpu_safety_margin_gb=safety,
        cpu_safety_margin_gb=get_float("FITLLM_CPU_SAFETY_MARGIN_GB", 1.5),
        shard_group_size=get_int("FITLLM_SHARD_GROUP_SIZE", 1),
        verify_checksums=get_bool("FITLLM_VERIFY_CHECKSUMS", True),
        prefetch_depth=get_int("FITLLM_PREFETCH_DEPTH", 2),
        weight_cache_gb=get_float("FITLLM_WEIGHT_CACHE_GB", 0.0),
        vram_limit_gb=vram_limit,
    )


def inference_config_from_env():
    """Return an InferenceConfig populated from env / .env values."""
    from .config import InferenceConfig

    draft = get_str("FITLLM_DRAFT_MODEL", "").strip() or None
    return InferenceConfig(
        draft_model=draft,
        speculative_k=get_int("FITLLM_SPECULATIVE_K", 4),
        temperature=get_float("FITLLM_TEMPERATURE", 1.0),
        max_new_tokens=get_int("FITLLM_MAX_NEW_TOKENS", 200),
    )


def training_config_from_env():
    """Return a TrainingConfig populated from env / .env values."""
    from .config import TrainingConfig

    return TrainingConfig(
        lr=get_float("FITLLM_LR", 2e-4),
        grad_accum=get_int("FITLLM_GRAD_ACCUM", 8),
        max_steps=get_int("FITLLM_MAX_STEPS", 1000),
        max_seq_len=get_int("FITLLM_MAX_SEQ_LEN", 512),
        log_wandb=get_bool("FITLLM_LOG_WANDB", False),
        checkpoint_dir=get_str("FITLLM_CHECKPOINT_DIR", "./checkpoints"),
        max_grad_norm=get_float("FITLLM_MAX_GRAD_NORM", 1.0),
        warmup_steps=get_int("FITLLM_WARMUP_STEPS", 50),
        shard_stationary=get_bool("FITLLM_SHARD_STATIONARY", True),
    )
