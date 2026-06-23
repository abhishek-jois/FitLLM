"""
FitLLM CLI entry point.

Subcommands:
  shard    - Shard a HuggingFace model into per-layer safetensors files
  generate - Run inference (greedy or speculative)
  train    - LoRA fine-tuning on a dataset
  probe    - Print hardware probe results
  verify   - Verify shard checksums
"""
from __future__ import annotations

import os
# Disable torch.compile / dynamo / inductor before any torch import.
# bitsandbytes and transformers trigger JIT compilation on the first forward
# pass, spawning dozens of compile workers that hang for hours on large models.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
# Reduce CUDA allocator fragmentation — critical when multiple processes share
# the GPU (vLLM etc.).  expandable_segments lets PyTorch return memory to the
# OS so other allocations can succeed instead of triggering OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
from pathlib import Path

from . import env as _env
_env.load()  # parse .env into os.environ before config objects are built

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fitllm")


def cmd_shard(args: argparse.Namespace) -> None:
    """Download and shard a HuggingFace model."""
    from .model import ShardedModel

    model_name = args.model or _env.get_str("FITLLM_MODEL")
    if not model_name:
        print("ERROR: --model is required (or set FITLLM_MODEL in .env)")
        sys.exit(1)

    shard_dir = args.output or _env.get_str("FITLLM_SHARD_DIR", "./shards")
    shard_config = _env.shard_config_from_env()
    if args.compression:
        shard_config.compression = args.compression

    print(f"Sharding model '{model_name}' → {shard_dir} (compression={shard_config.compression})")
    print(f"  GPU safety margin : {shard_config.gpu_safety_margin_gb:.2f} GB")
    model = ShardedModel.from_pretrained(
        model_name_or_path=model_name,
        shard_dir=shard_dir,
        shard_config=shard_config,
    )
    print(f"Done. {model.num_layers} shards saved to {shard_dir}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Run inference using a sharded model."""
    import torch
    from .model import ShardedModel

    shard_dir = args.shard_dir or _env.get_str("FITLLM_SHARD_DIR", "./shards")
    model_name = _env.get_str("FITLLM_MODEL") or shard_dir

    # --small-model > --draft > FITLLM_DRAFT_MODEL
    draft_model = args.small_model or args.draft or _env.get_str("FITLLM_DRAFT_MODEL", "").strip() or None

    if draft_model:
        print(f"Draft (small) model : {draft_model}")
        print(f"NOTE: Must be same family as verifier (shared tokenizer).")
        print()

    shard_config = _env.shard_config_from_env()
    inf_config = _env.inference_config_from_env()
    # CLI args override .env
    if args.small_model or args.draft:
        inf_config.draft_model = draft_model
    if args.speculative_k is not None:
        inf_config.speculative_k = args.speculative_k
    if args.max_new_tokens is not None:
        inf_config.max_new_tokens = args.max_new_tokens

    vram_limit = _env.get_float("FITLLM_VRAM_LIMIT_GB", 0.0)
    if vram_limit > 0:
        print(f"  VRAM budget       : {vram_limit:.1f} GB (set via FITLLM_VRAM_LIMIT_GB)")
    print(f"  GPU safety margin : {shard_config.gpu_safety_margin_gb:.2f} GB")

    print(f"Loading sharded model from {shard_dir} ...")
    model = ShardedModel.from_pretrained(
        model_name_or_path=model_name,
        shard_dir=shard_dir,
        shard_config=shard_config,
    )

    tokenizer = model.tokenizer
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"]

    print(f"Generating (prompt: {args.prompt!r}) ...")
    output_ids = model.generate(input_ids, inference_config=inf_config)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n--- Generated Text ---")
    print(text)
    print("---")


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune a sharded model using LoRA."""
    from datasets import load_dataset
    from .model import ShardedModel
    from .trainer import LoRATrainer

    shard_dir = args.shard_dir or _env.get_str("FITLLM_SHARD_DIR", "./shards")
    model_name = _env.get_str("FITLLM_MODEL") or shard_dir

    shard_config = _env.shard_config_from_env()
    if args.lora_rank is not None:
        shard_config.lora_rank = args.lora_rank

    train_config = _env.training_config_from_env()
    if args.lr is not None:
        train_config.lr = args.lr
    if args.grad_accum is not None:
        train_config.grad_accum = args.grad_accum
    if args.steps is not None:
        train_config.max_steps = args.steps
    if args.log_wandb:
        train_config.log_wandb = True

    vram_limit = _env.get_float("FITLLM_VRAM_LIMIT_GB", 0.0)
    if vram_limit > 0:
        print(f"  VRAM budget       : {vram_limit:.1f} GB (set via FITLLM_VRAM_LIMIT_GB)")

    print(f"Loading sharded model from {shard_dir} ...")
    model = ShardedModel.from_pretrained(
        model_name_or_path=model_name,
        shard_dir=shard_dir,
        shard_config=shard_config,
        training_config=train_config,
    )

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")

    trainer = LoRATrainer(model, train_config)

    if args.resume:
        step, sample_index = trainer.resume_from_checkpoint(args.resume)
        print(f"Resumed from step {step}, sample {sample_index}")

    print("Starting training ...")
    trainer.train(dataset)


def cmd_probe(args: argparse.Namespace) -> None:
    """Print hardware VRAM/RAM probe results."""
    from .probe import AdaptiveShardProbe

    shard_config = _env.shard_config_from_env()
    probe = AdaptiveShardProbe(
        shard_size_gb=args.shard_size_gb,
        total_shards=32,
        gpu_safety_margin_gb=shard_config.gpu_safety_margin_gb,
        cpu_safety_margin_gb=shard_config.cpu_safety_margin_gb,
        vram_limit_gb=shard_config.vram_limit_gb,
    )
    result = probe.compute_parallel_n()

    vram_limit = _env.get_float("FITLLM_VRAM_LIMIT_GB", 0.0)
    print("\n=== FitLLM Hardware Probe ===")
    if vram_limit > 0:
        print(f"  VRAM budget      : {vram_limit:.1f} GB  (FITLLM_VRAM_LIMIT_GB)")
    print(f"  Free GPU VRAM    : {result['free_gpu_gb']:.2f} GB")
    print(f"  Free CPU RAM     : {result['free_cpu_gb']:.2f} GB")
    print(f"  GPU safety margin: {shard_config.gpu_safety_margin_gb:.2f} GB")
    print(f"  GPU parallel_n   : {result['gpu_parallel_n']}")
    print(f"  CPU parallel_n   : {result['cpu_parallel_n']}")
    print(f"  Effective parallel: {result['effective_n']}")
    print(f"  Strategy         : {result['strategy']}")
    print(f"  Compute device   : {result['compute_device']}")
    print()


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify checksums of all shards in a directory."""
    from .scheduler import load_shard_with_checksum

    shard_dir = Path(args.shard_dir)
    shards = sorted(shard_dir.glob("layer_*_weights.safetensors"))

    if not shards:
        print(f"No shard files found in {shard_dir}")
        sys.exit(1)

    print(f"Verifying {len(shards)} shards in {shard_dir} ...")
    errors = 0
    for shard_path in shards:
        try:
            load_shard_with_checksum(shard_path, verify=True)
            print(f"  OK: {shard_path.name}")
        except RuntimeError as e:
            print(f"  FAIL: {shard_path.name}: {e}")
            errors += 1

    if errors:
        print(f"\n{errors}/{len(shards)} shards FAILED verification")
        sys.exit(1)
    else:
        print(f"\nAll {len(shards)} shards passed verification")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitllm",
        description="FitLLM: Layer-sharding LLM inference and training",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # shard
    p_shard = subparsers.add_parser("shard", help="Shard a model into per-layer files")
    p_shard.add_argument("--model", default=None,
                         help="HuggingFace model name or path (default: FITLLM_MODEL from .env)")
    p_shard.add_argument("--output", default=None,
                         help="Output directory for shard files (default: FITLLM_SHARD_DIR from .env)")
    p_shard.add_argument(
        "--compression", default=None, choices=["4bit", "8bit", "fp16"],
        help="Quantization compression (default: FITLLM_COMPRESSION from .env, else 4bit)"
    )
    p_shard.set_defaults(func=cmd_shard)

    # generate
    p_gen = subparsers.add_parser("generate", help="Run inference on a sharded model")
    p_gen.add_argument("--shard-dir", default=None,
                       help="Directory containing the sharded large (verifier) model")
    p_gen.add_argument(
        "--small-model", default=None,
        metavar="HF_MODEL_ID",
        help=(
            "HuggingFace model ID of the small draft model for speculative decoding. "
            "MUST be from the same model family as the verifier to share the same tokenizer. "
            "Examples: "
            "  Llama 3.x verifier  → meta-llama/Llama-3.2-1B-Instruct  "
            "  Qwen2.5 verifier    → Qwen/Qwen2.5-0.5B-Instruct  "
            "  Gemma 2 verifier    → google/gemma-2-2b-it  "
            "If omitted, falls back to plain greedy decoding (no speedup)."
        ),
    )
    p_gen.add_argument(
        "--draft", default=None,
        metavar="HF_MODEL_ID",
        help="Alias for --small-model (legacy flag, prefer --small-model).",
    )
    p_gen.add_argument("--speculative-k", type=int, default=None,
                       help="Number of tokens the draft model proposes per step (default: 4). "
                            "Ignored if --small-model is not set.")
    p_gen.add_argument("--adaptive-shards", action="store_true", default=True,
                       help="Enable adaptive multi-shard VRAM probe (default: on)")
    p_gen.add_argument("--prompt", required=True, help="Input prompt text")
    p_gen.add_argument("--max-new-tokens", type=int, default=None,
                       help="Maximum number of tokens to generate (default: 200)")
    p_gen.add_argument("--parallel-shards", type=int, default=None,
                       help="Override parallel shard count (default: auto from probe)")
    p_gen.set_defaults(func=cmd_generate)

    # train
    p_train = subparsers.add_parser("train", help="LoRA fine-tuning")
    p_train.add_argument("--shard-dir", required=True, help="Directory containing shards")
    p_train.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    p_train.add_argument("--lora-rank", type=int, default=None, help="LoRA rank (default: FITLLM_LORA_RANK from .env, else 16)")
    p_train.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (default: FITLLM_GRAD_ACCUM from .env, else 8)")
    p_train.add_argument("--steps", type=int, default=None, help="Max training steps (default: FITLLM_MAX_STEPS from .env, else 1000)")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate (default: FITLLM_LR from .env, else 2e-4)")
    p_train.add_argument("--adaptive-shards", action="store_true", default=True)
    p_train.add_argument("--log-wandb", action="store_true", default=False,
                         help="Log metrics to WandB")
    p_train.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p_train.set_defaults(func=cmd_train)

    # probe
    p_probe = subparsers.add_parser("probe", help="Print hardware VRAM/RAM probe")
    p_probe.add_argument("--shard-size-gb", type=float, default=0.5,
                         help="Estimated size of one shard in GB (default: 0.5)")
    p_probe.set_defaults(func=cmd_probe)

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify shard checksums")
    p_verify.add_argument("--shard-dir", required=True, help="Directory containing shards")
    p_verify.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
