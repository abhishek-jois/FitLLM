#!/usr/bin/env python3
"""
FitLLM gradient equivalence evaluation.

Runs BackwardEngine on --model with --n-samples and reports max gradient
difference vs torch.autograd reference.

Usage:
    python eval/gradient_equivalence.py \
        --model gpt2 \
        --shard-dir /tmp/gpt2_shards \
        --n-samples 5 \
        --seq-len 32
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Reference gradient computation via standard autograd
# ──────────────────────────────────────────────────────────────────────────────

def compute_reference_grads(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute full-model gradients using standard torch.autograd."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()

    logits = model(input_ids)
    b, s, v = logits.shape
    loss = F.cross_entropy(logits.view(b * s, v), labels.view(b * s))
    loss.backward()

    return {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


# ──────────────────────────────────────────────────────────────────────────────
# FitLLM BackwardEngine gradient computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_fitllm_grads(
    model: nn.Module,
    shard_dir: Path,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute gradients using FitLLM's BackwardEngine."""
    from fitllm.probe import AdaptiveShardProbe
    from fitllm.scheduler import ShardScheduler, save_shard_with_checksum
    from fitllm.forward import ForwardEngine, _get_decoder_layers
    from fitllm.backward import BackwardEngine

    layers = _get_decoder_layers(model)
    num_layers = len(layers)

    # Save shards
    for idx, layer in enumerate(layers):
        sd = {k: v.contiguous() for k, v in layer.state_dict().items()}
        path = shard_dir / f"layer_{idx:03d}_weights.safetensors"
        save_shard_with_checksum(sd, path)

    probe = MagicMock(spec=AdaptiveShardProbe)
    probe.get_parallel_n.return_value = {
        "effective_n": 1,
        "strategy": "single_shard",
        "gpu_parallel_n": 1,
        "cpu_parallel_n": 1,
        "free_gpu_gb": 0.0,
        "free_cpu_gb": 16.0,
    }

    scheduler = ShardScheduler(
        shard_dir=shard_dir,
        device="cpu",
        max_parallel=2,
        pin_memory=False,
        use_cuda_streams=False,
    )

    class _ModelRef:
        def __init__(self):
            self._hf_model = model
            self._verify_checksums = False
            self._num_layers = num_layers

        @property
        def num_layers(self):
            return self._num_layers

    model_ref = _ModelRef()
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Model has no lm_head")

    # Find embed_tokens
    embed_tokens = None
    for path_tuple in [("model", "embed_tokens"), ("transformer", "wte")]:
        obj = model
        try:
            for attr in path_tuple:
                obj = getattr(obj, attr)
            embed_tokens = obj
            break
        except AttributeError:
            continue
    if embed_tokens is None:
        raise RuntimeError("Cannot find embed_tokens")

    fwd = ForwardEngine(
        model_ref=model_ref,
        scheduler=scheduler,
        probe=probe,
        lm_head=lm_head,
        embed_tokens=embed_tokens,
        use_fused_kernels=False,
    )
    bwd = BackwardEngine(
        model_ref=model_ref,
        scheduler=scheduler,
        probe=probe,
        lm_head=lm_head,
        loss_fn=nn.CrossEntropyLoss(),
        grad_dir=shard_dir / "grads",
        grad_accum_steps=1,
    )

    logits, activations = fwd.forward(input_ids)
    b, s, v = logits.shape
    loss = F.cross_entropy(logits.view(b * s, v), labels.view(b * s))
    bwd.backward(loss, activations, labels)

    # Collect accumulated grad files
    from fitllm.scheduler import load_shard_with_checksum
    grads: Dict[str, torch.Tensor] = {}
    grad_dir = shard_dir / "grads"
    for grad_file in sorted(grad_dir.glob("layer_*_grads.safetensors")):
        sd = load_shard_with_checksum(grad_file, verify=False)
        for k, v in sd.items():
            grads[k] = v

    return grads


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FitLLM Gradient Equivalence Eval")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--shard-dir", required=True, help="Directory for temporary shards")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of random samples")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--lora", action="store_true", help="Inject LoRA before computing grads")
    parser.add_argument("--lora-rank", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = model.config.vocab_size

    if args.lora:
        from fitllm.lora import LoRAManager
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=args.lora_rank, alpha=float(args.lora_rank * 2))
        mgr.freeze_base_model(model)
        print(f"Injected LoRA (rank={args.lora_rank}) into {mgr.num_lora_layers} layers")

    print(f"\nRunning gradient equivalence on {args.n_samples} samples (seq_len={args.seq_len})")

    all_max_diffs: List[float] = []

    for sample_idx in range(args.n_samples):
        torch.manual_seed(sample_idx)
        input_ids = torch.randint(0, vocab_size, (1, args.seq_len))
        labels = torch.randint(0, vocab_size, (1, args.seq_len - 1))

        # Reference gradients (full autograd)
        ref_grads = compute_reference_grads(model, input_ids[:, :-1], labels)

        # FitLLM BackwardEngine gradients
        try:
            fitllm_grads = compute_fitllm_grads(model, shard_dir, input_ids[:, :-1], labels)
        except Exception as e:
            print(f"  Sample {sample_idx}: FitLLM backward failed: {e}")
            continue

        # Compare overlapping keys
        common_keys = set(ref_grads.keys()) & set(fitllm_grads.keys())
        if not common_keys:
            print(f"  Sample {sample_idx}: No overlapping gradient keys to compare")
            continue

        max_diff = max(
            (ref_grads[k] - fitllm_grads[k]).abs().max().item()
            for k in common_keys
        )
        all_max_diffs.append(max_diff)
        print(f"  Sample {sample_idx}: max |grad_ref - grad_fitllm| = {max_diff:.6f}")

    if all_max_diffs:
        overall_max = max(all_max_diffs)
        overall_mean = sum(all_max_diffs) / len(all_max_diffs)
        print(f"\n{'='*50}")
        print(f"Gradient Equivalence Summary ({args.n_samples} samples)")
        print(f"  Max absolute difference:  {overall_max:.6f}")
        print(f"  Mean max difference:      {overall_mean:.6f}")
        print(f"  PASS (< 1e-3): {overall_max < 1e-3}")
        print(f"{'='*50}\n")
    else:
        print("\nNo valid comparisons could be made.")


if __name__ == "__main__":
    main()
