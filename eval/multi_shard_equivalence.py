
#!/usr/bin/env python3
"""
FitLLM multi-shard equivalence evaluation.

Compares logits from 1-shard vs N-shard forward pass on --model,
reports max absolute logit difference.

Usage:
    python eval/multi_shard_equivalence.py \
        --model gpt2 \
        --shard-dir /tmp/gpt2_shards \
        --parallel-shards 4 \
        --n-samples 5 \
        --seq-len 16
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build ForwardEngine with a given parallel_n
# ──────────────────────────────────────────────────────────────────────────────

def build_forward_engine(
    model: nn.Module,
    shard_dir: Path,
    parallel_n: int,
    shards_already_saved: bool = False,
):
    """Build a ForwardEngine for `model` with shards in `shard_dir`."""
    from fitllm.probe import AdaptiveShardProbe
    from fitllm.scheduler import ShardScheduler, save_shard_with_checksum
    from fitllm.forward import ForwardEngine, _get_decoder_layers

    layers = _get_decoder_layers(model)
    if layers is None:
        raise RuntimeError("Cannot find decoder layers")

    num_layers = len(layers)

    if not shards_already_saved:
        for idx, layer in enumerate(layers):
            sd = {k: v.contiguous() for k, v in layer.state_dict().items()}
            path = shard_dir / f"layer_{idx:03d}_weights.safetensors"
            save_shard_with_checksum(sd, path)

    probe = MagicMock(spec=AdaptiveShardProbe)
    probe.get_parallel_n.return_value = {
        "effective_n": parallel_n,
        "strategy": "multi_shard" if parallel_n > 1 else "single_shard",
        "gpu_parallel_n": parallel_n,
        "cpu_parallel_n": parallel_n,
        "free_gpu_gb": 0.0,
        "free_cpu_gb": 16.0,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scheduler = ShardScheduler(
        shard_dir=shard_dir,
        device=device,
        max_parallel=max(2, parallel_n),
        pin_memory=False,
        use_cuda_streams=False,
    )

    # Find lm_head and embed_tokens
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Model has no lm_head")

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

    class _ModelRef:
        def __init__(self):
            self._hf_model = model
            self._verify_checksums = False
            self._num_layers = num_layers

        @property
        def num_layers(self):
            return self._num_layers

    engine = ForwardEngine(
        model_ref=_ModelRef(),
        scheduler=scheduler,
        probe=probe,
        lm_head=lm_head,
        embed_tokens=embed_tokens,
        use_fused_kernels=False,
    )
    return engine


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FitLLM Multi-Shard Equivalence Eval")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--shard-dir", required=True, help="Directory for shards")
    parser.add_argument(
        "--parallel-shards", type=int, default=4,
        help="Number of parallel shards (N) to compare against 1-shard baseline"
    )
    parser.add_argument("--n-samples", type=int, default=5, help="Number of random test inputs")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--atol", type=float, default=1e-4, help="Tolerance threshold for PASS/FAIL")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    vocab_size = model.config.vocab_size

    print(f"Saving shards to {shard_dir} ...")
    engine_1 = build_forward_engine(model, shard_dir, parallel_n=1, shards_already_saved=False)
    engine_n = build_forward_engine(model, shard_dir, parallel_n=args.parallel_shards, shards_already_saved=True)

    print(f"\nComparing 1-shard vs {args.parallel_shards}-shard forward on {args.n_samples} samples")
    print(f"Sequence length: {args.seq_len}, vocab_size: {vocab_size}")

    all_max_diffs: List[float] = []
    all_mean_diffs: List[float] = []

    for sample_idx in range(args.n_samples):
        torch.manual_seed(sample_idx * 17 + 3)
        input_ids = torch.randint(0, vocab_size, (1, args.seq_len))

        with torch.no_grad():
            logits_1, _ = engine_1.forward(input_ids)
            logits_n, _ = engine_n.forward(input_ids)

        abs_diff = (logits_1 - logits_n).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)

        status = "PASS" if max_diff < args.atol else "FAIL"
        print(f"  Sample {sample_idx}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}  [{status}]")

    print(f"\n{'='*60}")
    print(f"Multi-Shard Equivalence Summary")
    print(f"  Model:           {args.model}")
    print(f"  Shard dir:       {shard_dir}")
    print(f"  Parallel shards: {args.parallel_shards}")
    print(f"  N samples:       {args.n_samples}")
    print(f"  Tolerance:       {args.atol}")
    print(f"")
    print(f"  Max absolute diff (over all samples): {max(all_max_diffs):.6f}")
    print(f"  Mean max diff:                        {sum(all_max_diffs)/len(all_max_diffs):.6f}")
    print(f"  Mean mean diff:                       {sum(all_mean_diffs)/len(all_mean_diffs):.8f}")
    print(f"")
    overall_pass = max(all_max_diffs) < args.atol
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'} (threshold {args.atol})")
    print(f"{'='*60}\n")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
