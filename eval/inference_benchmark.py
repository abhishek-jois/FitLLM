#!/usr/bin/env python3
"""
FitLLM inference benchmark script.

Measures tokens/sec and time-to-first-token for various inference configurations.

Usage:
    python eval/inference_benchmark.py \
        --model <hf-model-or-shard-dir> \
        --shard-dir <shard-dir> \
        --configs all \
        --prompt "Tell me about the universe" \
        --max-new-tokens 50
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Available configs
# ──────────────────────────────────────────────────────────────────────────────

ALL_CONFIGS = ["baseline", "4bit", "multi_shard", "speculative"]


def print_table(results: List[Dict]) -> None:
    """Print a formatted table of benchmark results."""
    headers = ["Config", "TTFT (ms)", "Tokens/sec", "Total tokens", "Notes"]
    col_w = [max(len(h), max(len(str(r.get(h.lower().replace("/", "_").replace(" ", "_"), "")))
                             for r in results))
             for h in headers]
    col_w = [max(20, w) for w in col_w]

    def row_str(values):
        return " | ".join(str(v).ljust(w) for v, w in zip(values, col_w))

    sep = "-+-".join("-" * w for w in col_w)
    print("\n" + "=" * sum(col_w + [3 * (len(col_w) - 1)]))
    print("FitLLM Inference Benchmark Results")
    print("=" * sum(col_w + [3 * (len(col_w) - 1)]))
    print(row_str(headers))
    print(sep)
    for r in results:
        vals = [
            r.get("config", ""),
            f"{r.get('ttft_ms', 0):.1f}",
            f"{r.get('tokens_per_sec', 0):.2f}",
            str(r.get("total_tokens", 0)),
            r.get("notes", ""),
        ]
        print(row_str(vals))
    print("=" * sum(col_w + [3 * (len(col_w) - 1)]))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(model, tokenizer, prompt: str, max_new_tokens: int) -> Dict:
    """Greedy decoding, no quantization, no speculative."""
    from fitllm.config import InferenceConfig

    cfg = InferenceConfig(draft_model=None, max_new_tokens=max_new_tokens, temperature=0.01)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _ = model.forward(input_ids)
    ttft = (time.perf_counter() - t0) * 1000

    t_gen_start = time.perf_counter()
    output = model.generate(input_ids, inference_config=cfg)
    t_gen = time.perf_counter() - t_gen_start

    new_tokens = output.shape[1] - input_ids.shape[1]
    tps = new_tokens / max(t_gen, 1e-6)

    return {
        "config": "baseline",
        "ttft_ms": ttft,
        "tokens_per_sec": tps,
        "total_tokens": new_tokens,
        "notes": "greedy, no quant",
    }


def run_4bit(model_name: str, shard_dir: str, prompt: str, max_new_tokens: int) -> Dict:
    """Benchmark with 4-bit NF4 quantization."""
    from fitllm.config import ShardConfig, InferenceConfig
    from fitllm.model import ShardedModel

    shard_cfg = ShardConfig(compression="4bit")
    inf_cfg = InferenceConfig(max_new_tokens=max_new_tokens, temperature=0.01)

    print("  Loading 4bit model ...")
    model = ShardedModel.from_pretrained(
        model_name_or_path=model_name,
        shard_dir=shard_dir + "_4bit",
        shard_config=shard_cfg,
    )
    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    t0 = time.perf_counter()
    logits, _ = model.forward(input_ids)
    ttft = (time.perf_counter() - t0) * 1000

    t_start = time.perf_counter()
    output = model.generate(input_ids, inference_config=inf_cfg)
    t_total = time.perf_counter() - t_start

    new_tokens = output.shape[1] - input_ids.shape[1]
    return {
        "config": "4bit",
        "ttft_ms": ttft,
        "tokens_per_sec": new_tokens / max(t_total, 1e-6),
        "total_tokens": new_tokens,
        "notes": "NF4 double-quant",
    }


def run_multi_shard(model, tokenizer, prompt: str, max_new_tokens: int, parallel_n: int) -> Dict:
    """Benchmark with multiple shards in parallel."""
    from fitllm.config import InferenceConfig

    probe_result = model.probe.compute_parallel_n()
    effective_n = probe_result["effective_n"]

    cfg = InferenceConfig(max_new_tokens=max_new_tokens, temperature=0.01)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    t0 = time.perf_counter()
    logits, _ = model.forward(input_ids)
    ttft = (time.perf_counter() - t0) * 1000

    t_start = time.perf_counter()
    output = model.generate(input_ids, inference_config=cfg)
    t_total = time.perf_counter() - t_start

    new_tokens = output.shape[1] - input_ids.shape[1]
    return {
        "config": "multi_shard",
        "ttft_ms": ttft,
        "tokens_per_sec": new_tokens / max(t_total, 1e-6),
        "total_tokens": new_tokens,
        "notes": f"parallel_n={effective_n}",
    }


def run_speculative(
    model,
    tokenizer,
    draft_model_name: str,
    prompt: str,
    max_new_tokens: int,
    k: int,
) -> Dict:
    """Benchmark speculative decoding."""
    from fitllm.config import InferenceConfig

    cfg = InferenceConfig(
        draft_model=draft_model_name,
        speculative_k=k,
        dynamic_k=True,
        max_new_tokens=max_new_tokens,
        temperature=0.01,
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    t0 = time.perf_counter()
    logits, _ = model.forward(input_ids)
    ttft = (time.perf_counter() - t0) * 1000

    t_start = time.perf_counter()
    output = model.generate(input_ids, inference_config=cfg)
    t_total = time.perf_counter() - t_start

    new_tokens = output.shape[1] - input_ids.shape[1]
    return {
        "config": "speculative",
        "ttft_ms": ttft,
        "tokens_per_sec": new_tokens / max(t_total, 1e-6),
        "total_tokens": new_tokens,
        "notes": f"draft={draft_model_name}, k={k}",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FitLLM Inference Benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--shard-dir", required=True, help="Directory for shards")
    parser.add_argument(
        "--configs", default="all",
        help="Comma-separated configs: baseline,4bit,multi_shard,speculative,all"
    )
    parser.add_argument("--prompt", default="The future of artificial intelligence is",
                        help="Prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--draft-model", default=None, help="Draft model for speculative")
    parser.add_argument("--speculative-k", type=int, default=4)
    parser.add_argument("--parallel-shards", type=int, default=None)
    args = parser.parse_args()

    # Parse config list
    requested = args.configs.lower().split(",")
    if "all" in requested:
        requested = ALL_CONFIGS
    requested = [c.strip() for c in requested]

    print(f"Running FitLLM inference benchmark on: {args.model}")
    print(f"Configs: {requested}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Max new tokens: {args.max_new_tokens}\n")

    # Load base model (used by multiple configs)
    from fitllm.config import ShardConfig, InferenceConfig
    from fitllm.model import ShardedModel

    shard_cfg = ShardConfig(compression="fp16")
    print("Loading sharded model ...")
    model = ShardedModel.from_pretrained(
        model_name_or_path=args.model,
        shard_dir=args.shard_dir,
        shard_config=shard_cfg,
    )
    tokenizer = model.tokenizer

    results: List[Dict] = []

    for cfg_name in requested:
        print(f"\nRunning config: {cfg_name} ...")
        try:
            if cfg_name == "baseline":
                r = run_baseline(model, tokenizer, args.prompt, args.max_new_tokens)
            elif cfg_name == "4bit":
                r = run_4bit(args.model, args.shard_dir, args.prompt, args.max_new_tokens)
            elif cfg_name == "multi_shard":
                r = run_multi_shard(
                    model, tokenizer, args.prompt, args.max_new_tokens,
                    parallel_n=args.parallel_shards or 2,
                )
            elif cfg_name == "speculative":
                if args.draft_model is None:
                    print("  Skipping speculative: --draft-model not provided")
                    continue
                r = run_speculative(
                    model, tokenizer, args.draft_model,
                    args.prompt, args.max_new_tokens, args.speculative_k,
                )
            else:
                print(f"  Unknown config: {cfg_name}")
                continue

            results.append(r)
            print(f"  Done: {r['tokens_per_sec']:.2f} tok/s, TTFT={r['ttft_ms']:.1f}ms")

        except Exception as e:
            print(f"  ERROR running {cfg_name}: {e}")
            results.append({
                "config": cfg_name,
                "ttft_ms": 0,
                "tokens_per_sec": 0,
                "total_tokens": 0,
                "notes": f"ERROR: {e}",
            })

    print_table(results)


if __name__ == "__main__":
    main()
