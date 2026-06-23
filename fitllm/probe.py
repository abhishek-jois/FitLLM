from __future__ import annotations

import warnings
from typing import Dict

import psutil
import torch


class AdaptiveShardProbe:
    """
    Probes available VRAM and CPU RAM to determine how many shards can be
    loaded in parallel. Caches the result and re-probes every `reprobe_every`
    calls, emitting a warning if parallel_n has decreased.
    """

    def __init__(
        self,
        shard_size_gb: float,
        total_shards: int,
        gpu_safety_margin_gb: float = 0.75,
        cpu_safety_margin_gb: float = 1.5,
        reprobe_every: int = 50,
        vram_limit_gb: float = 0.0,
    ) -> None:
        self.shard_size_gb = shard_size_gb
        self.total_shards = total_shards
        self.gpu_safety_margin_gb = gpu_safety_margin_gb
        self.cpu_safety_margin_gb = cpu_safety_margin_gb
        self.reprobe_every = reprobe_every
        # Self-imposed cap on FitLLM's OWN GPU usage (0 = no cap). This GPU is
        # shared with other tenants whose usage fluctuates independently of
        # us, so the cap is enforced against OUR OWN allocations (tracked via
        # torch.cuda.memory_reserved) rather than a snapshot of system-wide
        # free memory taken once at startup — that snapshot goes stale the
        # moment another tenant's usage changes.
        self.vram_limit_gb = vram_limit_gb

        self._call_count: int = 0
        self._cached_result: Dict | None = None
        self._cached_effective_n: int = 1

        # VRAM reserved by permanent GPU residents (lm_head + embed_tokens).
        # For Qwen2.5-32B: vocab_size=152064, hidden=5120 → ~3 GB in fp16.
        # These are NOT shard allocations, so we subtract them from our own
        # reserved count when computing headroom — otherwise lm_head+embed
        # exhaust the 4 GB budget before any shard can load.
        # Set by ShardedModel.from_pretrained() after moving lm_head/embed to GPU.
        self.baseline_reserved_gb: float = 0.0
        self._last_logged_alloc_gb: float = -1.0  # throttle diagnostic prints

    def update_baseline(self) -> None:
        """Re-snapshot baseline to current memory_allocated().

        Call this just before the shard layer loop starts (after embedding all
        batches and computing position_embeddings) so that h_all, pos_emb, and
        any other non-shard GPU residents are included in the baseline and don't
        eat into the 4 GB shard budget.
        """
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        current_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        if current_alloc_gb > self.baseline_reserved_gb + 0.01:
            import logging as _log
            _log.getLogger(__name__).info(
                f"Baseline updated: {self.baseline_reserved_gb:.3f}GB → {current_alloc_gb:.3f}GB "
                f"(+{current_alloc_gb - self.baseline_reserved_gb:.3f}GB absorbed; "
                f"h_all/pos_emb/other non-shard residents)"
            )
            self.baseline_reserved_gb = current_alloc_gb

    def free_vram_gb(self) -> float:
        """Return free VRAM in GB, capped by FitLLM's own vram_limit_gb budget
        (if set). Supports CUDA, MPS, and CPU-only."""
        if torch.cuda.is_available():
            try:
                # Release our own cached-but-unused allocator blocks before querying
                # system free. Without this, our ~3 GB of allocator cache shows up
                # as "used" in mem_get_info(), making the GPU appear full even though
                # we're not actively using those blocks. Especially important on shared
                # GPUs where every MB counts against other tenants.
                torch.cuda.empty_cache()
                free_bytes, _ = torch.cuda.mem_get_info()
            except RuntimeError:
                # The GPU can be so saturated by other tenants that even this
                # query fails (no room left for CUDA context bookkeeping).
                # Treat as zero free rather than crashing the caller.
                return 0.0
            free_gb = free_bytes / (1024 ** 3)
            if self.vram_limit_gb > 0:
                # Use memory_allocated() (actual tensors in use) not memory_reserved()
                # (which includes PyTorch's allocator cache blocks that inflate the
                # count by 2-4 GB even when nothing new is allocated).
                # Subtract baseline (lm_head + embed_tokens) so those permanent
                # GPU residents don't eat into the shard loading budget.
                our_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                shard_alloc_gb = max(0.0, our_alloc_gb - self.baseline_reserved_gb)
                budget_remaining_gb = max(0.0, self.vram_limit_gb - shard_alloc_gb)
                if abs(our_alloc_gb - self._last_logged_alloc_gb) > 0.01:
                    import logging as _log
                    _log.getLogger(__name__).info(
                        f"VRAM probe: sys_free={free_gb:.3f}GB our_alloc={our_alloc_gb:.3f}GB "
                        f"baseline={self.baseline_reserved_gb:.3f}GB shard_alloc={shard_alloc_gb:.3f}GB "
                        f"budget_rem={budget_remaining_gb:.3f}GB → returning {min(free_gb,budget_remaining_gb):.3f}GB"
                    )
                    self._last_logged_alloc_gb = our_alloc_gb
                return min(free_gb, budget_remaining_gb)
            return free_gb
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                total = torch.mps.recommended_max_memory()
                used = torch.mps.current_allocated_memory()
                return max(0.0, (total - used) / (1024 ** 3))
            except Exception:
                return 0.0
        return 0.0  # CPU-only device

    def free_cpu_ram_gb(self) -> float:
        """Return free CPU RAM in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)

    def compute_parallel_n(self) -> Dict:
        """
        Compute how many shards can fit in parallel on GPU/MPS and CPU.

        Returns a dict with:
          - gpu_parallel_n: shards fitting in GPU/MPS VRAM (after safety margin)
          - cpu_parallel_n: shards fitting in CPU RAM (after safety margin)
          - effective_n: layers to process per batch (0 triggers cpu_only)
          - strategy: 'full_model' | 'multi_shard' | 'single_shard' | 'cpu_only'
          - compute_device: 'cuda' | 'mps' | 'cpu'
          - free_gpu_gb: measured free GPU VRAM in GB
          - free_cpu_gb: measured free CPU RAM in GB
        """
        free_gpu = self.free_vram_gb()
        free_cpu = self.free_cpu_ram_gb()

        usable_gpu = max(0.0, free_gpu - self.gpu_safety_margin_gb)
        usable_cpu = max(0.0, free_cpu - self.cpu_safety_margin_gb)

        if self.shard_size_gb > 0:
            gpu_parallel_n = int(usable_gpu / self.shard_size_gb)
            cpu_parallel_n = int(usable_cpu / self.shard_size_gb)
        else:
            gpu_parallel_n = self.total_shards
            cpu_parallel_n = self.total_shards

        has_accel = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

        if has_accel and gpu_parallel_n > 0:
            effective_n = min(gpu_parallel_n, self.total_shards)
            compute_device = "cuda" if torch.cuda.is_available() else "mps"
        else:
            # GPU exhausted or no GPU: fall back to CPU execution
            effective_n = max(1, min(cpu_parallel_n, self.total_shards))
            compute_device = "cpu"

        if not has_accel or gpu_parallel_n == 0:
            strategy = "cpu_only"
        elif effective_n >= self.total_shards:
            strategy = "full_model"
        elif effective_n > 1:
            strategy = "multi_shard"
        else:
            strategy = "single_shard"

        return {
            "gpu_parallel_n": gpu_parallel_n,
            "cpu_parallel_n": cpu_parallel_n,
            "effective_n": effective_n,
            "strategy": strategy,
            "compute_device": compute_device,
            "free_gpu_gb": free_gpu,
            "free_cpu_gb": free_cpu,
        }

    def get_parallel_n(self) -> Dict:
        """
        Cached version of compute_parallel_n. Re-probes every `reprobe_every`
        calls. Emits a warning if parallel_n has decreased since last probe.
        """
        self._call_count += 1

        should_probe = (
            self._cached_result is None
            or self._call_count % self.reprobe_every == 0
        )

        if should_probe:
            new_result = self.compute_parallel_n()
            new_n = new_result["effective_n"]

            if self._cached_result is not None and new_n < self._cached_effective_n:
                warnings.warn(
                    f"FitLLM: parallel_n decreased from {self._cached_effective_n} "
                    f"to {new_n}. Available VRAM has been reduced "
                    f"(free GPU: {new_result['free_gpu_gb']:.2f} GB). "
                    "Consider reducing batch size or freeing VRAM.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            self._cached_result = new_result
            self._cached_effective_n = new_n

        return self._cached_result
