from __future__ import annotations

import ctypes
import gc
import hashlib
import logging
import platform
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def save_shard_with_checksum(state_dict: Dict[str, torch.Tensor], path: Path) -> None:
    """Save a safetensors shard and write a companion .sha256 checksum file."""
    path = Path(path)
    save_file(state_dict, str(path))
    sha256 = _compute_file_sha256(path)
    checksum_path = path.with_suffix(".sha256")
    checksum_path.write_text(sha256)


def load_shard_with_checksum(
    path: Path, verify: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Load a safetensors shard via memory-mapped safe_open.
    If verify=True and a .sha256 file exists, compare the checksum and raise
    RuntimeError on mismatch.
    """
    path = Path(path)
    checksum_path = path.with_suffix(".sha256")

    if verify and checksum_path.exists():
        expected = checksum_path.read_text().strip()
        actual = _compute_file_sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for shard {path}. "
                f"Expected {expected}, got {actual}. "
                "The file may be corrupted."
            )

    result: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)
    return result


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ShardScheduler:
    """
    Manages async prefetching and eviction of per-layer weight shards.
    Uses a ThreadPoolExecutor to overlap I/O with compute.

    Supports N-layer grouping (shard_group_size > 1): multiple layers stored
    per file, with tensors prefixed by their intra-group offset index.

    Tracks per-layer load times (EMA) and exposes prefetch_sorted() to submit
    the slowest shards first, reducing stall time in the forward/backward loops.
    """

    def __init__(
        self,
        shard_dir: Path,
        device: str = "cuda",
        max_parallel: int = 16,
        pin_memory: bool = True,
        use_cuda_streams: bool = True,
        shard_group_size: int = 1,
        num_layers: int = 0,
        gpu_safety_margin_gb: float = 0.75,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.device = self._detect_device(device)
        self.max_parallel = max_parallel
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.use_cuda_streams = use_cuda_streams and torch.cuda.is_available()
        self.shard_group_size = shard_group_size
        self.num_layers = num_layers
        self.gpu_safety_margin_gb = gpu_safety_margin_gb

        self._executor = ThreadPoolExecutor(max_workers=max(2, min(8, max_parallel)))
        # Allow up to 8 concurrent pinned buffer allocations — matches thread count
        # (old cap of 4 was starving half the I/O threads)
        self._semaphore = threading.Semaphore(max(4, min(8, max_parallel)))

        if self.use_cuda_streams:
            self._load_stream = torch.cuda.Stream()
        else:
            self._load_stream = None

        # Per-layer EMA load times for prefetch priority ordering
        self._layer_load_times: Dict[int, float] = {}
        self._load_time_alpha: float = 0.3

        # ── Tier 1: GPU layer cache ───────────────────────────────────────────────
        # Layers currently resident in VRAM. Evicts LRU when VRAM is full.
        # Insertion order = access order (move-to-end on hit for LRU).
        self._gpu_layer_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # ── Tier 2: CPU weight cache ──────────────────────────────────────────────
        # CPU-pinned tensors kept after forward pass for backward reuse.
        # Fully dynamic — sized to available RAM minus safety margin, no fixed cap.
        self._weight_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.cache_enabled: bool = False
        self._cpu_safety_margin_gb: float = 2.0  # always leave this much RAM free
        self._user_cap_gb: float = 0.0            # optional hard cap (0 = no cap)

    @staticmethod
    def _detect_device(requested: str) -> str:
        """Resolve device string, adding MPS fallback when CUDA is unavailable."""
        if requested in ("auto", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return requested

    def shard_path(self, layer_idx: int) -> Path:
        if self.shard_group_size == 1:
            return self.shard_dir / f"layer_{layer_idx:03d}_weights.safetensors"
        group_start = (layer_idx // self.shard_group_size) * self.shard_group_size
        group_end = min(group_start + self.shard_group_size, self.num_layers) - 1
        return self.shard_dir / f"layer_{group_start:03d}-{group_end:03d}_weights.safetensors"

    def _update_load_time(self, layer_idx: int, elapsed: float) -> None:
        """Update EMA load time for a layer."""
        prev = self._layer_load_times.get(layer_idx)
        self._layer_load_times[layer_idx] = (
            self._load_time_alpha * elapsed + (1 - self._load_time_alpha) * prev
            if prev is not None else elapsed
        )

    def _extract_layer_tensors(
        self, layer_idx: int, all_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract this layer's tensors from a possibly grouped shard."""
        if self.shard_group_size > 1:
            group_start = (layer_idx // self.shard_group_size) * self.shard_group_size
            offset = layer_idx - group_start
            prefix = f"{offset}."
            return {k[len(prefix):]: v for k, v in all_tensors.items() if k.startswith(prefix)}
        return all_tensors

    def _transfer_to_gpu(
        self, cpu_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Stage 2: CPU pinned RAM → GPU/MPS via CUDA stream (if available)."""
        if self.device == "cpu":
            return cpu_tensors
        result: Dict[str, torch.Tensor] = {}
        for k, v in cpu_tensors.items():
            if self._load_stream is not None:
                with torch.cuda.stream(self._load_stream):
                    v = v.to(self.device, non_blocking=True)
            else:
                v = v.to(self.device)
            result[k] = v
        if self._load_stream is not None:
            torch.cuda.current_stream().wait_stream(self._load_stream)
        return result

    def enable_weight_cache(self, cpu_safety_margin_gb: float = 2.0, user_cap_gb: float = 0.0) -> None:
        """
        Enable dynamic CPU weight cache.

        cpu_safety_margin_gb: always keep this much RAM free (default 2 GB).
        user_cap_gb: optional hard ceiling (0 = use all available RAM minus margin).
        """
        self.cache_enabled = True
        self._cpu_safety_margin_gb = cpu_safety_margin_gb
        self._user_cap_gb = user_cap_gb
        budget = self._cache_budget_gb()
        logger.info(
            f"Weight cache enabled — budget: {budget:.1f} GB "
            f"(free RAM: {self._free_ram_gb():.1f} GB, "
            f"safety margin: {cpu_safety_margin_gb:.1f} GB"
            + (f", user cap: {user_cap_gb:.1f} GB" if user_cap_gb > 0 else "")
            + ")"
        )

    def disable_weight_cache(self) -> None:
        self.cache_enabled = False

    def clear_weight_cache(self) -> None:
        """Evict all cached shards. Call after optimizer step (weights now stale)."""
        self._weight_cache.clear()
        gc.collect()
        logger.debug("Weight cache cleared")

    def warm_cache_async(self, num_layers: int, verify_checksums: bool = False) -> None:
        """
        Kick off background NVMe→CPU reload for all layers right after optimizer step.
        Runs on the I/O thread pool so it overlaps with the next mini-batch's
        tokenization and label masking — cache is warm before forward pass starts.
        """
        if not self.cache_enabled:
            return
        for layer_idx in range(num_layers):
            if layer_idx not in self._weight_cache:
                self._executor.submit(self._read_to_cpu, layer_idx, verify_checksums)
        logger.debug(f"Background cache warm-up started for {num_layers} layers")

    def _free_ram_gb(self) -> float:
        """Live probe of available system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            return float("inf")

    def _cache_budget_gb(self) -> float:
        """
        Compute how much RAM the cache is allowed to use right now.
        Re-probed live so it shrinks automatically if other processes take RAM.
        """
        free = self._free_ram_gb()
        budget = max(0.0, free - self._cpu_safety_margin_gb)
        if self._user_cap_gb > 0:
            budget = min(budget, self._user_cap_gb)
        return budget

    def _cache_size_gb(self) -> float:
        total = sum(t.nbytes for tensors in self._weight_cache.values() for t in tensors.values())
        return total / (1024 ** 3)

    def _evict_to_fit(self, needed_gb: float) -> None:
        """
        Evict oldest cached shards (FIFO) until there is room for needed_gb.
        Layers inserted first are furthest from being needed again — safe to drop.
        """
        for layer_idx in list(self._weight_cache.keys()):
            if self._cache_budget_gb() - self._cache_size_gb() >= needed_gb:
                break
            evicted = self._weight_cache.pop(layer_idx)
            del evicted
            gc.collect()
            logger.debug(f"Cache evicted layer {layer_idx} to free space")

    # ── 3-tier shard access: GPU → CPU → Disk ────────────────────────────────

    def get_layer_tensors(
        self, layer_idx: int, verify_checksums: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        3-tier shard lookup:
          Tier 1 — GPU cache  : already in VRAM, return immediately (fastest)
          Tier 2 — CPU cache  : in CPU RAM, DMA-transfer to GPU (no disk read)
          Tier 3 — NVMe disk  : cold miss, full NVMe→CPU→GPU load (slowest)

        Keeps GPU cache up to date with LRU eviction when VRAM is tight.
        """
        # Tier 1: GPU hit
        if layer_idx in self._gpu_layer_cache:
            # Move to end to mark as most-recently-used
            tensors = self._gpu_layer_cache.pop(layer_idx)
            self._gpu_layer_cache[layer_idx] = tensors
            logger.debug(f"GPU cache hit: layer {layer_idx}")
            return tensors

        # Tier 2: CPU cache hit → transfer to GPU
        if layer_idx in self._weight_cache:
            logger.debug(f"CPU cache hit: layer {layer_idx}")
            gpu_tensors = self._transfer_to_gpu(self._weight_cache[layer_idx])
            self._store_in_gpu_cache(layer_idx, gpu_tensors)
            return gpu_tensors

        # Tier 3: disk miss → full load
        logger.debug(f"Disk load: layer {layer_idx}")
        cpu_tensors = self._read_to_cpu(layer_idx, verify_checksums)
        gpu_tensors = self._transfer_to_gpu(cpu_tensors)
        self._store_in_gpu_cache(layer_idx, gpu_tensors)
        return gpu_tensors

    def _store_in_gpu_cache(
        self, layer_idx: int, tensors: Dict[str, torch.Tensor]
    ) -> None:
        """Add tensors to GPU cache, evicting LRU if VRAM is tight."""
        # Evict LRU GPU entries until we are not over VRAM budget
        # Use a conservative check: if device is CPU, no eviction needed
        if self.device == "cpu":
            self._gpu_layer_cache[layer_idx] = tensors
            return

        # Evict oldest (LRU = first key in ordered dict) until headroom exists
        shard_vram = sum(t.nbytes for t in tensors.values()) / (1024 ** 3)
        while self._gpu_cache_size_gb() + shard_vram > self._gpu_vram_budget_gb():
            if not self._gpu_layer_cache:
                break
            oldest_idx = next(iter(self._gpu_layer_cache))
            evicted = self._gpu_layer_cache.pop(oldest_idx)
            for t in evicted.values():
                del t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(f"GPU cache LRU evicted layer {oldest_idx}")

        self._gpu_layer_cache[layer_idx] = tensors

    def evict_layer_from_gpu(self, layer_idx: int) -> None:
        """Explicitly free a specific layer from GPU cache."""
        if layer_idx in self._gpu_layer_cache:
            evicted = self._gpu_layer_cache.pop(layer_idx)
            for t in evicted.values():
                del t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def clear_gpu_cache(self) -> None:
        """Free all GPU-cached layers."""
        for tensors in self._gpu_layer_cache.values():
            for t in tensors.values():
                del t
        self._gpu_layer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _gpu_cache_size_gb(self) -> float:
        total = sum(t.nbytes for tensors in self._gpu_layer_cache.values() for t in tensors.values())
        return total / (1024 ** 3)

    def _gpu_vram_budget_gb(self) -> float:
        """How much VRAM the GPU cache is allowed to use."""
        if not torch.cuda.is_available():
            return float("inf")
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        return max(0.0, free_gb - self.gpu_safety_margin_gb) if hasattr(self, "gpu_safety_margin_gb") else max(0.0, free_gb - 0.75)

    def _read_to_cpu(
        self, layer_idx: int, verify_checksums: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Stage 1: NVMe → CPU pinned RAM. Checks weight cache first."""
        if self.cache_enabled and layer_idx in self._weight_cache:
            logger.debug(f"Weight cache hit: layer {layer_idx}")
            return self._weight_cache[layer_idx]

        self._semaphore.acquire()
        t_start = time.monotonic()
        try:
            path = self.shard_path(layer_idx)
            if not path.exists():
                raise FileNotFoundError(f"Shard not found: {path}")
            all_tensors = load_shard_with_checksum(path, verify=verify_checksums)
            tensors = self._extract_layer_tensors(layer_idx, all_tensors)
            result: Dict[str, torch.Tensor] = {}
            for k, v in tensors.items():
                if self.pin_memory and v.device.type == "cpu":
                    v = v.pin_memory()
                result[k] = v

            if self.cache_enabled:
                new_shard_gb = sum(t.nbytes for t in result.values()) / (1024 ** 3)
                headroom = self._cache_budget_gb() - self._cache_size_gb()
                if headroom < new_shard_gb:
                    self._evict_to_fit(new_shard_gb)
                    headroom = self._cache_budget_gb() - self._cache_size_gb()
                if headroom >= new_shard_gb:
                    self._weight_cache[layer_idx] = result
                else:
                    logger.debug(f"Cache full: layer {layer_idx} will load from disk")

            return result
        finally:
            elapsed = time.monotonic() - t_start
            self._semaphore.release()
            self._update_load_time(layer_idx, elapsed)

    def _load_layer(
        self, layer_idx: int, verify_checksums: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Load a shard from disk to device (NVMe → CPU → GPU in two stages)."""
        cpu_tensors = self._read_to_cpu(layer_idx, verify_checksums)
        return self._transfer_to_gpu(cpu_tensors)

    def prefetch_to_cpu(
        self, layer_idx: int, verify_checksums: bool = True
    ) -> "Future[Dict[str, torch.Tensor]]":
        """Async Stage-1-only prefetch: NVMe → CPU. For triple-buffer read-ahead."""
        return self._executor.submit(self._read_to_cpu, layer_idx, verify_checksums)

    def prefetch(self, layer_idx: int, verify_checksums: bool = True) -> "Future[Dict[str, torch.Tensor]]":
        """
        Asynchronously load a shard. Returns a Future that resolves to
        a dict of tensors already on the target device.
        """
        return self._executor.submit(self._load_layer, layer_idx, verify_checksums)

    def prefetch_sorted(
        self,
        layer_indices: List[int],
        verify_checksums: bool = True,
    ) -> "List[Future[Dict[str, torch.Tensor]]]":
        """
        Submit prefetch futures ordered by descending estimated load time
        (slowest shards queued first) to minimize stall at result() call sites.

        Returns futures in the ORIGINAL order of layer_indices.
        """
        if len(layer_indices) <= 1:
            return [self.prefetch(i, verify_checksums) for i in layer_indices]

        known = list(self._layer_load_times.values())
        default_time = sum(known) / len(known) if known else 1.0

        sorted_indices = sorted(
            [i for i in layer_indices if i not in self._weight_cache],
            key=lambda i: self._layer_load_times.get(i, default_time),
            reverse=True,
        )

        future_map: Dict[int, Future] = {}
        for i in sorted_indices:
            future_map[i] = self._executor.submit(self._load_layer, i, verify_checksums)
        # Cached layers: wrap in an already-resolved future
        for i in layer_indices:
            if i in self._weight_cache:
                f: Future = Future()
                f.set_result(self._transfer_to_gpu(self._weight_cache[i]))
                future_map[i] = f
        return [future_map[i] for i in layer_indices]

    def evict(self, layer_tensors: Dict[str, torch.Tensor]) -> None:
        """
        Free GPU memory used by a loaded shard.
        Deletes tensor references, runs GC, clears CUDA cache, and on Linux
        calls malloc_trim to return freed heap pages to the OS.
        """
        for k in list(layer_tensors.keys()):
            t = layer_tensors.pop(k)
            del t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if platform.system() == "Linux":
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

    def shutdown(self) -> None:
        """Shut down the background thread pool."""
        self._executor.shutdown(wait=True)

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
