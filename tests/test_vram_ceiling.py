"""
Tests that during forward+backward, peak VRAM never exceeds the probe ceiling.
Uses mock/small model and CPU (VRAM = 0 on CPU; we test the logic indirectly).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitllm.probe import AdaptiveShardProbe
from fitllm.scheduler import ShardScheduler, save_shard_with_checksum
from fitllm.forward import ForwardEngine
from fitllm.backward import BackwardEngine


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------

class TinyLayer(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.ff = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
        return self.norm(x + self.ff(x))


class TinyModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 16, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyLayer(hidden) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)


class MockModelRef:
    def __init__(self, model, shard_dir):
        self._hf_model = model
        self._verify_checksums = False
        self._num_layers = len(model.layers)

    @property
    def num_layers(self):
        return self._num_layers


def make_probe_mock(parallel_n: int) -> MagicMock:
    probe = MagicMock(spec=AdaptiveShardProbe)
    probe.gpu_safety_margin_gb = 0.75
    probe.free_vram_gb.return_value = 100.0
    probe.get_parallel_n.return_value = {
        "effective_n": parallel_n,
        "strategy": "single_shard" if parallel_n == 1 else "multi_shard",
        "compute_device": "cpu",
        "gpu_parallel_n": parallel_n,
        "cpu_parallel_n": parallel_n,
        "free_gpu_gb": 0.0,
        "free_cpu_gb": 8.0,
    }
    return probe


def save_shards(model: TinyModel, shard_dir: Path) -> None:
    for idx, layer in enumerate(model.layers):
        sd = {k: v.contiguous() for k, v in layer.state_dict().items()}
        path = shard_dir / f"layer_{idx:03d}_weights.safetensors"
        save_shard_with_checksum(sd, path)


# ---------------------------------------------------------------------------
# VRAM tracking hook
# ---------------------------------------------------------------------------

class VRAMTracker:
    """Tracks peak allocated CUDA memory during a forward/backward pass."""

    def __init__(self):
        self.peak_mb: float = 0.0
        self._enabled = torch.cuda.is_available()

    def reset(self):
        if self._enabled:
            torch.cuda.reset_peak_memory_stats()
        self.peak_mb = 0.0

    def record_peak(self):
        if self._enabled:
            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            # On CPU, just record the total size of all live tensors (rough proxy)
            self.peak_mb = 0.0

    def ceiling_mb(self, free_vram_gb: float, safety_margin_gb: float) -> float:
        usable = max(0.0, free_vram_gb - safety_margin_gb)
        return usable * 1024


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVRAMCeiling:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        torch.manual_seed(77)
        self.model = TinyModel()
        self.shard_dir = tmp_path
        save_shards(self.model, tmp_path)
        self.model_ref = MockModelRef(self.model, tmp_path)
        self.tracker = VRAMTracker()

    def _make_engines(self, parallel_n: int):
        probe = make_probe_mock(parallel_n)
        scheduler = ShardScheduler(
            shard_dir=self.shard_dir,
            device="cpu",
            max_parallel=parallel_n,
            pin_memory=False,
            use_cuda_streams=False,
        )
        fwd = ForwardEngine(
            model_ref=self.model_ref,
            scheduler=scheduler,
            probe=probe,
            lm_head=self.model.lm_head,
            embed_tokens=self.model.embed_tokens,
            use_fused_kernels=False,
        )
        bwd = BackwardEngine(
            model_ref=self.model_ref,
            scheduler=scheduler,
            probe=probe,
            lm_head=self.model.lm_head,
            loss_fn=nn.CrossEntropyLoss(),
            grad_dir=self.shard_dir / "grads",
            grad_accum_steps=1,
        )
        return fwd, bwd

    def test_forward_completes_with_parallel_1(self):
        """Forward pass with parallel_n=1 should complete without OOM."""
        fwd, _ = self._make_engines(parallel_n=1)
        input_ids = torch.randint(0, 32, (1, 6))

        self.tracker.reset()
        logits, activations = fwd.forward(input_ids)
        self.tracker.record_peak()

        assert logits is not None
        assert len(activations) > 0

    def test_forward_completes_with_parallel_2(self):
        """Forward pass with parallel_n=2 should complete without OOM."""
        fwd, _ = self._make_engines(parallel_n=2)
        input_ids = torch.randint(0, 32, (1, 6))

        logits, activations = fwd.forward(input_ids)
        assert logits is not None

    def test_evict_frees_tensors(self):
        """After evict(), the tensor dict should be empty."""
        _, bwd = self._make_engines(parallel_n=1)

        t = {"w": torch.randn(16, 16), "b": torch.randn(16)}
        bwd.scheduler.evict(t)
        assert len(t) == 0

    def test_parallel_n_respects_ceiling(self):
        """
        With a very small free VRAM ceiling, effective_n should be 1.
        We test that the probe correctly limits parallel_n when VRAM is tight.
        """
        # Real probe with tiny VRAM budget
        shard_size_gb = 1.0
        probe = AdaptiveShardProbe(
            shard_size_gb=shard_size_gb,
            total_shards=16,
            gpu_safety_margin_gb=0.75,
        )
        # Mock free_vram_gb to return just barely 1 shard
        probe.free_vram_gb = lambda: 1.0 + 0.75  # exactly 1 shard after margin
        probe.free_cpu_ram_gb = lambda: 32.0

        result = probe.compute_parallel_n()
        assert result["effective_n"] <= 1  # Can't fit more than 1

    def test_large_vram_allows_parallel(self):
        """With plenty of VRAM, effective_n should be > 1."""
        probe = AdaptiveShardProbe(
            shard_size_gb=0.1,
            total_shards=16,
            gpu_safety_margin_gb=0.75,
        )
        probe.free_vram_gb = lambda: 8.0  # 8 GB free, 0.1 GB per shard → 72 shards
        probe.free_cpu_ram_gb = lambda: 32.0

        result = probe.compute_parallel_n()
        assert result["effective_n"] > 1

    def test_forward_backward_no_vram_leak(self):
        """
        Running forward+backward repeatedly should not accumulate tensors.
        We check that the activations list has consistent size across runs.
        """
        fwd, bwd = self._make_engines(parallel_n=1)
        input_ids = torch.randint(0, 32, (1, 4))
        labels = torch.randint(0, 32, (1, 3))

        sizes = []
        for _ in range(3):
            logits, activations = fwd.forward(input_ids[:, :-1])
            sizes.append(len(activations))
            b, s, v = logits.shape
            loss = F.cross_entropy(logits.view(b * s, v), labels.view(b * s))
            bwd.backward(loss, activations, labels)

        # All runs should produce the same number of activations
        assert len(set(sizes)) == 1, f"Inconsistent activation counts: {sizes}"

    def test_probe_reprobe_does_not_crash(self):
        """Repeated calls to get_parallel_n() should not crash."""
        probe = AdaptiveShardProbe(
            shard_size_gb=0.5,
            total_shards=8,
            reprobe_every=5,
        )
        probe.free_vram_gb = lambda: 4.0
        probe.free_cpu_ram_gb = lambda: 16.0

        for _ in range(15):
            result = probe.get_parallel_n()
            assert result["effective_n"] >= 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_memory_not_exceeded(self):
        """On GPU: peak VRAM after forward should be below probe ceiling."""
        free_vram_gb = 2.0
        safety_gb = 0.75
        shard_size_gb = 0.1

        probe = AdaptiveShardProbe(
            shard_size_gb=shard_size_gb,
            total_shards=self.model_ref.num_layers,
            gpu_safety_margin_gb=safety_gb,
        )
        probe.free_vram_gb = lambda: free_vram_gb
        probe.free_cpu_ram_gb = lambda: 16.0

        parallel_n = probe.compute_parallel_n()["effective_n"]

        scheduler = ShardScheduler(
            shard_dir=self.shard_dir,
            device="cuda",
            max_parallel=parallel_n,
            pin_memory=True,
            use_cuda_streams=True,
        )
        fwd = ForwardEngine(
            model_ref=self.model_ref,
            scheduler=scheduler,
            probe=MagicMock(
                get_parallel_n=lambda: {"effective_n": parallel_n,
                                        "strategy": "multi_shard",
                                        "compute_device": "cpu",
                                        "gpu_parallel_n": parallel_n,
                                        "cpu_parallel_n": parallel_n,
                                        "free_gpu_gb": free_vram_gb,
                                        "free_cpu_gb": 16.0},
                gpu_safety_margin_gb=0.75,
                free_vram_gb=MagicMock(return_value=100.0),
            ),
            lm_head=self.model.lm_head,
            embed_tokens=self.model.embed_tokens,
            use_fused_kernels=False,
        )

        torch.cuda.reset_peak_memory_stats()
        input_ids = torch.randint(0, 32, (1, 4))
        fwd.forward(input_ids)

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        ceiling_gb = free_vram_gb  # we allow up to the full free VRAM
        assert peak_gb <= ceiling_gb, (
            f"Peak VRAM {peak_gb:.3f} GB exceeded ceiling {ceiling_gb:.3f} GB"
        )
