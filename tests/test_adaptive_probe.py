"""Tests for AdaptiveShardProbe."""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from fitllm.probe import AdaptiveShardProbe


VALID_STRATEGIES = {"full_model", "multi_shard", "single_shard", "cpu_only"}
REQUIRED_KEYS = {"gpu_parallel_n", "cpu_parallel_n", "effective_n", "strategy", "compute_device", "free_gpu_gb", "free_cpu_gb"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_probe(
    shard_size_gb: float = 0.5,
    total_shards: int = 16,
    free_vram_gb: float = 4.0,
    free_cpu_gb: float = 8.0,
    reprobe_every: int = 50,
) -> AdaptiveShardProbe:
    probe = AdaptiveShardProbe(
        shard_size_gb=shard_size_gb,
        total_shards=total_shards,
        reprobe_every=reprobe_every,
    )
    # Patch the measurement methods
    probe.free_vram_gb = lambda: free_vram_gb
    probe.free_cpu_ram_gb = lambda: free_cpu_gb
    return probe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdaptiveShardProbeKeys:
    def test_returns_all_expected_keys(self):
        probe = make_probe()
        result = probe.compute_parallel_n()
        assert REQUIRED_KEYS == set(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - set(result.keys())}"
        )

    def test_effective_n_is_at_least_one(self):
        # Even with 0 free VRAM, effective_n must be >= 1
        probe = make_probe(free_vram_gb=0.0, free_cpu_gb=0.0)
        result = probe.compute_parallel_n()
        assert result["effective_n"] >= 1

    def test_effective_n_at_least_one_with_large_shard(self):
        probe = make_probe(shard_size_gb=100.0, free_vram_gb=0.5, free_cpu_gb=1.0)
        result = probe.compute_parallel_n()
        assert result["effective_n"] >= 1

    def test_strategy_is_valid(self):
        probe = make_probe()
        result = probe.compute_parallel_n()
        assert result["strategy"] in VALID_STRATEGIES


class TestAdaptiveShardProbeStrategies:
    def test_full_model_strategy(self):
        """When all shards fit in VRAM, strategy should be full_model."""
        probe = make_probe(
            shard_size_gb=0.1,
            total_shards=4,
            free_vram_gb=10.0,  # way more than needed
        )
        result = probe.compute_parallel_n()
        assert result["strategy"] == "full_model"
        assert result["effective_n"] >= 4

    def test_single_shard_strategy(self):
        """When only one shard fits, strategy should be single_shard."""
        probe = make_probe(
            shard_size_gb=0.5,
            total_shards=16,
            free_vram_gb=0.5 + 0.75 + 0.01,  # just barely fits 1 shard
        )
        result = probe.compute_parallel_n()
        assert result["strategy"] == "single_shard"
        assert result["effective_n"] == 1

    def test_multi_shard_strategy(self):
        """When a few shards fit, strategy should be multi_shard."""
        probe = make_probe(
            shard_size_gb=0.5,
            total_shards=16,
            free_vram_gb=2.0 + 0.75,  # fits 4 shards
        )
        result = probe.compute_parallel_n()
        # effective_n is min(gpu_parallel, total_shards)
        assert result["effective_n"] >= 2
        assert result["strategy"] in {"multi_shard", "full_model"}


class TestAdaptiveShardProbeCaching:
    def test_caches_result_between_calls(self):
        """get_parallel_n() should return cached result without re-probing."""
        probe = make_probe(reprobe_every=50)
        call_count = 0

        original_compute = probe.compute_parallel_n
        def counting_compute():
            nonlocal call_count
            call_count += 1
            return original_compute()

        probe.compute_parallel_n = counting_compute

        # First call should probe
        r1 = probe.get_parallel_n()
        assert call_count == 1

        # Subsequent calls up to reprobe_every should use cache
        for _ in range(48):
            probe.get_parallel_n()
        assert call_count == 1  # still only 1 compute call

        # Call at reprobe_every should re-probe
        probe.get_parallel_n()  # call #50
        assert call_count == 2

    def test_first_call_always_probes(self):
        probe = make_probe(reprobe_every=10)
        assert probe._cached_result is None
        probe.get_parallel_n()
        assert probe._cached_result is not None

    def test_reprobe_at_exact_interval(self):
        probe = make_probe(reprobe_every=5)
        compute_calls = []

        orig = probe.compute_parallel_n
        def tracking_compute():
            result = orig()
            compute_calls.append(1)
            return result

        probe.compute_parallel_n = tracking_compute

        # Calls 1..5: probe happens at 1 and 5
        for _ in range(5):
            probe.get_parallel_n()

        # Should have probed at call 1 and call 5
        assert len(compute_calls) == 2


class TestAdaptiveShardProbeWarning:
    def test_warns_when_parallel_n_decreases(self):
        """Should emit RuntimeWarning if effective_n decreases on re-probe."""
        probe = AdaptiveShardProbe(
            shard_size_gb=0.5,
            total_shards=16,
            reprobe_every=1,  # probe every call
        )

        call_num = 0
        def mock_compute():
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                # First probe: 8 shards fit
                return {
                    "gpu_parallel_n": 8,
                    "cpu_parallel_n": 16,
                    "effective_n": 8,
                    "strategy": "multi_shard",
                    "compute_device": "cuda",
                    "free_gpu_gb": 5.0,
                    "free_cpu_gb": 12.0,
                }
            else:
                # Subsequent probes: only 2 shards fit
                return {
                    "gpu_parallel_n": 2,
                    "cpu_parallel_n": 16,
                    "effective_n": 2,
                    "strategy": "multi_shard",
                    "compute_device": "cuda",
                    "free_gpu_gb": 1.75,
                    "free_cpu_gb": 12.0,
                }

        probe.compute_parallel_n = mock_compute

        # First call
        probe.get_parallel_n()

        # Second call should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            probe.get_parallel_n()
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "parallel_n decreased" in str(w[0].message)

    def test_no_warning_when_parallel_n_stable(self):
        probe = AdaptiveShardProbe(
            shard_size_gb=0.5,
            total_shards=16,
            reprobe_every=1,
        )
        probe.free_vram_gb = lambda: 8.0
        probe.free_cpu_ram_gb = lambda: 16.0

        probe.get_parallel_n()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            probe.get_parallel_n()
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert len(runtime_warnings) == 0


class TestAdaptiveShardProbeMockVRAM:
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.mem_get_info")
    @patch("psutil.virtual_memory")
    def test_uses_cuda_mem_get_info_when_available(
        self, mock_vm, mock_mem_info, mock_cuda
    ):
        """free_vram_gb() should call torch.cuda.mem_get_info."""
        free_bytes = int(4.0 * 1024 ** 3)
        total_bytes = int(8.0 * 1024 ** 3)
        mock_mem_info.return_value = (free_bytes, total_bytes)

        probe = AdaptiveShardProbe(shard_size_gb=0.5, total_shards=8)
        gb = probe.free_vram_gb()

        mock_mem_info.assert_called_once()
        assert abs(gb - 4.0) < 0.01

    @patch("torch.cuda.is_available", return_value=False)
    def test_returns_zero_vram_when_no_cuda(self, mock_cuda):
        """free_vram_gb() should return 0 when CUDA unavailable."""
        probe = AdaptiveShardProbe(shard_size_gb=0.5, total_shards=8)
        assert probe.free_vram_gb() == 0.0

    @patch("psutil.virtual_memory")
    def test_uses_psutil_for_cpu_ram(self, mock_vm):
        """free_cpu_ram_gb() should call psutil.virtual_memory()."""
        avail_bytes = int(16.0 * 1024 ** 3)
        mock_vm.return_value = MagicMock(available=avail_bytes)

        probe = AdaptiveShardProbe(shard_size_gb=0.5, total_shards=8)
        gb = probe.free_cpu_ram_gb()

        mock_vm.assert_called_once()
        assert abs(gb - 16.0) < 0.01
