"""
Tests that N-shard forward produces logits identical to 1-shard forward.
Uses a tiny mock model to keep tests fast.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from fitllm.probe import AdaptiveShardProbe
from fitllm.scheduler import ShardScheduler, save_shard_with_checksum
from fitllm.forward import ForwardEngine


# ---------------------------------------------------------------------------
# Tiny mock transformer architecture
# ---------------------------------------------------------------------------

class TinyDecoderLayer(nn.Module):
    """A minimal transformer-style decoder layer."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
        return self.ln(x + self.ff(x))


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 100, hidden_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden_size) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def build_tiny_model(num_layers: int = 4) -> TinyModel:
    torch.manual_seed(42)
    model = TinyModel(num_layers=num_layers)
    model.eval()
    return model


def save_model_shards(model: TinyModel, shard_dir: Path) -> None:
    for idx, layer in enumerate(model.layers):
        sd = {k: v.contiguous() for k, v in layer.state_dict().items()}
        path = shard_dir / f"layer_{idx:03d}_weights.safetensors"
        save_shard_with_checksum(sd, path)


class MockShardedModelRef:
    """Minimal object that ForwardEngine needs from its model_ref."""

    def __init__(self, model: TinyModel, shard_dir: Path, verify: bool = True):
        self._hf_model = model
        self._verify_checksums = verify
        self._num_layers = len(model.layers)

    @property
    def num_layers(self) -> int:
        return self._num_layers


def make_forward_engine(
    model: TinyModel,
    shard_dir: Path,
    parallel_n: int,
) -> ForwardEngine:
    device = "cpu"

    probe = MagicMock(spec=AdaptiveShardProbe)
    probe.gpu_safety_margin_gb = 0.75
    probe.free_vram_gb.return_value = 100.0
    probe.get_parallel_n.return_value = {
        "effective_n": parallel_n,
        "strategy": "multi_shard" if parallel_n > 1 else "single_shard",
        "compute_device": "cpu",
        "gpu_parallel_n": parallel_n,
        "cpu_parallel_n": parallel_n,
        "free_gpu_gb": 0.0,
        "free_cpu_gb": 8.0,
    }

    scheduler = ShardScheduler(
        shard_dir=shard_dir,
        device=device,
        max_parallel=parallel_n,
        pin_memory=False,
        use_cuda_streams=False,
    )

    model_ref = MockShardedModelRef(model, shard_dir)

    engine = ForwardEngine(
        model_ref=model_ref,
        scheduler=scheduler,
        probe=probe,
        lm_head=model.lm_head,
        embed_tokens=model.embed_tokens,
        use_fused_kernels=False,
    )
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiShardForwardEquivalence:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model = build_tiny_model(num_layers=4)
        save_model_shards(self.model, tmp_path)
        self.shard_dir = tmp_path

        # Reference: direct model forward
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            self.reference_logits = self.model(self.input_ids)

    def _run_forward_engine(self, parallel_n: int) -> torch.Tensor:
        engine = make_forward_engine(self.model, self.shard_dir, parallel_n)
        logits, activations = engine.forward(self.input_ids)
        return logits, activations

    def test_single_shard_matches_reference(self):
        logits, _ = self._run_forward_engine(parallel_n=1)
        assert torch.allclose(logits, self.reference_logits, atol=1e-4), (
            f"Max diff: {(logits - self.reference_logits).abs().max().item()}"
        )

    def test_two_shard_matches_reference(self):
        logits, _ = self._run_forward_engine(parallel_n=2)
        assert torch.allclose(logits, self.reference_logits, atol=1e-4), (
            f"Max diff: {(logits - self.reference_logits).abs().max().item()}"
        )

    def test_four_shard_matches_reference(self):
        logits, _ = self._run_forward_engine(parallel_n=4)
        assert torch.allclose(logits, self.reference_logits, atol=1e-4), (
            f"Max diff: {(logits - self.reference_logits).abs().max().item()}"
        )

    def test_full_model_matches_reference(self):
        # parallel_n >= num_layers
        logits, _ = self._run_forward_engine(parallel_n=8)
        assert torch.allclose(logits, self.reference_logits, atol=1e-4)

    def test_activations_length(self):
        _, activations = self._run_forward_engine(parallel_n=2)
        # activations[0] = embed, activations[1..4] = after each layer
        assert len(activations) == 5  # 4 layers + initial embed

    def test_activations_are_cpu_tensors(self):
        _, activations = self._run_forward_engine(parallel_n=2)
        for act in activations:
            assert act.device.type == "cpu"

    def test_logit_shape(self):
        logits, _ = self._run_forward_engine(parallel_n=1)
        b, s, v = logits.shape
        assert b == 1
        assert s == 8
        assert v == 100  # vocab_size

    def test_parallel_n_1_vs_2_identical(self):
        logits_1, _ = self._run_forward_engine(parallel_n=1)
        logits_2, _ = self._run_forward_engine(parallel_n=2)
        assert torch.allclose(logits_1, logits_2, atol=1e-4), (
            f"Max diff between parallel_n=1 and 2: "
            f"{(logits_1 - logits_2).abs().max().item()}"
        )

    def test_parallel_n_1_vs_3_identical(self):
        logits_1, _ = self._run_forward_engine(parallel_n=1)
        logits_3, _ = self._run_forward_engine(parallel_n=3)
        assert torch.allclose(logits_1, logits_3, atol=1e-4)
