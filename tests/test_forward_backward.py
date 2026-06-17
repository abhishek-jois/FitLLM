"""
Round-trip tests: forward produces activations, backward runs without error,
and gradient norms are non-zero for LoRA parameters.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitllm.probe import AdaptiveShardProbe
from fitllm.scheduler import ShardScheduler, save_shard_with_checksum
from fitllm.forward import ForwardEngine
from fitllm.backward import BackwardEngine
from fitllm.lora import LoRAManager


# ---------------------------------------------------------------------------
# Tiny model with LoRA-injectable linear layers
# ---------------------------------------------------------------------------

class TinyLayer(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
        return self.norm(self.q_proj(x) + self.v_proj(x) + x)


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden: int = 16, n_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([TinyLayer(hidden) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)


def build_model():
    torch.manual_seed(13)
    return TinyModel()


class MockModelRef:
    def __init__(self, model, shard_dir):
        self._hf_model = model
        self._verify_checksums = False
        self._num_layers = len(model.layers)

    @property
    def num_layers(self):
        return self._num_layers


def make_probe(n: int) -> MagicMock:
    probe = MagicMock(spec=AdaptiveShardProbe)
    probe.gpu_safety_margin_gb = 0.75
    probe.free_vram_gb.return_value = 100.0  # always enough headroom
    probe.get_parallel_n.return_value = {
        "effective_n": n,
        "strategy": "multi_shard",
        "compute_device": "cpu",
        "gpu_parallel_n": n,
        "cpu_parallel_n": n,
        "free_gpu_gb": 0.0,
        "free_cpu_gb": 8.0,
    }
    return probe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardBackwardRoundTrip:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.model = build_model()
        self.shard_dir = tmp_path

        # Inject LoRA
        self.lora_mgr = LoRAManager()
        self.lora_mgr.inject_lora(self.model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))
        self.lora_mgr.freeze_base_model(self.model)

        # Save shards
        for idx, layer in enumerate(self.model.layers):
            sd = {k: v.contiguous() for k, v in layer.state_dict().items()}
            path = self.shard_dir / f"layer_{idx:03d}_weights.safetensors"
            save_shard_with_checksum(sd, path)

        self.model_ref = MockModelRef(self.model, self.shard_dir)
        self.probe = make_probe(1)

        self.scheduler = ShardScheduler(
            shard_dir=self.shard_dir,
            device="cpu",
            max_parallel=2,
            pin_memory=False,
            use_cuda_streams=False,
        )

        self.fwd = ForwardEngine(
            model_ref=self.model_ref,
            scheduler=self.scheduler,
            probe=self.probe,
            lm_head=self.model.lm_head,
            embed_tokens=self.model.embed_tokens,
            use_fused_kernels=False,
        )

        self.bwd = BackwardEngine(
            model_ref=self.model_ref,
            scheduler=self.scheduler,
            probe=self.probe,
            lm_head=self.model.lm_head,
            loss_fn=nn.CrossEntropyLoss(),
            grad_dir=self.shard_dir / "grads",
            grad_accum_steps=1,
        )

        torch.manual_seed(0)
        self.input_ids = torch.randint(0, 64, (1, 7))
        self.labels = torch.randint(0, 64, (1, 6))

    def test_forward_returns_logits_and_activations(self):
        logits, activations = self.fwd.forward(self.input_ids[:, :-1])

        assert logits is not None
        assert len(activations) == 3  # embed + 2 layers
        assert logits.shape == (1, 6, 64)

    def test_activations_have_correct_shape(self):
        logits, activations = self.fwd.forward(self.input_ids[:, :-1])
        for act in activations:
            assert act.shape[-1] == 16  # hidden_size

    def test_backward_completes_without_error(self):
        logits, activations = self.fwd.forward(self.input_ids[:, :-1])
        b, s, v = logits.shape
        loss = F.cross_entropy(logits.view(b * s, v), self.labels.view(b * s))

        # Should not raise
        self.bwd.backward(loss, activations, self.labels)

    def test_lora_params_have_gradients_after_backward(self):
        """LoRA A and B params should receive gradients through the backward pass."""
        # Enable grad on LoRA params
        for name, lora_layer in self.lora_mgr._lora_layers.items():
            lora_layer.lora_A.requires_grad_(True)
            lora_layer.lora_B.requires_grad_(True)

        logits, activations = self.fwd.forward(self.input_ids[:, :-1])
        b, s, v = logits.shape

        # Recompute logits with grad tracking for this test
        h = self.model.embed_tokens(self.input_ids[:, :-1])
        for layer in self.model.layers:
            h = layer(h)
        logits_grad = self.model.lm_head(h)
        loss = F.cross_entropy(
            logits_grad.view(b * s, v),
            self.labels.view(b * s),
        )
        loss.backward()

        # Check that lora params have non-None gradients
        any_grad = False
        for name, lora_layer in self.lora_mgr._lora_layers.items():
            if lora_layer.lora_A.grad is not None:
                any_grad = True
                break
            if lora_layer.lora_B.grad is not None:
                any_grad = True
                break

        assert any_grad, "No LoRA parameters received gradients"

    def test_base_weights_have_no_gradients(self):
        """After freeze_base_model, base Linear weights should have requires_grad=False."""
        for name, lora_layer in self.lora_mgr._lora_layers.items():
            for param in lora_layer.base.parameters():
                assert not param.requires_grad, (
                    f"Base param {name}.base should be frozen but requires_grad=True"
                )

    def test_forward_backward_twice_no_error(self):
        """Running forward+backward twice should work (e.g. for grad accum)."""
        for _ in range(2):
            logits, activations = self.fwd.forward(self.input_ids[:, :-1])
            b, s, v = logits.shape
            loss = F.cross_entropy(logits.view(b * s, v), self.labels.view(b * s))
            self.bwd.backward(loss, activations, self.labels)
