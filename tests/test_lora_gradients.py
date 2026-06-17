"""
Tests that after inject_lora and backward, only A and B matrices have
non-None gradients; base Linear weights remain frozen (requires_grad=False).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitllm.lora import LoRALinear, LoRAManager


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        attn = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        out = self.o_proj(attn)
        return self.norm(out + self.mlp(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 50, hidden: int = 16, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([SimpleTransformerLayer(hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_forward_shape_preserved(self):
        base = nn.Linear(16, 32, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        x = torch.randn(2, 8, 16)
        out = lora(x)
        assert out.shape == (2, 8, 32)

    def test_lora_output_differs_from_base(self):
        """LoRA adds a non-zero delta to the base output."""
        torch.manual_seed(1)
        base = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)

        x = torch.randn(1, 8, 16)
        base_out = base(x)
        lora_out = lora(x)

        # lora_B init is zeros, so initially delta=0; after perturbation:
        with torch.no_grad():
            lora.lora_B.data = torch.randn_like(lora.lora_B) * 0.1

        lora_out2 = lora(x)
        assert not torch.allclose(base_out, lora_out2, atol=1e-6)

    def test_base_weights_frozen(self):
        base = nn.Linear(8, 8, bias=False)
        lora = LoRALinear(base, rank=2, alpha=4.0)
        for p in lora.base.parameters():
            assert not p.requires_grad

    def test_lora_weights_trainable(self):
        base = nn.Linear(8, 8, bias=False)
        lora = LoRALinear(base, rank=2, alpha=4.0)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_A_init_nonzero(self):
        base = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert not torch.all(lora.lora_A == 0)

    def test_lora_B_init_zero(self):
        base = nn.Linear(16, 16, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert torch.all(lora.lora_B == 0)

    def test_scale_is_alpha_over_rank(self):
        base = nn.Linear(8, 8, bias=False)
        lora = LoRALinear(base, rank=4, alpha=8.0)
        assert lora.scale == 2.0  # 8.0 / 4


class TestLoRAManagerInjection:
    def test_inject_replaces_target_linears(self):
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))

        for layer in model.layers:
            assert isinstance(layer.q_proj, LoRALinear)
            assert isinstance(layer.v_proj, LoRALinear)

    def test_non_target_linears_unchanged(self):
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=4, alpha=8.0, targets=("q_proj",))

        for layer in model.layers:
            # k_proj, v_proj, o_proj, mlp should NOT be LoRALinear
            assert not isinstance(layer.k_proj, LoRALinear)
            assert not isinstance(layer.v_proj, LoRALinear)

    def test_only_lora_params_have_gradients_after_backward(self):
        """After freeze_base_model + backward, only lora_A/lora_B should have grads."""
        torch.manual_seed(42)
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=2, alpha=4.0, targets=("q_proj", "v_proj"))
        mgr.freeze_base_model(model)

        x = torch.randint(0, 50, (1, 6))
        labels = torch.randint(0, 50, (1, 6))

        logits = model(x)
        b, s, v = logits.shape
        loss = F.cross_entropy(logits.view(b * s, v), labels.view(b * s))
        loss.backward()

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.grad is not None, f"{name} should have gradient"
            else:
                assert param.grad is None, f"{name} should NOT have gradient (frozen)"

    def test_freeze_base_model_freezes_all_non_lora(self):
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=2, alpha=4.0, targets=("q_proj",))
        mgr.freeze_base_model(model)

        for name, param in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_state_dict_contains_only_lora_params(self):
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=2, alpha=4.0, targets=("q_proj", "v_proj"))

        sd = mgr.state_dict()
        for k in sd:
            assert "lora_A" in k or "lora_B" in k, f"Non-LoRA key in state_dict: {k}"

    def test_state_dict_round_trip(self):
        torch.manual_seed(5)
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=4, alpha=8.0, targets=("q_proj",))

        # Modify lora params
        for lora in mgr._lora_layers.values():
            with torch.no_grad():
                lora.lora_A.data.fill_(3.14)

        sd = mgr.state_dict()

        # Reset
        for lora in mgr._lora_layers.values():
            with torch.no_grad():
                lora.lora_A.data.zero_()

        # Restore
        mgr.load_state_dict(sd)

        for key, lora in mgr._lora_layers.items():
            assert torch.allclose(lora.lora_A, torch.full_like(lora.lora_A, 3.14))

    def test_trainable_parameters_yields_only_ab(self):
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=2, alpha=4.0, targets=("q_proj", "v_proj"))

        trainable = list(mgr.trainable_parameters())
        # 2 layers × 2 targets × 2 params (A, B) = 8 params
        assert len(trainable) == 8

    def test_lora_grad_norm_nonzero_after_backward(self):
        """Verify LoRA grad norms are nonzero after a backward pass."""
        torch.manual_seed(99)
        model = TinyTransformer()
        mgr = LoRAManager()
        mgr.inject_lora(model, rank=4, alpha=8.0, targets=("q_proj", "v_proj"))
        mgr.freeze_base_model(model)

        x = torch.randint(0, 50, (2, 8))
        labels = torch.randint(0, 50, (2, 8))
        logits = model(x)
        b, s, v = logits.shape
        loss = F.cross_entropy(logits.view(b * s, v), labels.view(b * s))
        loss.backward()

        total_grad_norm = sum(
            p.grad.norm().item()
            for p in mgr.trainable_parameters()
            if p.grad is not None
        )
        assert total_grad_norm > 0, "LoRA grad norms should be nonzero"
