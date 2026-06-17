"""
Tests for speculative decoding:
- DynamicKController acceptance rate logic
- LayerSkipMonitor cosine similarity logic
- AcceleratedInference fallback to greedy when no draft model
- Token acceptance/rejection mechanics
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fitllm.inference import (
    DynamicKController,
    LayerSkipMonitor,
    AcceleratedInference,
    _sample_token,
)
from fitllm.config import InferenceConfig


# ---------------------------------------------------------------------------
# Tests for DynamicKController
# ---------------------------------------------------------------------------

class TestDynamicKController:
    def test_initial_k(self):
        ctrl = DynamicKController(k_init=4, k_min=2, k_max=12)
        assert ctrl.current_k == 4

    def test_k_increases_on_high_acceptance(self):
        ctrl = DynamicKController(k_init=4, k_min=2, k_max=12, window=10)
        # Record 10 rounds with acceptance rate = 1.0 > 0.88
        for _ in range(10):
            ctrl.record(proposed=4, accepted=4)
        ctrl.update_k()
        assert ctrl.current_k == 5

    def test_k_decreases_on_low_acceptance(self):
        ctrl = DynamicKController(k_init=6, k_min=2, k_max=12, window=10)
        # Record 10 rounds with acceptance rate = 0.5 < 0.65
        for _ in range(10):
            ctrl.record(proposed=4, accepted=2)
        ctrl.update_k()
        assert ctrl.current_k == 5

    def test_k_stable_in_neutral_range(self):
        ctrl = DynamicKController(k_init=4, k_min=2, k_max=12, window=10)
        # Acceptance rate = 0.75 (between 0.65 and 0.88)
        for _ in range(10):
            ctrl.record(proposed=4, accepted=3)
        k_before = ctrl.current_k
        ctrl.update_k()
        assert ctrl.current_k == k_before

    def test_k_never_exceeds_k_max(self):
        ctrl = DynamicKController(k_init=12, k_min=2, k_max=12, window=10)
        for _ in range(10):
            ctrl.record(proposed=4, accepted=4)
        ctrl.update_k()
        assert ctrl.current_k <= 12

    def test_k_never_falls_below_k_min(self):
        ctrl = DynamicKController(k_init=2, k_min=2, k_max=12, window=10)
        for _ in range(10):
            ctrl.record(proposed=4, accepted=0)
        ctrl.update_k()
        assert ctrl.current_k >= 2

    def test_empty_history_no_crash(self):
        ctrl = DynamicKController()
        ctrl.update_k()  # Should not raise
        assert ctrl.current_k >= ctrl.k_min

    def test_rolling_window_eviction(self):
        ctrl = DynamicKController(k_init=4, k_min=2, k_max=12, window=5)
        # Fill with low acceptance
        for _ in range(5):
            ctrl.record(proposed=4, accepted=1)
        # Then add high acceptance (should evict the low ones)
        for _ in range(5):
            ctrl.record(proposed=4, accepted=4)
        ctrl.update_k()
        # After window full of high acceptance, k should increase
        assert ctrl.current_k >= 4


# ---------------------------------------------------------------------------
# Tests for LayerSkipMonitor
# ---------------------------------------------------------------------------

class TestLayerSkipMonitor:
    def test_first_call_never_skips(self):
        mon = LayerSkipMonitor(threshold=0.01)
        h = torch.randn(1, 8, 16)
        assert not mon.should_skip(h)

    def test_identical_states_can_skip(self):
        """If the hidden state doesn't change, it should be skippable."""
        mon = LayerSkipMonitor(threshold=0.01, skip_window=4)
        h = torch.randn(1, 8, 16)
        mon.should_skip(h)  # first call: no skip
        result = mon.should_skip(h)  # second call: identical, should skip
        assert result

    def test_different_states_not_skipped(self):
        """Significantly different hidden states should not be skipped."""
        mon = LayerSkipMonitor(threshold=0.01)
        h1 = torch.randn(1, 8, 16)
        h2 = -h1  # opposite direction, cosine sim = -1
        mon.should_skip(h1)
        result = mon.should_skip(h2)
        assert not result

    def test_skip_window_limits_consecutive_skips(self):
        """skip_window caps *consecutive* skips before forcing a non-skip reset."""
        mon = LayerSkipMonitor(threshold=0.5, skip_window=2)
        h = torch.randn(1, 8, 16)
        mon.should_skip(h)  # first call: no skip (initialises prev_h)

        # Collect results for 6 identical hidden states
        results = [mon.should_skip(h) for _ in range(6)]
        # Expected pattern with skip_window=2: T T F T T F
        # Never more than 2 consecutive Trues
        consecutive = 0
        for r in results:
            if r:
                consecutive += 1
                assert consecutive <= 2, "Consecutive skip count exceeded skip_window"
            else:
                consecutive = 0

    def test_zero_threshold_never_skips(self):
        """threshold=0.0 should never skip."""
        mon = LayerSkipMonitor(threshold=0.0)
        h = torch.randn(1, 8, 16)
        mon.should_skip(h)
        for _ in range(5):
            assert not mon.should_skip(h)

    def test_reset_clears_state(self):
        mon = LayerSkipMonitor(threshold=0.01, skip_window=4)
        h = torch.randn(1, 8, 16)
        mon.should_skip(h)
        mon.should_skip(h)  # should skip

        mon.reset()
        assert mon._prev_h is None
        # After reset, first call should not skip
        assert not mon.should_skip(h)


# ---------------------------------------------------------------------------
# Tests for _sample_token
# ---------------------------------------------------------------------------

class TestSampleToken:
    def test_low_temperature_is_greedy(self):
        logits = torch.tensor([[0.0, 10.0, 0.0]])  # token 1 is most likely
        tok = _sample_token(logits, temperature=1e-6)
        assert tok.item() == 1

    def test_output_shape(self):
        logits = torch.randn(2, 50)
        tok = _sample_token(logits, temperature=1.0)
        assert tok.shape == (2, 1)

    def test_token_in_valid_range(self):
        vocab = 100
        logits = torch.randn(3, vocab)
        tok = _sample_token(logits, temperature=1.0)
        assert (tok >= 0).all()
        assert (tok < vocab).all()


# ---------------------------------------------------------------------------
# Tests for AcceleratedInference (greedy fallback + reset cache)
# ---------------------------------------------------------------------------

class TinyVerifier:
    """Minimal verifier mock that produces deterministic outputs."""

    def __init__(self, vocab: int = 20, hidden: int = 8, seq_len_out: int = 5):
        self.vocab = vocab
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = vocab - 1
        self._seq_len_out = seq_len_out
        # forward_engine stub needed by _greedy_generate
        self.forward_engine = MagicMock()
        self.forward_engine.reset_kv_cache = MagicMock()

    def forward(self, input_ids):
        b, s = input_ids.shape
        logits = torch.zeros(b, s, self.vocab)
        # Always predict token 1 (make it very likely)
        logits[:, :, 1] = 10.0
        # Fake activations
        activations = [torch.randn(b, s, 8)]
        return logits, activations

    def forward_with_kvcache(self, input_ids, attention_mask=None, step: int = 0):
        b, s = input_ids.shape
        logits = torch.zeros(b, s, self.vocab, device=input_ids.device)
        logits[:, :, 1] = 10.0
        return logits


class TestAcceleratedInferenceGreedy:
    def setup_method(self):
        self.verifier = TinyVerifier()
        self.config = InferenceConfig(
            draft_model=None,
            speculative_k=3,
            dynamic_k=False,
            max_new_tokens=5,
            temperature=0.01,
        )
        self.inf = AcceleratedInference(
            verifier=self.verifier,
            draft_model_name=None,  # triggers greedy fallback
            inference_config=self.config,
            scheduler=MagicMock(),
            probe=MagicMock(),
        )

    def test_greedy_generate_returns_tensor(self):
        input_ids = torch.tensor([[0, 1, 2]])
        output = self.inf.speculative_generate(input_ids, max_new_tokens=5, temperature=0.01)
        assert isinstance(output, torch.Tensor)

    def test_greedy_generates_expected_length(self):
        input_ids = torch.tensor([[0, 1, 2]])
        output = self.inf.speculative_generate(input_ids, max_new_tokens=5, temperature=0.01)
        # Output should be longer than input
        assert output.shape[1] >= input_ids.shape[1]

    def test_greedy_generates_mostly_token_1(self):
        """With temp≈0 and logits dominated by token 1, output should be mostly 1s."""
        input_ids = torch.tensor([[0]])
        output = self.inf.speculative_generate(input_ids, max_new_tokens=5, temperature=1e-6)
        new_tokens = output[0, 1:].tolist()
        # Most tokens should be 1 (EOS is 19, which could also appear)
        assert all(t in (1, self.verifier.tokenizer.eos_token_id) for t in new_tokens)

    def test_reset_draft_cache(self):
        inf = self.inf
        inf.draft_past_kv = {"fake": "cache"}
        inf.draft_context_len = 10
        inf.reset_draft_cache()
        assert inf.draft_past_kv is None
        assert inf.draft_context_len == 0

    def test_no_draft_model_sets_none(self):
        assert self.inf.draft_model is None

    def test_dynamic_k_controller_not_created_when_disabled(self):
        assert self.inf.k_controller is None

    def test_dynamic_k_controller_created_when_enabled(self):
        config = InferenceConfig(dynamic_k=True, speculative_k=4)
        inf = AcceleratedInference(
            verifier=self.verifier,
            draft_model_name=None,
            inference_config=config,
            scheduler=MagicMock(),
            probe=MagicMock(),
        )
        assert inf.k_controller is not None
        assert inf.k_controller.current_k == 4
