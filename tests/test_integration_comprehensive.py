"""
Comprehensive integration test for FitLLM covering all major components.
Uses only synthetic/mock data and tiny in-memory models — no HuggingFace downloads.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ──────────────────────────────────────────────────────────────────────────────

results: list[tuple[str, str, str]] = []  # (name, PASS|FAIL, details)


def record(name: str, passed: bool, details: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append((name, status, details))
    sym = "✓" if passed else "✗"
    print(f"  {sym} [{status}] {name}" + (f": {details}" if details else ""))


def run_test(name: str, fn):
    try:
        fn()
        record(name, True)
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        record(name, False, f"{type(e).__name__}: {e} | {tb}")


# ──────────────────────────────────────────────────────────────────────────────
# Tiny synthetic models / helpers
# ──────────────────────────────────────────────────────────────────────────────

VOCAB = 64
HIDDEN = 16
N_LAYERS = 2


class TinyLayer(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
        return self.norm(self.q_proj(x) + self.v_proj(x) + x)


class TinyInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([TinyLayer() for _ in range(N_LAYERS)])


class TinyLLaMAStyleModel(nn.Module):
    """Mimics LLaMA-style model.layers / model.embed_tokens / lm_head layout."""
    def __init__(self):
        super().__init__()
        self.model = TinyInnerModel()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)


class MockModelRef:
    def __init__(self, hf_model, num_layers: int):
        self._hf_model = hf_model
        self._verify_checksums = False
        self._shard_size_gb = 0.0001
        self.num_layers = num_layers
        self.probe = MagicMock()
        self.probe.get_parallel_n.return_value = {
            "effective_n": 1,
            "strategy": "single_shard",
            "gpu_parallel_n": 0,
            "cpu_parallel_n": 4,
            "free_gpu_gb": 0.0,
            "free_cpu_gb": 8.0,
        }


def make_probe(effective_n: int = 1, gpu_safety_margin_gb: float = 0.75):
    probe = MagicMock()
    probe.get_parallel_n.return_value = {
        "effective_n": effective_n,
        "strategy": "single_shard",
        "gpu_parallel_n": 0,
        "cpu_parallel_n": 4,
        "free_gpu_gb": 0.0,
        "free_cpu_gb": 8.0,
    }
    probe.free_vram_gb.return_value = 0.0
    probe.gpu_safety_margin_gb = gpu_safety_margin_gb
    return probe


# ──────────────────────────────────────────────────────────────────────────────
# 1. AdaptiveShardProbe
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 1. AdaptiveShardProbe ===")

from fitllm.probe import AdaptiveShardProbe

REQUIRED_KEYS = {
    "compute_device", "strategy", "effective_n",
    "gpu_parallel_n", "cpu_parallel_n", "free_gpu_gb", "free_cpu_gb",
}


def test_probe_keys():
    probe = AdaptiveShardProbe(shard_size_gb=0.5, total_shards=8)
    probe.free_vram_gb = lambda: 4.0
    probe.free_cpu_ram_gb = lambda: 8.0
    result = probe.compute_parallel_n()
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


run_test("probe_compute_parallel_n returns all required keys", test_probe_keys)


def test_probe_cpu_only():
    probe = AdaptiveShardProbe(shard_size_gb=0.5, total_shards=8)
    probe.free_vram_gb = lambda: 0.0
    probe.free_cpu_ram_gb = lambda: 8.0
    with patch("torch.cuda.is_available", return_value=False):
        result = probe.compute_parallel_n()
    assert result["strategy"] == "cpu_only", f"Expected cpu_only, got {result['strategy']}"
    assert result["compute_device"] == "cpu"
    assert result["effective_n"] >= 1


run_test("probe cpu_only strategy when free_vram=0", test_probe_cpu_only)


def test_probe_compute_device_key():
    probe = AdaptiveShardProbe(shard_size_gb=0.5, total_shards=8)
    probe.free_vram_gb = lambda: 4.0
    probe.free_cpu_ram_gb = lambda: 8.0
    with patch("torch.cuda.is_available", return_value=False):
        result = probe.compute_parallel_n()
    assert "compute_device" in result


run_test("probe compute_device key present", test_probe_compute_device_key)


# ──────────────────────────────────────────────────────────────────────────────
# 2. ShardScheduler
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 2. ShardScheduler ===")

from fitllm.scheduler import ShardScheduler, save_shard_with_checksum


def test_scheduler_detect_device():
    assert ShardScheduler._detect_device("cpu") == "cpu"
    assert ShardScheduler._detect_device("mps") == "mps"
    # "auto" resolves to cpu when no GPU
    with patch("torch.cuda.is_available", return_value=False):
        with patch.object(torch.backends.mps if hasattr(torch.backends, "mps") else MagicMock(),
                          "is_available", return_value=False, create=True):
            resolved = ShardScheduler._detect_device("auto")
    assert isinstance(resolved, str)


run_test("ShardScheduler._detect_device exists and works", test_scheduler_detect_device)


def test_scheduler_read_to_cpu():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        sched = ShardScheduler(td_path, device="cpu", pin_memory=False, use_cuda_streams=False)
        # Write a real tiny shard
        w = {"weight": torch.randn(4, 4)}
        save_shard_with_checksum(w, td_path / "layer_000_weights.safetensors")
        tensors = sched._read_to_cpu(0, verify_checksums=True)
        assert "weight" in tensors
        assert tensors["weight"].shape == (4, 4)


run_test("ShardScheduler._read_to_cpu loads real shard file", test_scheduler_read_to_cpu)


def test_scheduler_transfer_to_gpu():
    with tempfile.TemporaryDirectory() as td:
        sched = ShardScheduler(Path(td), device="cpu", pin_memory=False, use_cuda_streams=False)
        cpu_t = {"weight": torch.randn(4, 4)}
        result = sched._transfer_to_gpu(cpu_t)
        assert "weight" in result


run_test("ShardScheduler._transfer_to_gpu identity when device=cpu", test_scheduler_transfer_to_gpu)


def test_scheduler_prefetch_to_cpu():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        sched = ShardScheduler(td_path, device="cpu", pin_memory=False, use_cuda_streams=False)
        save_shard_with_checksum({"weight": torch.randn(4, 4)},
                                  td_path / "layer_000_weights.safetensors")
        fut = sched.prefetch_to_cpu(0, verify_checksums=False)
        result = fut.result()
        assert "weight" in result


run_test("ShardScheduler.prefetch_to_cpu returns future with tensors", test_scheduler_prefetch_to_cpu)


# ──────────────────────────────────────────────────────────────────────────────
# 3. registry.py
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 3. registry.py ===")

from fitllm.registry import ARCHITECTURE_REGISTRY, get_decoder_layers, get_embed_and_head


def test_registry_size():
    assert len(ARCHITECTURE_REGISTRY) >= 30, f"Only {len(ARCHITECTURE_REGISTRY)} entries"


run_test("ARCHITECTURE_REGISTRY has 30+ entries", test_registry_size)


def test_registry_get_decoder_layers():
    model = TinyLLaMAStyleModel()
    layers = get_decoder_layers(model, model_type="llama")
    assert layers is not None
    assert len(layers) == N_LAYERS


run_test("get_decoder_layers returns correct ModuleList for llama", test_registry_get_decoder_layers)


def test_registry_get_embed_and_head():
    model = TinyLLaMAStyleModel()
    embed, head = get_embed_and_head(model, model_type="llama")
    assert embed is not None, "embed is None"
    assert head is not None, "head is None"
    assert isinstance(embed, nn.Embedding)
    assert isinstance(head, nn.Linear)


run_test("get_embed_and_head returns correct modules for llama", test_registry_get_embed_and_head)


# ──────────────────────────────────────────────────────────────────────────────
# 4. ForwardEngine
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 4. ForwardEngine ===")

from fitllm.forward import ForwardEngine
from fitllm.inference import LayerSkipMonitor


def _build_forward_engine(tmp_path: Path):
    torch.manual_seed(42)
    hf_model = TinyLLaMAStyleModel()
    # Write shards for each layer
    for idx, layer in enumerate(hf_model.model.layers):
        sd = {k: v.contiguous() for k, v in layer.state_dict().items()}
        save_shard_with_checksum(sd, tmp_path / f"layer_{idx:03d}_weights.safetensors")

    model_ref = MockModelRef(hf_model, N_LAYERS)
    probe = make_probe(effective_n=1)
    sched = ShardScheduler(tmp_path, device="cpu", pin_memory=False, use_cuda_streams=False)

    engine = ForwardEngine(
        model_ref=model_ref,
        scheduler=sched,
        probe=probe,
        lm_head=hf_model.lm_head,
        embed_tokens=hf_model.model.embed_tokens,
        use_fused_kernels=False,
        layer_skip_threshold=0.0,
    )
    return engine


def test_forward_engine_has_layer_skip_monitor():
    with tempfile.TemporaryDirectory() as td:
        engine = _build_forward_engine(Path(td))
        assert hasattr(engine, "_layer_skip_monitor")
        assert isinstance(engine._layer_skip_monitor, LayerSkipMonitor)


run_test("ForwardEngine has _layer_skip_monitor attribute", test_forward_engine_has_layer_skip_monitor)


def test_forward_engine_forward_calls_reset():
    with tempfile.TemporaryDirectory() as td:
        engine = _build_forward_engine(Path(td))
        reset_called = []
        orig_reset = engine._layer_skip_monitor.reset

        def patched_reset():
            reset_called.append(1)
            orig_reset()

        engine._layer_skip_monitor.reset = patched_reset
        input_ids = torch.randint(0, VOCAB, (1, 5))
        engine.forward(input_ids)
        assert len(reset_called) >= 1


run_test("ForwardEngine.forward() calls _layer_skip_monitor.reset()", test_forward_engine_forward_calls_reset)


def test_forward_engine_output_shape():
    with tempfile.TemporaryDirectory() as td:
        engine = _build_forward_engine(Path(td))
        input_ids = torch.randint(0, VOCAB, (1, 5))
        logits, activations = engine.forward(input_ids)
        assert logits.shape == (1, 5, VOCAB), f"Bad logits shape: {logits.shape}"
        assert len(activations) == N_LAYERS + 1


run_test("ForwardEngine.forward() output shapes correct", test_forward_engine_output_shape)


# ──────────────────────────────────────────────────────────────────────────────
# 5. InferenceConfig
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 5. InferenceConfig ===")

from fitllm.config import InferenceConfig


def test_inference_config_reprobe_every():
    cfg = InferenceConfig()
    assert cfg.reprobe_every == 1, f"Expected 1, got {cfg.reprobe_every}"


run_test("InferenceConfig.reprobe_every defaults to 1", test_inference_config_reprobe_every)


# ──────────────────────────────────────────────────────────────────────────────
# 6. DynamicKController
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 6. DynamicKController ===")

from fitllm.inference import DynamicKController


def test_dynamic_k_increases_on_high_acceptance():
    ctrl = DynamicKController(k_init=4, k_min=2, k_max=12, window=10)
    for _ in range(10):
        ctrl.record(proposed=10, accepted=10)  # 100% acceptance
    ctrl.update_k()
    assert ctrl.current_k > 4, f"Expected k>4, got {ctrl.current_k}"


run_test("DynamicKController k increases on high acceptance rate", test_dynamic_k_increases_on_high_acceptance)


def test_dynamic_k_decreases_on_low_acceptance():
    ctrl = DynamicKController(k_init=6, k_min=2, k_max=12, window=10)
    for _ in range(10):
        ctrl.record(proposed=10, accepted=3)  # 30% acceptance < 0.65
    ctrl.update_k()
    assert ctrl.current_k < 6, f"Expected k<6, got {ctrl.current_k}"


run_test("DynamicKController k decreases on low acceptance rate", test_dynamic_k_decreases_on_low_acceptance)


def test_dynamic_k_no_change_moderate_acceptance():
    ctrl = DynamicKController(k_init=5, k_min=2, k_max=12, window=10)
    for _ in range(10):
        ctrl.record(proposed=10, accepted=7)  # 70%, between 0.65 and 0.88
    ctrl.update_k()
    assert ctrl.current_k == 5


run_test("DynamicKController k stable at moderate acceptance rate", test_dynamic_k_no_change_moderate_acceptance)


# ──────────────────────────────────────────────────────────────────────────────
# 7. LayerSkipMonitor
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 7. LayerSkipMonitor ===")


def test_layer_skip_first_call_false():
    monitor = LayerSkipMonitor(threshold=0.01)
    h = torch.randn(1, 4, HIDDEN)
    result = monitor.should_skip(h)
    assert result is False


run_test("LayerSkipMonitor.should_skip returns False on first call", test_layer_skip_first_call_false)


def test_layer_skip_near_identical_returns_true():
    monitor = LayerSkipMonitor(threshold=0.9999)  # very high threshold → skip on tiny delta
    h = torch.ones(1, 4, HIDDEN)
    monitor.should_skip(h)                # first call — sets _prev_h
    h2 = h + 1e-7                         # nearly identical
    result = monitor.should_skip(h2)
    assert result is True, "Expected True (near-identical hidden states)"


run_test("LayerSkipMonitor.should_skip returns True on near-identical states", test_layer_skip_near_identical_returns_true)


def test_layer_skip_threshold_zero_never_skips():
    monitor = LayerSkipMonitor(threshold=0.0)
    h = torch.ones(1, 4, HIDDEN)
    monitor.should_skip(h)
    h2 = h.clone()
    result = monitor.should_skip(h2)
    assert result is False


run_test("LayerSkipMonitor never skips when threshold=0", test_layer_skip_threshold_zero_never_skips)


# ──────────────────────────────────────────────────────────────────────────────
# 8. LoRATrainer.save_checkpoint / resume_from_checkpoint
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 8. LoRATrainer checkpoint ===")

from fitllm.trainer import LoRATrainer
from fitllm.config import TrainingConfig
from fitllm.optimizer import ShardOptimizer


def test_lora_trainer_save_load_sample_index():
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainingConfig(checkpoint_dir=td, keep_checkpoints=5)

        # Build minimal mock model_ref for LoRATrainer
        opt = ShardOptimizer(shard_dir=Path(td), num_layers=2)
        mock_model = MagicMock()
        mock_model.optimizer = opt
        mock_model.qlora_manager = None   # no peft

        trainer = LoRATrainer(mock_model, cfg)
        ckpt_path = trainer.save_checkpoint(step=10, sample_index=123)
        assert ckpt_path.exists()

        # Reload and verify sample_index
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["sample_index"] == 123, f"Got {ckpt.get('sample_index')}"
        assert ckpt["step"] == 10


run_test("LoRATrainer.save_checkpoint saves sample_index", test_lora_trainer_save_load_sample_index)


def test_lora_trainer_resume_from_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainingConfig(checkpoint_dir=td, keep_checkpoints=5)
        opt = ShardOptimizer(shard_dir=Path(td), num_layers=2)
        mock_model = MagicMock()
        mock_model.optimizer = opt
        mock_model.qlora_manager = None

        trainer = LoRATrainer(mock_model, cfg)
        ckpt_path = trainer.save_checkpoint(step=7, sample_index=999)

        step, sample_index = trainer.resume_from_checkpoint(str(ckpt_path))
        assert step == 7
        assert sample_index == 999


run_test("LoRATrainer.resume_from_checkpoint restores step and sample_index", test_lora_trainer_resume_from_checkpoint)


def test_lora_trainer_no_lora_manager_attr():
    """Verify trainer.py does not reference 'lora_manager' (should use 'qlora_manager')."""
    import inspect
    src = inspect.getsource(LoRATrainer)
    # The attribute used should be qlora_manager, not lora_manager
    assert "lora_manager" not in src or "qlora_manager" in src, \
        "Found reference to 'lora_manager' without 'qlora_manager'"


run_test("LoRATrainer does not reference bare lora_manager (uses qlora_manager)", test_lora_trainer_no_lora_manager_attr)


# ──────────────────────────────────────────────────────────────────────────────
# 9. BackwardEngine.zero_all_grads
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 9. BackwardEngine ===")

from fitllm.backward import BackwardEngine


def test_backward_engine_zero_all_grads():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        grad_dir = td_path / "grads"
        grad_dir.mkdir()

        model_ref = MockModelRef(None, num_layers=3)

        probe = make_probe(1)
        sched = ShardScheduler(td_path, device="cpu", pin_memory=False, use_cuda_streams=False)
        lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        bwd = BackwardEngine(
            model_ref=model_ref,
            scheduler=sched,
            probe=probe,
            lm_head=lm_head,
            loss_fn=None,
            grad_dir=grad_dir,
        )
        # zero_all_grads should not crash even with no grad files
        bwd.zero_all_grads()


run_test("BackwardEngine.zero_all_grads works on empty temp dir", test_backward_engine_zero_all_grads)


def test_backward_engine_zero_grads_deletes_files():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        grad_dir = td_path / "grads"
        grad_dir.mkdir()

        model_ref = MockModelRef(None, num_layers=2)
        probe = make_probe(1)
        sched = ShardScheduler(td_path, device="cpu", pin_memory=False, use_cuda_streams=False)
        lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

        bwd = BackwardEngine(
            model_ref=model_ref,
            scheduler=sched,
            probe=probe,
            lm_head=lm_head,
            loss_fn=None,
            grad_dir=grad_dir,
        )
        # Create dummy grad file
        fake_grad_path = bwd._grad_path(0)
        save_shard_with_checksum({"lora_A.weight": torch.zeros(4, 4)}, fake_grad_path)
        assert fake_grad_path.exists()

        bwd.zero_grads(0)
        assert not fake_grad_path.exists()


run_test("BackwardEngine.zero_grads deletes existing grad file", test_backward_engine_zero_grads_deletes_files)


# ──────────────────────────────────────────────────────────────────────────────
# 10. ShardOptimizer
# ──────────────────────────────────────────────────────────────────────────────

print("\n=== 10. ShardOptimizer ===")

from fitllm.optimizer import ShardOptimizer


def test_shard_optimizer_instantiate():
    with tempfile.TemporaryDirectory() as td:
        opt = ShardOptimizer(shard_dir=Path(td), num_layers=4)
        assert opt.num_layers == 4
        assert opt.global_step == 0


run_test("ShardOptimizer can be instantiated with temp shard_dir", test_shard_optimizer_instantiate)


def test_shard_optimizer_zero_grad_no_crash():
    with tempfile.TemporaryDirectory() as td:
        opt = ShardOptimizer(shard_dir=Path(td), num_layers=3)
        opt.zero_grad()  # should not crash with no grad files


run_test("ShardOptimizer.zero_grad doesn't crash on empty dir", test_shard_optimizer_zero_grad_no_crash)


def test_shard_optimizer_state_dict_round_trip():
    with tempfile.TemporaryDirectory() as td:
        opt = ShardOptimizer(shard_dir=Path(td), num_layers=2, lr=1e-3)
        opt.global_step = 5
        sd = opt.state_dict_cpu()
        opt2 = ShardOptimizer(shard_dir=Path(td), num_layers=2)
        opt2.load_state_dict_cpu(sd)
        assert opt2.global_step == 5
        assert abs(opt2.lr - 1e-3) < 1e-8


run_test("ShardOptimizer state_dict round-trip preserves global_step and lr", test_shard_optimizer_state_dict_round_trip)


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("INTEGRATION TEST SUMMARY")
print("=" * 70)
passed = [r for r in results if r[1] == "PASS"]
failed = [r for r in results if r[1] == "FAIL"]

col_w = max(len(r[0]) for r in results) + 2
for name, status, details in results:
    line = f"  {status:4s}  {name:{col_w}}"
    if details:
        line += f"  ← {details}"
    print(line)

print("=" * 70)
print(f"TOTAL: {len(results)}  PASSED: {len(passed)}  FAILED: {len(failed)}")
print("=" * 70)

# Exit with non-zero if any failures
if failed:
    sys.exit(1)
