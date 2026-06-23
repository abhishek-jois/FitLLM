from __future__ import annotations

import concurrent.futures as _cf
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .kernels import apply_fused_kernels
from .probe import AdaptiveShardProbe
from .registry import get_decoder_layers as _registry_get_decoder_layers
from .scheduler import ShardScheduler

if TYPE_CHECKING:
    from .model import ShardedModel

logger = logging.getLogger(__name__)


def _restore_bnb_quant_state(layer: nn.Module, tensors: dict) -> None:
    """Rebuild NF4 QuantState for every Params4bit in the layer after load_state_dict.

    load_state_dict only copies the raw 4-bit bytes (uint8); it ignores the
    companion tensors like 'weight.absmax', 'weight.nested_quant_map', etc. that
    are stored as flat keys in the shard dict.  Without the QuantState the
    bitsandbytes dequantization uses wrong scale factors, producing wildly
    out-of-range hidden states and logits (observed: logit range [-406, 512]
    instead of the normal [-10, 10]).

    This function reconstructs the QuantState from those companion tensors and
    attaches it to each Params4bit parameter so that dequantization is correct.
    """
    try:
        from bitsandbytes.nn import Params4bit
        from bitsandbytes.functional import QuantState
    except ImportError:
        return

    for param_name, param in layer.named_parameters():
        if not isinstance(param, Params4bit):
            continue
        prefix = param_name + "."
        qs_dict = {
            k[len(prefix):]: v
            for k, v in tensors.items()
            if k.startswith(prefix)
        }
        if not qs_dict:
            continue
        try:
            param.quant_state = QuantState.from_dict(qs_dict, device=param.device)
        except Exception as e:
            logger.debug(f"Could not restore quant_state for {param_name}: {e}")


class _FunctionalRMSNorm(nn.Module):
    """CPU-safe RMSNorm used to replace LigerRMSNorm when compute_device is CPU."""

    def __init__(self, weight: nn.Parameter, eps: float):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


def _strip_liger_patches(layer: nn.Module) -> None:
    """Replace LigerRMSNorm submodules with a CPU-safe equivalent.

    _reconstruct_layer_module returns the actual cached layer object, so
    apply_fused_kernels patches it in-place.  When compute_device later
    switches to CPU (VRAM probe falls below threshold), the cached layer still
    carries LigerRMSNorm — calling it with CPU tensors fires a Triton kernel
    and crashes.  This function reverts those patches before the layer is
    moved to CPU.
    """
    try:
        import liger_kernel.transformers as _lk
        LigerRMSNorm = getattr(_lk, "LigerRMSNorm", None)
        if LigerRMSNorm is None:
            return
    except Exception:
        return

    modules_dict = dict(layer.named_modules())
    replacements: dict = {}
    for name, child in modules_dict.items():
        if isinstance(child, LigerRMSNorm):
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
            replacements[name] = _FunctionalRMSNorm(child.weight, eps)

    for name, new_mod in replacements.items():
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = modules_dict[parts[0]]
            setattr(parent, parts[1], new_mod)
        else:
            setattr(layer, name, new_mod)


def _resolve_compute_device(
    probe: AdaptiveShardProbe,
    scheduler: ShardScheduler,
    shard_size_gb: float,
    gpu_device: torch.device,
    max_wait_seconds: float = 600.0,
) -> torch.device:
    """Pick where to run this layer's compute, waiting out transient VRAM
    contention rather than immediately falling back to CPU.

    This GPU is shared with other tenants (vLLM, RL rollout/model-service
    processes) whose usage can swing from a few GB to fully saturated within
    seconds. bitsandbytes 4-bit matmul has no viable CPU fallback path —
    landing there silently stalls training for hours on a single layer. The
    VRAM headroom we actually need (shard_size * 1.2, generally well under
    1GB) typically frees up again within seconds as the other tenant's
    batches cycle, so waiting up to max_wait_seconds is far cheaper than a
    CPU stall. Only after that genuinely sustained saturation do we fall back.
    """
    if not (scheduler.device != "cpu" and torch.cuda.is_available()):
        return torch.device("cpu")

    deadline = time.monotonic() + max_wait_seconds
    delay = 0.5
    last_log = 0.0
    free_accel = probe.free_vram_gb() - probe.gpu_safety_margin_gb
    while free_accel < shard_size_gb * 1.2:
        now = time.monotonic()
        if now >= deadline:
            logger.warning(
                f"GPU VRAM contention persisted for {max_wait_seconds:.0f}s "
                f"(free={free_accel + probe.gpu_safety_margin_gb:.2f}GB, need "
                f"{shard_size_gb * 1.2:.2f}GB) — falling back to CPU compute "
                f"for this layer. This will be extremely slow (no CPU path "
                f"for 4-bit matmul)."
            )
            return torch.device("cpu")
        if now - last_log >= 15.0:
            logger.info(
                f"Waiting on GPU VRAM (other tenants busy): free="
                f"{free_accel + probe.gpu_safety_margin_gb:.2f}GB, need "
                f"{shard_size_gb * 1.2:.2f}GB — retrying for up to "
                f"{deadline - now:.0f}s more"
            )
            last_log = now
        time.sleep(min(delay, deadline - now))
        delay = min(delay * 1.5, 5.0)
        free_accel = probe.free_vram_gb() - probe.gpu_safety_margin_gb

    return gpu_device


def _make_resolved_future(value):
    """Return a Future already resolved with value."""
    f = _cf.Future()
    f.set_result(value)
    return f


class ForwardEngine:
    """
    Executes a layer-by-layer forward pass through a sharded transformer model.

    Layers are loaded in batches determined by the AdaptiveShardProbe.
    Pipelining overlaps loading the next batch with compute on the current batch.
    Activations (hidden states) are saved to CPU after each layer for use by
    the BackwardEngine.

    Also provides forward_with_kvcache() for autoregressive inference, which
    stores per-layer KV state on CPU between decode steps.
    """

    def __init__(
        self,
        model_ref: "ShardedModel",
        scheduler: ShardScheduler,
        probe: AdaptiveShardProbe,
        lm_head: nn.Module,
        embed_tokens: nn.Module,
        use_fused_kernels: bool = True,
        layer_skip_threshold: float = 0.0,
        mixed_precision: bool = True,
        final_norm: Optional[nn.Module] = None,
    ) -> None:
        self.model_ref = model_ref
        self.scheduler = scheduler
        self.probe = probe
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens
        self.final_norm = final_norm
        self.use_fused_kernels = use_fused_kernels
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self._amp_dtype = (
            torch.bfloat16 if (self.mixed_precision and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        # CPU-side KV cache for autoregressive generation
        self._kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        # Layer skip monitor (cosine-similarity gating)
        from .inference import LayerSkipMonitor
        self._layer_skip_monitor = LayerSkipMonitor(threshold=layer_skip_threshold)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a full forward pass through all sharded layers.

        Returns:
            logits: (batch, seq_len, vocab_size) final logits
            activations: list of CPU tensors, one per layer boundary
        """
        device = next(self.embed_tokens.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        _amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if self.mixed_precision else torch.autocast("cpu", enabled=False)
        )

        with torch.no_grad(), _amp_ctx:
            h = self.embed_tokens(input_ids)

        # Compute rotary position embeddings once for all layers (newer transformers)
        position_embeddings = self._compute_position_embeddings(h, position_ids, device)

        activations: List[torch.Tensor] = []
        activations.append(h.detach().cpu())

        num_layers = self.model_ref.num_layers
        probe_result = self.probe.get_parallel_n()
        parallel_n = probe_result["effective_n"]

        verify = getattr(self.model_ref, "_verify_checksums", True)
        shard_size = getattr(self.model_ref, "_shard_size_gb", 0.5)

        self._layer_skip_monitor.reset()

        # Triple-buffer pipeline: Stage A = NVMe→CPU, Stage B = CPU→GPU, Stage C = compute
        # Kick off Stage A prefetch for batches 0 and 1 before the loop starts
        def _cpu_futures_for_batch(start: int) -> list:
            end = min(start + parallel_n, num_layers)
            indices = list(range(start, end))
            return [self.scheduler.prefetch_to_cpu(i, verify) for i in indices], indices

        cpu_futures_cur, cur_indices = _cpu_futures_for_batch(0)
        cpu_futures_nxt = None
        nxt_indices = []
        if parallel_n < num_layers:
            cpu_futures_nxt, nxt_indices = _cpu_futures_for_batch(parallel_n)

        batch_start = 0
        while batch_start < num_layers:
            batch_end = min(batch_start + parallel_n, num_layers)
            batch_indices = list(range(batch_start, batch_end))

            # Stage A for batch+2 (issue while we work on batch)
            next2_start = batch_end + parallel_n
            if next2_start < num_layers:
                cpu_futures_nn, _ = _cpu_futures_for_batch(next2_start)
            else:
                cpu_futures_nn = None

            # Stage B: transfer current batch CPU→GPU (collect Stage A result first)
            compute_device = _resolve_compute_device(self.probe, self.scheduler, shard_size, device)

            # Wait for Stage A of current batch, then transfer
            cpu_tensors_list = [f.result() for f in cpu_futures_cur]
            batch_tensors = [
                self.scheduler._transfer_to_gpu(ct) if compute_device != torch.device("cpu")
                else ct
                for ct in cpu_tensors_list
            ]

            # Stage B for next batch (overlaps with our compute below)
            if cpu_futures_nxt is not None:
                nxt_cpu_tensors = [f.result() for f in cpu_futures_nxt]
                # submit GPU transfer for next batch asynchronously
                gpu_futures_nxt = [
                    self.scheduler._executor.submit(self.scheduler._transfer_to_gpu, ct)
                    for ct in nxt_cpu_tensors
                ]
            else:
                gpu_futures_nxt = None
                nxt_cpu_tensors = []

            # Stage C: GPU compute on current batch
            h = h.to(compute_device)
            for local_idx, layer_tensors in enumerate(batch_tensors):
                layer_idx = batch_start + local_idx
                layer = self._reconstruct_layer_module(layer_idx, layer_tensors)

                if self._layer_skip_monitor.should_skip(h):
                    activations.append(h.detach().cpu())
                    del layer
                    continue

                # Manage Liger patches: apply on GPU, strip on CPU (Triton is CUDA-only).
                # cached_layers are shared objects — patches from prior CUDA steps persist.
                if compute_device == torch.device("cpu"):
                    _strip_liger_patches(layer)
                elif self.use_fused_kernels:
                    layer = apply_fused_kernels(layer)

                layer = layer.to(compute_device)

                with torch.no_grad(), _amp_ctx:
                    output = self._run_layer(layer, h, attention_mask, position_ids, position_embeddings)
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output

                activations.append(h.detach().cpu())
                del layer

            for layer_tensors in batch_tensors:
                self.scheduler.evict(layer_tensors)

            batch_start = batch_end

            # Advance pipeline: next batch's GPU tensors become current
            if gpu_futures_nxt is not None:
                # collect already-transferred tensors for next iteration
                cpu_futures_cur = [_make_resolved_future(f.result()) for f in gpu_futures_nxt]
            elif cpu_futures_nxt is not None:
                cpu_futures_cur = [_make_resolved_future(ct) for ct in nxt_cpu_tensors]
            else:
                cpu_futures_cur = []

            cpu_futures_nxt = cpu_futures_nn

        lm_head_device = next(self.lm_head.parameters()).device
        h = h.to(lm_head_device)
        with torch.no_grad(), _amp_ctx:
            logits = self.lm_head(h)

        return logits, activations

    def shard_stationary_forward(
        self,
        batch_input_ids: List[torch.Tensor],
        batch_attention_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Shard-Stationary forward pass.

        Inverts the standard training loop from:
            for batch in grad_accum_batches:    # outer
                for layer in 0..63:             # inner — shard loaded grad_accum times
        to:
            for layer in 0..63:                 # outer — shard loaded ONCE
                for batch in grad_accum_batches: # inner — all batches through same shard

        Reduces RAM→GPU shard transfers from (grad_accum × num_layers) to
        (num_layers) for the entire forward phase.  The CPU weight cache is
        warmed by this pass, so the backward's Tier-2 hits are fast.

        Args:
            batch_input_ids:      list of input_ids tensors (one per grad_accum step)
            batch_attention_masks: list of attention masks or None entries

        Returns:
            logits_all:      list of (1, seq_len, vocab_size) CPU tensors, one per batch
            activations_all: activations_all[b][i] = hidden state before layer i
                             for batch b (CPU tensors — used by shard_stationary_backward)
        """
        device = next(self.embed_tokens.parameters()).device
        num_batches = len(batch_input_ids)
        num_layers = self.model_ref.num_layers
        verify = getattr(self.model_ref, "_verify_checksums", True)
        shard_size = getattr(self.model_ref, "_shard_size_gb", 0.5)

        _amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self._amp_dtype)
            if self.mixed_precision else torch.autocast("cpu", enabled=False)
        )

        # ── Embed all batches ──────────────────────────────────────────────
        h_all: List[torch.Tensor] = []
        for input_ids in batch_input_ids:
            with torch.no_grad(), _amp_ctx:
                h = self.embed_tokens(input_ids.to(device))
            h_all.append(h)

        # Position embeddings once — identical for all batches (same padded seq_len)
        position_embeddings = self._compute_position_embeddings(h_all[0], None, device)

        # activations_all[b][0] = embed output for batch b (stored on CPU)
        activations_all: List[List[torch.Tensor]] = [
            [h.detach().cpu()] for h in h_all
        ]

        # ── Sync baseline to capture h_all + pos_emb before shard budget checks ─
        # embed_tokens outputs (h_all) and position_embeddings are non-shard
        # GPU residents that weren't present when baseline_reserved_gb was
        # first measured. Update it now so they don't consume shard budget.
        self.probe.update_baseline()

        # ── Diagnostic: show GPU state before layer loop (logged once per forward) ─
        if torch.cuda.is_available():
            _a = torch.cuda.memory_allocated() / 1024**3
            _r = torch.cuda.memory_reserved() / 1024**3
            _f, _ = torch.cuda.mem_get_info()
            logger.info(
                f"[DIAG] pre-layer-loop: alloc={_a:.3f}GB reserved={_r:.3f}GB "
                f"sys_free={_f/1024**3:.3f}GB h_all_dtype={h_all[0].dtype} "
                f"h_shape={list(h_all[0].shape)} n_batches={len(h_all)}"
            )

        # ── Layer-outer loop: load each shard ONCE, run all batches through it ─
        for layer_idx in range(num_layers):
            # Re-check VRAM per layer — this GPU is shared with other tenants
            # (vLLM, RL rollout workers) whose usage fluctuates layer-to-layer.
            # A single check at function entry would let one transient spike
            # doom all 64 layers to the (non-viable) CPU fallback.
            compute_device = _resolve_compute_device(self.probe, self.scheduler, shard_size, device)

            # Async prefetch of the NEXT shard to CPU while we compute on current
            if layer_idx + 1 < num_layers:
                self.scheduler.prefetch_to_cpu(layer_idx + 1, verify)

            # 3-tier fetch: GPU cache → CPU weight cache → disk
            # After this call the shard lives in GPU cache AND CPU weight cache.
            # The CPU weight cache entry persists for the backward phase (Tier-2 hit).
            layer_tensors = self.scheduler.get_layer_tensors(layer_idx, verify_checksums=verify)
            layer = self._reconstruct_layer_module(layer_idx, layer_tensors)

            if compute_device == torch.device("cpu"):
                _strip_liger_patches(layer)
            elif self.use_fused_kernels:
                layer = apply_fused_kernels(layer)
            layer = layer.to(compute_device)

            # Run EVERY grad_accum batch through this one loaded shard
            for batch_idx in range(num_batches):
                h = h_all[batch_idx].to(compute_device)
                attn_mask = None
                if batch_attention_masks and batch_attention_masks[batch_idx] is not None:
                    attn_mask = batch_attention_masks[batch_idx].to(compute_device)

                with torch.no_grad(), _amp_ctx:
                    output = self._run_layer(layer, h, attn_mask, None, position_embeddings)
                    h_out = output[0] if isinstance(output, tuple) else output

                # Keep updated hidden state on compute_device for next layer
                h_all[batch_idx] = h_out
                # Save activation to CPU for backward pass
                activations_all[batch_idx].append(h_out.detach().cpu())

            # Move layer back to CPU before evicting GPU cache.
            # _reconstruct_layer_module returns a reference to the actual
            # hf_model decoder layer (not a copy), so del layer alone only
            # drops the local ref — the hf_model still holds the object and
            # its GPU tensors would remain, accumulating ~0.23 GB per layer.
            layer.cpu()
            # Evict from GPU cache — frees VRAM for next layer load.
            # CPU weight cache keeps the shard for the backward phase.
            self.scheduler.evict_layer_from_gpu(layer_idx)
            del layer

        # ── lm_head for all batches ────────────────────────────────────────
        lm_head_device = next(self.lm_head.parameters()).device
        logits_all: List[torch.Tensor] = []
        for h in h_all:
            with torch.no_grad(), _amp_ctx:
                h_dev = h.to(lm_head_device)
                if self.final_norm is not None:
                    h_dev = self.final_norm(h_dev)
                logits = self.lm_head(h_dev)
            logits_all.append(logits.detach().cpu())

        return logits_all, activations_all

    def reset_kv_cache(self) -> None:
        """Clear the CPU-side KV cache. Call before each new generation sequence."""
        self._kv_cache.clear()

    def forward_with_kvcache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive forward pass with CPU-offloaded KV cache. Inference only.

        On step=0 (prefill): processes the full prompt, stores KV state per layer on CPU.
        On step>0 (decode): processes a single new token, reloads KV from CPU for each layer.

        Args:
            input_ids: (batch, seq_len) on step=0; (batch, 1) on step>0
            attention_mask: optional mask
            step: decode step; 0=prefill, >0=decode

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        device = next(self.embed_tokens.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            h = self.embed_tokens(input_ids)

        num_layers = self.model_ref.num_layers
        probe_result = self.probe.get_parallel_n()
        parallel_n = probe_result["effective_n"]
        verify = getattr(self.model_ref, "_verify_checksums", True)
        shard_size = getattr(self.model_ref, "_shard_size_gb", 0.5)

        self._layer_skip_monitor.reset()
        batch_start = 0
        current_futures = None

        while batch_start < num_layers:
            batch_end = min(batch_start + parallel_n, num_layers)

            if current_futures is None:
                current_futures = self.scheduler.prefetch_sorted(
                    list(range(batch_start, batch_end)), verify_checksums=verify
                )

            next_start = batch_end
            next_end = min(next_start + parallel_n, num_layers)
            next_futures = None
            if next_start < num_layers:
                next_futures = self.scheduler.prefetch_sorted(
                    list(range(next_start, next_end)), verify_checksums=verify
                )

            # Per-batch heterogeneous placement
            compute_device = _resolve_compute_device(self.probe, self.scheduler, shard_size, device)

            batch_tensors = [f.result() for f in current_futures]
            h = h.to(compute_device)

            for local_idx, layer_tensors in enumerate(batch_tensors):
                layer_idx = batch_start + local_idx
                layer = self._reconstruct_layer_module(layer_idx, layer_tensors)

                if compute_device == torch.device("cpu"):
                    _strip_liger_patches(layer)
                elif self.use_fused_kernels:
                    layer = apply_fused_kernels(layer)
                layer = layer.to(compute_device)

                # Load KV cache from CPU for this layer (decode steps only)
                past_kv = None
                if step > 0 and layer_idx in self._kv_cache:
                    k_cpu, v_cpu = self._kv_cache[layer_idx]
                    past_kv = (k_cpu.to(device), v_cpu.to(device))

                with torch.no_grad():
                    output = self._run_layer_with_cache(layer, h, attention_mask,
                                                        past_key_value=past_kv)

                if isinstance(output, tuple):
                    h = output[0]
                    present_kv = output[1] if len(output) > 1 else None
                else:
                    h = output
                    present_kv = None

                # Offload updated KV to CPU
                if present_kv is not None:
                    try:
                        k_new, v_new = present_kv[0], present_kv[1]
                        self._kv_cache[layer_idx] = (k_new.cpu(), v_new.cpu())
                    except (IndexError, TypeError):
                        pass

                del layer

            for layer_tensors in batch_tensors:
                self.scheduler.evict(layer_tensors)

            batch_start = batch_end
            current_futures = next_futures

        lm_head_device = next(self.lm_head.parameters()).device
        h = h.to(lm_head_device)
        with torch.no_grad():
            logits = self.lm_head(h)

        return logits

    def _run_layer(
        self,
        layer: nn.Module,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """Run a single decoder layer. Handles variable output formats."""
        # Move position_embeddings to same device as h if needed
        pe = None
        if position_embeddings is not None:
            pe = tuple(t.to(h.device) for t in position_embeddings)
        # Convert attention mask to bool (HF SDPA expects bool or float, not int64)
        if attention_mask is not None:
            attn_mask = attention_mask.to(h.device)
            if attn_mask.dtype == torch.long:
                attn_mask = attn_mask.bool()
        else:
            attn_mask = None
        pos_ids = position_ids.to(h.device) if position_ids is not None else None

        try:
            output = layer(
                h,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                position_embeddings=pe,
                use_cache=False,
            )
        except TypeError:
            try:
                output = layer(
                    h,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    use_cache=False,
                )
            except TypeError:
                try:
                    output = layer(h, attention_mask=attn_mask)
                except TypeError:
                    output = layer(h)

        return output

    def _run_layer_with_cache(
        self,
        layer: nn.Module,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
    ) -> tuple:
        """Run a single decoder layer with use_cache=True for KV state capture."""
        try:
            out = layer(h, attention_mask=attention_mask,
                        past_key_value=past_key_value, use_cache=True)
            return out if isinstance(out, tuple) else (out,)
        except TypeError:
            try:
                out = layer(h, attention_mask=attention_mask, use_cache=True)
                return out if isinstance(out, tuple) else (out,)
            except TypeError:
                out = layer(h)
                return out if isinstance(out, tuple) else (out,)

    def _reconstruct_layer_module(
        self, layer_idx: int, tensors: dict
    ) -> nn.Module:
        """Reconstruct a runnable nn.Module from a flat dict of tensors.
        Uses cached _decoder_layers list to avoid repeated HF model tree traversal.
        """
        # Fast path: use cached layer list (set on ShardedModel after init)
        cached_layers = getattr(self.model_ref, "_decoder_layers", None)
        if cached_layers is not None and layer_idx < len(cached_layers):
            layer = cached_layers[layer_idx]
            missing, _ = layer.load_state_dict(tensors, strict=False)
            if missing:
                logger.debug(f"Layer {layer_idx}: {len(missing)} missing keys in shard")
            _restore_bnb_quant_state(layer, tensors)
            return layer

        # Fallback: traverse HF model tree
        model = self.model_ref._hf_model
        if model is not None:
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            _base = getattr(getattr(model, "base_model", model), "model", model)
            layers = _get_decoder_layers(_base, model_type)
            if layers is None:
                layers = _get_decoder_layers(model, model_type)
            if layers is not None and layer_idx < len(layers):
                layer = layers[layer_idx]
                layer.load_state_dict(tensors, strict=False)
                _restore_bnb_quant_state(layer, tensors)
                return layer

        raise RuntimeError(
            f"Cannot reconstruct layer {layer_idx}: no HF model reference available"
        )

    def _compute_position_embeddings(
        self,
        h: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[Tuple]:
        """
        Compute (cos, sin) rotary embeddings for models that require them
        to be passed per-layer (e.g. Qwen2, Llama3.x in newer transformers).
        Returns None if the model doesn't expose a rotary_emb module.
        """
        hf_model = self.model_ref._hf_model
        if hf_model is None:
            return None
        # Unwrap PeftModel
        base = getattr(getattr(hf_model, "base_model", hf_model), "model", hf_model)
        inner = getattr(base, "model", base)  # e.g. Qwen2Model inside Qwen2ForCausalLM
        rotary_emb = getattr(inner, "rotary_emb", None)
        if rotary_emb is None:
            return None
        seq_len = h.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        try:
            rotary_emb = rotary_emb.to(device)
            cos, sin = rotary_emb(h, position_ids)
            return (cos, sin)
        except Exception:
            return None


def _get_decoder_layers(model: nn.Module, model_type: Optional[str] = None):
    """Return decoder layers via registry lookup with trial-and-error fallback."""
    return _registry_get_decoder_layers(model, model_type)
