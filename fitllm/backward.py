from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
import torch.nn as nn

from .probe import AdaptiveShardProbe
from .scheduler import ShardScheduler
from .forward import _get_decoder_layers, _restore_bnb_quant_state

if TYPE_CHECKING:
    from .model import ShardedModel

logger = logging.getLogger(__name__)


class BackwardEngine:
    """
    Computes gradients for LoRA parameters by replaying the forward pass
    in reverse, layer by layer, using saved activations.

    Gradients are accumulated to disk in fp32 so that the optimizer can
    apply them across multiple gradient accumulation steps.
    """

    def __init__(
        self,
        model_ref: "ShardedModel",
        scheduler: ShardScheduler,
        probe: AdaptiveShardProbe,
        lm_head: nn.Module,
        loss_fn: Optional[nn.Module],
        grad_dir: Path,
        grad_accum_steps: int = 8,
        final_norm: Optional[nn.Module] = None,
    ) -> None:
        self.model_ref = model_ref
        self.scheduler = scheduler
        self.probe = probe
        self.lm_head = lm_head
        self.final_norm = final_norm
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.grad_dir = Path(grad_dir)
        self.grad_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = grad_accum_steps

        # LoRA grad accumulator — lives entirely in CPU RAM, never touches disk.
        # Structure: { layer_idx: { param_name: grad_tensor (fp32, cpu) } }
        self._lora_grad_accum: Dict[int, Dict[str, torch.Tensor]] = {}

    def backward(
        self,
        loss: torch.Tensor,
        activations: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Backpropagate gradients through all layers in reverse order.

        Args:
            loss: scalar loss tensor (already computed in forward)
            activations: list of CPU tensors saved during forward pass.
                         activations[i] = hidden state BEFORE layer i
                         activations[-1] = hidden state after last layer
            labels: optional token labels; if provided, recomputes loss
        """
        probe_result = self.probe.get_parallel_n()
        parallel_n = probe_result["effective_n"]
        num_layers = self.model_ref.num_layers
        device = self._get_device()
        verify = getattr(self.model_ref, "_verify_checksums", True)

        # We need grad_h with respect to the last hidden state
        # activations[-1] is the output of the final layer (before lm_head)
        # Move h_last to lm_head's device to avoid in-place mutation of lm_head
        lm_head_device = next(self.lm_head.parameters()).device
        h_last = activations[-1].to(lm_head_device).detach().requires_grad_(True)

        # Re-run lm_head (with final norm) to get a fresh computation graph
        h_for_head = self.final_norm(h_last) if self.final_norm is not None else h_last
        logits = self.lm_head(h_for_head)

        if labels is not None:
            b, s, v = logits.shape
            computed_loss = self.loss_fn(
                logits.view(b * s, v), labels.to(lm_head_device).view(b * s)
            )
        else:
            computed_loss = loss

        computed_loss.backward()
        grad_h = h_last.grad.detach().cpu()

        # Position embeddings are identical for every layer (sequence-position-dependent,
        # not layer-dependent) — compute once and reuse across all 64 backward layers
        pe = self._compute_position_embeddings(activations[0].to(device), 0, device)

        # Reverse layer loop — use 3-tier shard access (GPU → CPU → disk)
        for layer_idx in reversed(range(num_layers)):
            h_prev = activations[layer_idx].to(device).detach().requires_grad_(True)

            # 3-tier: GPU cache hit → CPU cache hit → NVMe read
            layer_tensors = self.scheduler.get_layer_tensors(layer_idx, verify_checksums=verify)
            layer = self._reconstruct_layer(layer_idx, layer_tensors)
            layer = layer.to(device)
            try:
                output = layer(h_prev, position_embeddings=pe, use_cache=False)
            except TypeError:
                try:
                    output = layer(h_prev, use_cache=False)
                except TypeError:
                    try:
                        output = layer(h_prev)
                    except Exception as e:
                        logger.warning(f"Layer {layer_idx} forward failed in backward: {e}")
                        layer.cpu()
                        self.scheduler.evict_layer_from_gpu(layer_idx)
                        del layer
                        continue

            if isinstance(output, tuple):
                h_out = output[0]
            else:
                h_out = output

            grad_h_device = grad_h.to(device)
            h_out.backward(grad_h_device)

            if h_prev.grad is not None:
                grad_h = h_prev.grad.detach().cpu()
            else:
                logger.warning(f"Layer {layer_idx}: h_prev.grad is None, zeroing grad_h")
                grad_h = torch.zeros_like(h_prev).cpu()

            # Accumulate LoRA grads in CPU RAM — no disk write
            self._accumulate_grads_to_ram(layer_idx, layer)

            # Move layer back to CPU — _reconstruct_layer returns a ref to the
            # actual hf_model decoder layer (not a copy), so del alone doesn't
            # free GPU VRAM (hf_model keeps the reference alive).
            layer.cpu()
            # Done with this layer's GPU copy — free VRAM for next layer
            self.scheduler.evict_layer_from_gpu(layer_idx)
            del layer

    def shard_stationary_backward(
        self,
        activations_all: List[List[torch.Tensor]],
        labels_all: List[torch.Tensor],
        loss_scale: float = 1.0,
    ) -> float:
        """
        Shard-Stationary backward pass.

        Mirrors shard_stationary_forward: each shard is loaded ONCE in reverse
        order and ALL grad_accum batches are backpropped through it before eviction.

        The CPU weight cache was warmed by shard_stationary_forward, so every
        get_layer_tensors call hits Tier-2 (CPU RAM) instead of disk.

        LoRA gradients from all batches accumulate in self._lora_grad_accum
        exactly as the per-batch backward does — the optimizer step is unchanged.

        Args:
            activations_all: from shard_stationary_forward —
                             activations_all[b][i] = hidden state before layer i
            labels_all:      label tensors, one per batch
            loss_scale:      scale applied to each batch loss before backward.
                             Pass 1/grad_accum so accumulated grads equal the mean,
                             not the sum (prevents effective LR blow-up).

        Returns:
            mean loss across all batches (float, for logging)
        """
        device = self._get_device()
        num_batches = len(activations_all)
        num_layers = self.model_ref.num_layers
        verify = getattr(self.model_ref, "_verify_checksums", True)
        lm_head_device = next(self.lm_head.parameters()).device

        # ── Seed grad_h for each batch from lm_head ───────────────────────
        grad_h_all: List[torch.Tensor] = []
        total_loss = 0.0

        for batch_idx in range(num_batches):
            h_last = (activations_all[batch_idx][-1]
                      .to(lm_head_device).detach().requires_grad_(True))
            h_for_head = self.final_norm(h_last) if self.final_norm is not None else h_last
            logits = self.lm_head(h_for_head)
            b, s, v = logits.shape
            labels = labels_all[batch_idx].to(lm_head_device)
            loss = self.loss_fn(logits.view(b * s, v), labels.view(b * s))
            total_loss += loss.item()
            (loss * loss_scale).backward()
            grad_h_all.append(h_last.grad.detach().cpu())

        # Position embeddings — computed once, identical for all layers × batches
        pe = self._compute_position_embeddings(
            activations_all[0][0].to(device), 0, device
        )

        # ── Layer-outer loop (reverse): load each shard ONCE, backprop all batches ─
        for layer_idx in reversed(range(num_layers)):
            # Async prefetch of the PREVIOUS shard to CPU while we compute
            if layer_idx - 1 >= 0:
                self.scheduler.prefetch_to_cpu(layer_idx - 1, verify)

            # Tier-2 hit: forward phase already placed this shard in CPU weight cache
            layer_tensors = self.scheduler.get_layer_tensors(layer_idx, verify_checksums=verify)
            layer = self._reconstruct_layer(layer_idx, layer_tensors)
            layer = layer.to(device)

            # Backprop EVERY batch through this one loaded shard
            for batch_idx in range(num_batches):
                # Zero layer grads before each batch — prevents cross-batch grad
                # accumulation on param.grad (we accumulate ourselves in _lora_grad_accum)
                layer.zero_grad()

                h_prev = (activations_all[batch_idx][layer_idx]
                          .to(device).detach().requires_grad_(True))

                try:
                    output = layer(h_prev, position_embeddings=pe, use_cache=False)
                except TypeError:
                    try:
                        output = layer(h_prev, use_cache=False)
                    except TypeError:
                        output = layer(h_prev)

                h_out = output[0] if isinstance(output, tuple) else output
                h_out.backward(grad_h_all[batch_idx].to(device))

                if h_prev.grad is not None:
                    grad_h_all[batch_idx] = h_prev.grad.detach().cpu()
                else:
                    grad_h_all[batch_idx] = torch.zeros_like(h_prev).cpu()

                # Accumulate this batch's LoRA grads into the per-layer RAM bucket
                self._accumulate_grads_to_ram(layer_idx, layer)

            # Move layer back to CPU before evicting (same reason as in forward:
            # _reconstruct_layer returns a hf_model ref, del alone doesn't free GPU).
            layer.cpu()
            # Free GPU memory — CPU weight cache keeps shard until clear_weight_cache()
            self.scheduler.evict_layer_from_gpu(layer_idx)
            del layer

        return total_loss / max(num_batches, 1)

    def _reconstruct_layer(self, layer_idx: int, tensors: dict) -> nn.Module:
        """Reconstruct a layer from shard tensors.
        Uses cached _decoder_layers list to avoid repeated HF model tree traversal.
        """
        # Fast path: cached layer list
        cached_layers = getattr(self.model_ref, "_decoder_layers", None)
        if cached_layers is not None and layer_idx < len(cached_layers):
            layer = cached_layers[layer_idx]
            layer.load_state_dict(tensors, strict=False)
            _restore_bnb_quant_state(layer, tensors)
            return layer

        # Fallback: traverse HF model tree
        model = self.model_ref._hf_model
        if model is None:
            raise RuntimeError("No HF model reference for layer reconstruction")
        _base = getattr(getattr(model, "base_model", model), "model", model)
        layers = _get_decoder_layers(_base)
        if layers is None:
            layers = _get_decoder_layers(model)
        if layers is None or layer_idx >= len(layers):
            raise RuntimeError(f"Cannot find layer {layer_idx}")
        layer = layers[layer_idx]
        layer.load_state_dict(tensors, strict=False)
        _restore_bnb_quant_state(layer, tensors)
        return layer

    def _accumulate_grads_to_ram(self, layer_idx: int, layer: nn.Module) -> None:
        """
        Collect LoRA gradients from this layer and accumulate them in CPU RAM.
        No disk I/O — grads live in self._lora_grad_accum until optimizer step.
        """
        bucket = self._lora_grad_accum.setdefault(layer_idx, {})
        for name, param in layer.named_parameters():
            if ("lora_A" in name or "lora_B" in name) and param.grad is not None:
                g = param.grad.detach().float().cpu()
                if name in bucket:
                    bucket[name].add_(g)
                else:
                    bucket[name] = g

    def get_and_zero_grads(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Return the accumulated LoRA grad dict and reset it.
        Called by the optimizer at the end of each grad-accum cycle.
        """
        grads = self._lora_grad_accum
        self._lora_grad_accum = {}
        return grads

    def zero_all_grads(self) -> None:
        """Reset all accumulated gradients."""
        self._lora_grad_accum.clear()

    def _compute_position_embeddings(self, h: torch.Tensor, layer_idx: int, device: torch.device):
        """Compute (cos, sin) rotary embeddings for backward layer re-run."""
        hf_model = self.model_ref._hf_model
        if hf_model is None:
            return None
        base = getattr(getattr(hf_model, "base_model", hf_model), "model", hf_model)
        inner = getattr(base, "model", base)
        rotary_emb = getattr(inner, "rotary_emb", None)
        if rotary_emb is None:
            return None
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        try:
            rotary_emb = rotary_emb.to(device)
            cos, sin = rotary_emb(h, position_ids)
            return (cos, sin)
        except Exception:
            return None

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
