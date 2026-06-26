from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from .scheduler import load_shard_with_checksum, save_shard_with_checksum

logger = logging.getLogger(__name__)


class ShardOptimizer:
    """
    AdamW optimizer that operates on disk-backed shards.

    For full (non-LoRA) mode:
      - Master weights stored as fp32 .pt files per layer
      - Optimizer state (m, v, step) stored as safetensors per layer
      - Gradients loaded from accumulated grad files
      - After update, shard is re-quantized to fp16 safetensors

    For LoRA mode:
      - Only A and B parameters are updated
      - Optimizer state kept in CPU RAM dict (much faster)
    """

    def __init__(
        self,
        shard_dir: Path,
        num_layers: int,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        peft_model=None,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.peft_model = peft_model  # PeftModel or None
        self.global_step: int = 0

        # LoRA mode: keep optimizer states in CPU RAM (no disk I/O needed)
        self._lora_opt_state: Dict[str, Dict] = {}
        # Reverse lookup: local grad param name → full peft_model param name
        # Built lazily on first step_from_memory call to avoid init-time cost
        self._grad_to_full_name: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Full shard update (non-LoRA)
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Run one AdamW update step for all layers.
        Operates on disk; loads master weights, grad file, opt state,
        runs update, saves everything back.
        """
        self.global_step += 1

        if self.peft_model is not None:
            self._lora_step()
            return

        for layer_idx in range(self.num_layers):
            grad_path = self._grad_path(layer_idx)
            if not grad_path.exists():
                continue

            master_path = self._master_path(layer_idx)
            optstate_path = self._optstate_path(layer_idx)
            shard_path = self._shard_path(layer_idx)

            # Load gradients
            grads = load_shard_with_checksum(grad_path, verify=False)

            # Load master fp32 weights (or fall back to fp16 shard)
            if master_path.exists():
                master = torch.load(master_path, map_location="cpu")
            elif shard_path.exists():
                master = {
                    k: v.float()
                    for k, v in load_shard_with_checksum(shard_path, verify=False).items()
                }
            else:
                logger.warning(f"No master weights found for layer {layer_idx}, skipping")
                continue

            # Load or initialize optimizer state
            if optstate_path.exists():
                optstate = load_shard_with_checksum(optstate_path, verify=False)
                m = {k: optstate[f"m_{k}"] for k in master if f"m_{k}" in optstate}
                v = {k: optstate[f"v_{k}"] for k in master if f"v_{k}" in optstate}
                step_tensor = optstate.get("_step", torch.tensor(float(self.global_step - 1)))
                t = int(step_tensor.item()) + 1
            else:
                m = {k: torch.zeros_like(p) for k, p in master.items()}
                v = {k: torch.zeros_like(p) for k, p in master.items()}
                t = self.global_step

            updated = {}
            for k, p in master.items():
                if k not in grads:
                    updated[k] = p
                    continue

                g = grads[k].float()
                m_k = self.beta1 * m.get(k, torch.zeros_like(p)) + (1 - self.beta1) * g
                v_k = self.beta2 * v.get(k, torch.zeros_like(p)) + (1 - self.beta2) * g * g
                m[k] = m_k
                v[k] = v_k

                # Bias correction
                m_hat = m_k / (1 - self.beta1 ** t)
                v_hat = v_k / (1 - self.beta2 ** t)

                # AdamW update
                p_new = p - self.lr * (
                    m_hat / (v_hat.sqrt() + self.eps)
                    + self.weight_decay * p
                )
                updated[k] = p_new

            # Save master fp32
            torch.save(updated, master_path)

            # Save optimizer state
            flat_optstate: Dict[str, torch.Tensor] = {}
            for k in updated:
                flat_optstate[f"m_{k}"] = m.get(k, torch.zeros_like(updated[k]))
                flat_optstate[f"v_{k}"] = v.get(k, torch.zeros_like(updated[k]))
            flat_optstate["_step"] = torch.tensor(float(t))
            save_shard_with_checksum(flat_optstate, optstate_path)

            # Re-quantize to fp16 and save as shard
            fp16_dict = {k: v_param.half() for k, v_param in updated.items()}
            save_shard_with_checksum(fp16_dict, shard_path)

            # Delete grad file
            grad_path.unlink(missing_ok=True)
            checksum = grad_path.with_suffix(".sha256")
            if checksum.exists():
                checksum.unlink()

    # ------------------------------------------------------------------
    # LoRA mode: update only A and B, keep state in RAM
    # ------------------------------------------------------------------

    def set_lr(self, lr: float) -> None:
        """Update learning rate (called by LR scheduler each step)."""
        self.lr = lr

    def step_from_memory(
        self, grad_accum: Dict[int, Dict[str, torch.Tensor]]
    ) -> None:
        """
        CPU-only AdamW update using in-memory LoRA grad accumulator.

        grad_accum: { layer_idx: { param_name: grad_tensor (fp32, cpu) } }

        All computation happens on CPU — no VRAM needed.
        LoRA A/B weights are read from and written back to the peft_model in-place.
        Optimizer states (m, v) stay in self._lora_opt_state in CPU RAM permanently.
        """
        self.global_step += 1

        if self.peft_model is None:
            return

        # Build a name→param map for fast lookup
        param_map = {n: p for n, p in self.peft_model.named_parameters() if p.requires_grad}

        for layer_idx, layer_grads in grad_accum.items():
            for param_name, g in layer_grads.items():
                # Use (layer_idx, param_name) as cache key.
                # Without layer_idx, all 64 layers' identically-named params (e.g.
                # "self_attn.q_proj.lora_A.default.weight") would match the SAME
                # full peft_model name (layer 0), leaving layers 1-63 never updated.
                cache_key = f"{layer_idx}:{param_name}"
                matched_name = self._grad_to_full_name.get(cache_key)

                if matched_name is None:
                    # Search for the full param name that belongs to this specific layer
                    layer_marker = f"layers.{layer_idx}."
                    for full_name in param_map:
                        if layer_marker in full_name and full_name.endswith(param_name):
                            matched_name = full_name
                            self._grad_to_full_name[cache_key] = full_name
                            break

                if matched_name is None:
                    logger.debug(f"No param match for layer {layer_idx} / {param_name}")
                    continue

                param = param_map[matched_name]
                g = g.float().cpu()

                if matched_name not in self._lora_opt_state:
                    self._lora_opt_state[matched_name] = {
                        "m": torch.zeros_like(g),
                        "v": torch.zeros_like(g),
                        "t": 0,
                    }

                state = self._lora_opt_state[matched_name]
                state["t"] += 1
                t_local = state["t"]

                m_k = self.beta1 * state["m"] + (1 - self.beta1) * g
                v_k = self.beta2 * state["v"] + (1 - self.beta2) * g * g
                state["m"] = m_k
                state["v"] = v_k

                m_hat = m_k / (1 - self.beta1 ** t_local)
                v_hat = v_k / (1 - self.beta2 ** t_local)

                # Update on CPU, then copy back to wherever param lives
                with torch.no_grad():
                    delta = self.lr * (
                        m_hat / (v_hat.sqrt() + self.eps)
                        + self.weight_decay * param.data.float().cpu()
                    )
                    param.data -= delta.to(param.device)

    def _lora_step(self) -> None:
        """AdamW update for LoRA A and B parameters — reads from param.grad."""
        t = self.global_step

        for param_name, param in self._named_lora_params():
            if param.grad is None:
                continue

            g = param.grad.detach().float()

            if param_name not in self._lora_opt_state:
                self._lora_opt_state[param_name] = {
                    "m": torch.zeros_like(g),
                    "v": torch.zeros_like(g),
                    "t": 0,
                }

            state = self._lora_opt_state[param_name]
            state["t"] += 1
            t_local = state["t"]

            m_k = self.beta1 * state["m"] + (1 - self.beta1) * g
            v_k = self.beta2 * state["v"] + (1 - self.beta2) * g * g
            state["m"] = m_k
            state["v"] = v_k

            m_hat = m_k / (1 - self.beta1 ** t_local)
            v_hat = v_k / (1 - self.beta2 ** t_local)

            with torch.no_grad():
                param.data -= self.lr * (
                    m_hat.to(param.device) / (v_hat.to(param.device).sqrt() + self.eps)
                    + self.weight_decay * param.data
                )

    def _named_lora_params(self):
        """Yield (name, param) for all trainable PEFT LoRA parameters."""
        if self.peft_model is None:
            return
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                yield name, param

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict_cpu(self) -> Dict:
        """Serialize optimizer state for checkpointing."""
        return {
            "global_step": self.global_step,
            "lora_opt_state": {
                k: {ik: (iv.cpu() if isinstance(iv, torch.Tensor) else iv) for ik, iv in v.items()}
                for k, v in self._lora_opt_state.items()
            },
            "lr": self.lr,
        }

    def load_state_dict_cpu(self, sd: Dict) -> None:
        """Restore optimizer state from a checkpoint dict."""
        self.global_step = sd.get("global_step", 0)
        self.lr = sd.get("lr", self.lr)
        for k, v in sd.get("lora_opt_state", {}).items():
            self._lora_opt_state[k] = {
                ik: iv for ik, iv in v.items()
            }
            # 't' was missing in checkpoints saved before this fix — default to global_step
            if "t" not in self._lora_opt_state[k]:
                self._lora_opt_state[k]["t"] = self.global_step

    def zero_grad(self) -> None:
        """Zero gradients: delete disk grad files and zero in-memory grads."""
        for layer_idx in range(self.num_layers):
            grad_path = self._grad_path(layer_idx)
            if grad_path.exists():
                grad_path.unlink(missing_ok=True)
                chk = grad_path.with_suffix(".sha256")
                if chk.exists():
                    chk.unlink(missing_ok=True)

        if self.peft_model is not None:
            for _, param in self._named_lora_params():
                if param.grad is not None:
                    param.grad.zero_()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _grad_path(self, layer_idx: int) -> Path:
        return self.shard_dir / f"layer_{layer_idx:03d}_grads.safetensors"

    def _master_path(self, layer_idx: int) -> Path:
        return self.shard_dir / f"layer_{layer_idx:03d}_master_fp32.pt"

    def _shard_path(self, layer_idx: int) -> Path:
        return self.shard_dir / f"layer_{layer_idx:03d}_weights.safetensors"

    def _optstate_path(self, layer_idx: int) -> Path:
        return self.shard_dir / f"layer_{layer_idx:03d}_optstate.safetensors"
