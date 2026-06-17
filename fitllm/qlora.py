from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QLoRAManager:
    """
    Manages the full QLoRA pipeline (Dettmers et al., 2023):

      1. prepare_model_for_kbit_training() — stabilizes 4-bit model for training
         by casting LayerNorm / lm_head to fp32 and enabling gradient checkpointing.
      2. get_peft_model(LoraConfig) — attaches trainable LoRA A/B matrices on top
         of the frozen NF4-quantized base weights via PEFT.

    The resulting peft_model has only the A/B adapter matrices as trainable params
    (~0.19% of 70B), everything else stays frozen in 4-bit on disk.
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: float = 32.0,
        targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
        dropout: float = 0.05,
        bias: str = "none",
    ) -> None:
        self.rank = rank
        self.alpha = alpha
        self.targets = list(targets)
        self.dropout = dropout
        self.bias = bias
        self._peft_model = None

    def apply(self, hf_model: nn.Module, is_quantized: bool = True) -> nn.Module:
        """
        Apply the full QLoRA pipeline to hf_model.

        Args:
            hf_model:     A loaded HuggingFace CausalLM model.
            is_quantized: True if model was loaded with BitsAndBytesConfig (4-bit/8-bit).
                          Enables prepare_model_for_kbit_training().

        Returns:
            PeftModel wrapping hf_model with trainable LoRA A/B matrices.
        """
        try:
            from peft import (
                LoraConfig,
                TaskType,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
        except ImportError:
            raise ImportError(
                "peft is required for QLoRA. Install it with: pip install peft"
            )

        # Step 1 — stabilize quantized model for training
        if is_quantized:
            hf_model = prepare_model_for_kbit_training(
                hf_model,
                use_gradient_checkpointing=True,
            )
            logger.info("prepare_model_for_kbit_training applied (norms cast to fp32)")

        # Step 2 — attach LoRA A/B adapters via PEFT
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.targets,
            lora_dropout=self.dropout,
            bias=self.bias,
            inference_mode=False,
        )
        self._peft_model = get_peft_model(hf_model, lora_config)
        self._peft_model.print_trainable_parameters()
        logger.info(
            f"QLoRA adapters attached — rank={self.rank}, alpha={self.alpha}, "
            f"targets={self.targets}"
        )
        return self._peft_model

    @property
    def peft_model(self):
        return self._peft_model

    def trainable_parameters(self):
        """Yield (name, param) for all trainable LoRA parameters."""
        if self._peft_model is None:
            return
        for name, param in self._peft_model.named_parameters():
            if param.requires_grad:
                yield name, param

    def trainable_param_count(self) -> int:
        return sum(p.numel() for _, p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """
        Save LoRA adapter weights using PEFT's save_pretrained.
        Saves adapter_config.json + adapter_model.safetensors to `path/`.
        Compatible with transformers, vllm, ollama.
        """
        if self._peft_model is None:
            logger.warning("QLoRAManager: no peft_model to save")
            return
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._peft_model.save_pretrained(str(out))
        logger.info(f"QLoRA adapters saved to {out}")

    def load(self, path: str) -> None:
        """
        Load LoRA adapter weights back into the attached peft_model.
        Expects the directory written by save().
        """
        if self._peft_model is None:
            raise RuntimeError(
                "QLoRAManager: call apply() before load() to attach the peft_model first"
            )
        from peft import set_peft_model_state_dict

        adapter_path = Path(path)
        st_path = adapter_path / "adapter_model.safetensors"
        bin_path = adapter_path / "adapter_model.bin"

        if st_path.exists():
            from safetensors.torch import load_file
            sd = load_file(str(st_path))
        elif bin_path.exists():
            sd = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No adapter_model.safetensors or adapter_model.bin found in {path}"
            )

        set_peft_model_state_dict(self._peft_model, sd)
        logger.info(f"QLoRA adapters loaded from {path}")

    def zero_grad(self) -> None:
        """Zero gradients on all trainable LoRA parameters."""
        for _, param in self.trainable_parameters():
            if param.grad is not None:
                param.grad.zero_()

    def __repr__(self) -> str:
        n = self.trainable_param_count()
        return (
            f"QLoRAManager(rank={self.rank}, alpha={self.alpha}, "
            f"targets={self.targets}, trainable_params={n:,})"
        )
