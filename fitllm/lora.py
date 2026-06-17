from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds low-rank adaptation matrices A and B.

    forward(x) = base(x) + scale * (x @ A @ B)
    where scale = alpha / rank
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # A: (in_features, rank), B: (rank, out_features)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

        # Freeze base weights
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class LoRAManager:
    """
    Manages injection and persistence of LoRA adapters into a transformer model.
    """

    def __init__(self) -> None:
        self._lora_layers: Dict[str, LoRALinear] = {}

    def inject_lora(
        self,
        model: nn.Module,
        rank: int,
        alpha: float,
        targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> nn.Module:
        """
        Walk all named modules in model. Replace any nn.Linear whose name ends
        with one of the target suffixes with a LoRALinear wrapper.
        """
        self._lora_layers.clear()

        replacements: List[Tuple[nn.Module, str, LoRALinear]] = []

        for module_name, module in model.named_modules():
            for child_name, child in module.named_children():
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                if isinstance(child, nn.Linear):
                    if any(child_name == t or full_name.endswith(t) for t in targets):
                        lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                        replacements.append((module, child_name, lora_layer))
                        self._lora_layers[full_name] = lora_layer
                        logger.debug(f"Injected LoRA into {full_name}")

        for parent_module, child_name, lora_layer in replacements:
            setattr(parent_module, child_name, lora_layer)

        logger.info(f"Injected LoRA into {len(self._lora_layers)} linear layers")
        return model

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only the LoRA A and B parameters."""
        sd: Dict[str, torch.Tensor] = {}
        for layer_name, lora_layer in self._lora_layers.items():
            sd[f"{layer_name}.lora_A"] = lora_layer.lora_A.data.clone()
            sd[f"{layer_name}.lora_B"] = lora_layer.lora_B.data.clone()
        return sd

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        """Restore A and B matrices from a state dict."""
        for layer_name, lora_layer in self._lora_layers.items():
            key_a = f"{layer_name}.lora_A"
            key_b = f"{layer_name}.lora_B"
            if key_a in sd:
                lora_layer.lora_A.data.copy_(sd[key_a])
            if key_b in sd:
                lora_layer.lora_B.data.copy_(sd[key_b])

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only LoRA A and B parameters."""
        for lora_layer in self._lora_layers.values():
            yield lora_layer.lora_A
            yield lora_layer.lora_B

    def freeze_base_model(self, model: nn.Module) -> None:
        """
        Set requires_grad=False on all parameters that are NOT part of a
        LoRALinear adapter (i.e., the frozen base weights).
        """
        lora_param_ids = {
            id(p)
            for lora_layer in self._lora_layers.values()
            for p in [lora_layer.lora_A, lora_layer.lora_B]
        }
        for param in model.parameters():
            if id(param) not in lora_param_ids:
                param.requires_grad_(False)
        logger.info("Froze all non-LoRA parameters")

    @property
    def num_lora_layers(self) -> int:
        return len(self._lora_layers)
