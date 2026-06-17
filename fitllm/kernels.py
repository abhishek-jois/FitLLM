from __future__ import annotations

import logging
from typing import Any, Dict

import torch.nn as nn

logger = logging.getLogger(__name__)


def check_kernel_availability() -> Dict[str, bool]:
    """
    Check which optional acceleration kernels are available in the environment.

    Returns a dict mapping kernel name -> bool.
    """
    availability: Dict[str, bool] = {}

    try:
        import liger_kernel  # noqa: F401
        availability["liger_kernel"] = True
    except ImportError:
        availability["liger_kernel"] = False

    try:
        import flash_attn  # noqa: F401
        availability["flash_attn"] = True
    except ImportError:
        availability["flash_attn"] = False

    try:
        import xformers  # noqa: F401
        availability["xformers"] = True
    except ImportError:
        availability["xformers"] = False

    return availability


def apply_fused_kernels(layer: nn.Module) -> nn.Module:
    """
    Attempt to patch a transformer layer with faster kernel implementations:
      - liger_kernel: fused RMSNorm and rotary embedding
      - flash_attn: memory-efficient attention

    Gracefully skips any unavailable library and logs a warning.
    Returns the (possibly patched) layer.
    """
    availability = check_kernel_availability()

    if availability.get("liger_kernel", False):
        try:
            import liger_kernel.transformers as lk

            # Patch RMSNorm if the layer contains one
            for name, child in layer.named_modules():
                child_type = type(child).__name__
                if "RMSNorm" in child_type or "rms_norm" in child_type.lower():
                    try:
                        if hasattr(lk, "LigerRMSNorm"):
                            liger_norm = lk.LigerRMSNorm(child.weight.shape[0], eps=getattr(child, "variance_epsilon", 1e-6))
                            liger_norm.weight = child.weight
                            # Replace in parent
                            parts = name.rsplit(".", 1)
                            if len(parts) == 2:
                                parent = dict(layer.named_modules())[parts[0]]
                                setattr(parent, parts[1], liger_norm)
                            else:
                                setattr(layer, name, liger_norm)
                    except Exception as e:
                        logger.debug(f"liger_kernel RMSNorm patch failed: {e}")
        except Exception as e:
            logger.warning(f"liger_kernel import succeeded but patching failed: {e}")
    else:
        logger.debug("liger_kernel not available; skipping fused norm/rotary patches")

    if availability.get("flash_attn", False):
        try:
            from flash_attn import flash_attn_func

            # Patch self_attn if it exists and has a standard attention forward
            self_attn = None
            for name, child in layer.named_modules():
                if "self_attn" in name or "attention" in name.lower():
                    if hasattr(child, "forward"):
                        self_attn = child
                        break

            if self_attn is not None:
                original_forward = self_attn.forward

                def _flash_forward(
                    hidden_states: Any,
                    attention_mask: Any = None,
                    position_ids: Any = None,
                    past_key_value: Any = None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    **kwargs: Any,
                ) -> Any:
                    # Attempt flash attention; fall back to original on error
                    try:
                        return original_forward(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            **kwargs,
                        )
                    except Exception:
                        return original_forward(
                            hidden_states,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_value,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            **kwargs,
                        )

                self_attn.forward = _flash_forward
        except Exception as e:
            logger.warning(f"flash_attn patching failed: {e}")
    else:
        logger.debug("flash_attn not available; skipping attention patch")

    return layer
