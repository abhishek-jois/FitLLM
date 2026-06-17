from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn

# Maps HuggingFace config.model_type → (layers_attr, embed_attr, head_attr)
# All paths are dot-separated from the model root.
ARCHITECTURE_REGISTRY: dict[str, Tuple[str, str, str]] = {
    # Dense decoder-only — standard LLaMA-style paths
    "llama":        ("model.layers", "model.embed_tokens", "lm_head"),
    "llama4":       ("model.layers", "model.embed_tokens", "lm_head"),
    "qwen2":        ("model.layers", "model.embed_tokens", "lm_head"),
    "qwen3":        ("model.layers", "model.embed_tokens", "lm_head"),
    "mistral":      ("model.layers", "model.embed_tokens", "lm_head"),
    "mistral3":     ("model.layers", "model.embed_tokens", "lm_head"),
    "gemma":        ("model.layers", "model.embed_tokens", "lm_head"),
    "gemma2":       ("model.layers", "model.embed_tokens", "lm_head"),
    "gemma3":       ("model.layers", "model.embed_tokens", "lm_head"),
    "phi3":         ("model.layers", "model.embed_tokens", "lm_head"),
    "phi4":         ("model.layers", "model.embed_tokens", "lm_head"),
    "yi":           ("model.layers", "model.embed_tokens", "lm_head"),
    "granite":      ("model.layers", "model.embed_tokens", "lm_head"),
    "stablelm":     ("model.layers", "model.embed_tokens", "lm_head"),
    "stablelm_epoch": ("model.layers", "model.embed_tokens", "lm_head"),
    "cohere":       ("model.layers", "model.embed_tokens", "lm_head"),
    "cohere2":      ("model.layers", "model.embed_tokens", "lm_head"),
    "internlm2":    ("model.layers", "model.embed_tokens", "lm_head"),
    "baichuan":     ("model.layers", "model.embed_tokens", "lm_head"),
    "starcoder2":   ("model.layers", "model.embed_tokens", "lm_head"),
    "olmo":         ("model.layers", "model.embed_tokens", "lm_head"),
    "olmo2":        ("model.layers", "model.embed_tokens", "lm_head"),
    "nemotron":     ("model.layers", "model.embed_tokens", "lm_head"),
    # Dense decoder-only — variant paths
    "phi":          ("model.layers", "model.embed_tokens", "lm_head"),
    "falcon":       ("transformer.h", "transformer.word_embeddings", "lm_head"),
    "gpt2":         ("transformer.h", "transformer.wte", "lm_head"),
    "bloom":        ("transformer.h", "transformer.word_embeddings", "lm_head"),
    "gpt_neox":     ("gpt_neox.layers", "gpt_neox.embed_in", "embed_out"),
    "gpt_j":        ("transformer.h", "transformer.wte", "lm_head"),
    "opt":          ("model.decoder.layers", "model.decoder.embed_tokens", "lm_head"),
    "chatglm":      ("transformer.encoder.layers", "transformer.embedding", "output_layer"),
    # MoE families
    "mixtral":      ("model.layers", "model.embed_tokens", "lm_head"),
    "deepseek_v2":  ("model.layers", "model.embed_tokens", "lm_head"),
    "deepseek_v3":  ("model.layers", "model.embed_tokens", "lm_head"),
    "qwen3_moe":    ("model.layers", "model.embed_tokens", "lm_head"),
    "granite_moe":  ("model.layers", "model.embed_tokens", "lm_head"),
}


def _resolve_attr(obj: object, dotted_path: str) -> Optional[object]:
    """Traverse a dot-separated attribute path. Returns None if any step is missing."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def get_decoder_layers(
    model: nn.Module, model_type: Optional[str] = None
) -> Optional[nn.ModuleList]:
    """Return the nn.ModuleList of decoder layers via registry lookup with fallback."""
    if model_type and model_type in ARCHITECTURE_REGISTRY:
        layers_attr, _, _ = ARCHITECTURE_REGISTRY[model_type]
        layers = _resolve_attr(model, layers_attr)
        if layers is not None:
            return layers  # type: ignore[return-value]

    # Fallback: trial-and-error for unknown model types
    for attr in ("layers", "decoder", "transformer"):
        obj = getattr(model, attr, None)
        if obj is not None:
            for sub in ("layers", "h", "block"):
                sub_obj = getattr(obj, sub, None)
                if sub_obj is not None:
                    return sub_obj  # type: ignore[return-value]
            if isinstance(obj, nn.ModuleList):
                return obj
    inner = getattr(model, "model", None)
    if inner is not None:
        for attr in ("layers", "h", "block"):
            obj = getattr(inner, attr, None)
            if obj is not None and isinstance(obj, nn.ModuleList):
                return obj
    return None


def get_embed_and_head(
    model: nn.Module, model_type: Optional[str] = None
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """Return (embed_tokens, lm_head) via registry lookup. Returns (None, None) on miss."""
    if model_type and model_type in ARCHITECTURE_REGISTRY:
        _, embed_attr, head_attr = ARCHITECTURE_REGISTRY[model_type]
        embed = _resolve_attr(model, embed_attr)
        head = _resolve_attr(model, head_attr)
        if embed is not None and head is not None:
            return embed, head  # type: ignore[return-value]
    return None, None
