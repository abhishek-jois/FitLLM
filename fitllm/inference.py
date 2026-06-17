from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import InferenceConfig
from .probe import AdaptiveShardProbe
from .scheduler import ShardScheduler

if TYPE_CHECKING:
    from .model import ShardedModel

logger = logging.getLogger(__name__)


def validate_draft_tokenizer(verifier_tokenizer, draft_model_name: str) -> None:
    """
    Validate that the draft model's tokenizer is compatible with the verifier.

    Speculative decoding requires both models to share the same vocabulary and
    tokenizer. Mismatched tokenizers produce wrong acceptance probabilities and
    garbled output.

    Raises:
        ValueError: if the tokenizers are incompatible.
    """
    logger.info(f"Validating tokenizer compatibility for draft model: {draft_model_name}")
    try:
        draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
    except Exception as e:
        raise ValueError(
            f"Could not load tokenizer for draft model '{draft_model_name}': {e}"
        )

    verifier_vocab_size = verifier_tokenizer.vocab_size
    draft_vocab_size = draft_tokenizer.vocab_size

    if verifier_vocab_size != draft_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary size mismatch:\n"
            f"  Verifier : {verifier_vocab_size} tokens\n"
            f"  Draft    : {draft_vocab_size} tokens ({draft_model_name})\n\n"
            f"Speculative decoding requires both models to share the same tokenizer.\n"
            f"Use a draft model from the same family as your verifier, for example:\n"
            f"  Llama 3.x verifier  → meta-llama/Llama-3.2-1B-Instruct\n"
            f"  Qwen2.5 verifier    → Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen2.5-1.5B-Instruct\n"
            f"  Gemma 2 verifier    → google/gemma-2-2b-it"
        )

    # Spot-check a few token ids to catch same-size-but-different tokenizers
    test_strings = ["hello", "world", " the", "\n"]
    mismatches = []
    for s in test_strings:
        v_ids = verifier_tokenizer.encode(s, add_special_tokens=False)
        d_ids = draft_tokenizer.encode(s, add_special_tokens=False)
        if v_ids != d_ids:
            mismatches.append(f"  '{s}': verifier={v_ids} draft={d_ids}")

    if mismatches:
        raise ValueError(
            f"Tokenizer token-id mismatch for draft model '{draft_model_name}'.\n"
            f"These test strings encode differently:\n"
            + "\n".join(mismatches) +
            f"\n\nEven with matching vocab sizes, the models must use identical tokenizers.\n"
            f"Use a draft model from the same family as your verifier."
        )

    logger.info(
        f"Tokenizer check passed — vocab_size={verifier_vocab_size}, "
        f"spot-check OK for {len(test_strings)} test strings."
    )


def merge_lora_into_shards(model_ref, scheduler) -> None:
    """
    Phase 0 (one-time before inference): merge LoRA A/B weights into frozen W shards.

    For each layer:
      W_Q_merged = W_Q + scale * (B_Q @ A_Q)
      W_V_merged = W_V + scale * (B_V @ A_V)

    Overwrites shard files in-place. After this, LoRA is baked in — no separate
    adapter needed at inference time, and A/B matrices can be freed.
    """
    from .scheduler import load_shard_with_checksum, save_shard_with_checksum
    from peft import PeftModel
    import math

    hf_model = model_ref._hf_model
    if hf_model is None or not isinstance(hf_model, PeftModel):
        logger.info("No LoRA adapters found — skipping merge")
        return

    base = getattr(getattr(hf_model, "base_model", hf_model), "model", hf_model)
    from .forward import _get_decoder_layers
    layers = _get_decoder_layers(base)
    if layers is None:
        logger.warning("merge_lora_into_shards: could not find decoder layers")
        return

    lora_config = hf_model.peft_config.get("default", None)
    rank = getattr(lora_config, "r", 16)
    alpha = getattr(lora_config, "lora_alpha", 32.0)
    scale = alpha / rank

    logger.info(f"Merging LoRA into {len(layers)} shards (scale={scale:.3f}) ...")
    for layer_idx, layer in enumerate(layers):
        shard_path = scheduler.shard_path(layer_idx)
        if not shard_path.exists():
            continue

        tensors = load_shard_with_checksum(shard_path, verify=False)
        modified = False

        for name, param in layer.named_parameters():
            # Find matching lora_A and lora_B for this param
            base_name = name.replace(".base_layer", "").replace(".default", "")
            lora_A_key = None
            lora_B_key = None
            for k in tensors:
                if "lora_A" in k and base_name.split(".")[-2] in k:
                    lora_A_key = k
                if "lora_B" in k and base_name.split(".")[-2] in k:
                    lora_B_key = k

            if lora_A_key and lora_B_key and base_name in tensors:
                A = tensors[lora_A_key].float()
                B = tensors[lora_B_key].float()
                W = tensors[base_name].float()
                tensors[base_name] = (W + scale * (B @ A)).to(param.dtype)
                del tensors[lora_A_key]
                del tensors[lora_B_key]
                modified = True

        if modified:
            save_shard_with_checksum(tensors, shard_path)
            logger.debug(f"Merged LoRA into layer {layer_idx}")

    logger.info("LoRA merge complete")


class EWMAAcceptRate:
    """Exponentially weighted moving average of draft token acceptance rate."""

    def __init__(self, alpha: float = 0.1, init: float = 0.7) -> None:
        self.alpha = alpha
        self.value = init

    def update(self, accepted: int, proposed: int) -> float:
        if proposed == 0:
            return self.value
        rate = accepted / proposed
        self.value = self.alpha * rate + (1 - self.alpha) * self.value
        return self.value


class DynamicKController:
    """
    Dynamically adjusts the speculative decoding draft length K based on
    EWMA token acceptance rate. Matches the algorithm:

      rate < 0.10 → skip draft entirely (use big model autoregressive)
      rate < 0.30 → shrink k by 1
      rate > 0.95 → grow k by 2 (fast ramp)
      rate > 0.80 → grow k by 1
      skip_draft mode: retry draft when rate recovers above 0.30
    """

    def __init__(
        self,
        k_init: int = 4,
        k_min: int = 1,
        k_max: int = 8,
        ewma_alpha: float = 0.1,
    ) -> None:
        self._k = k_init
        self.k_min = k_min
        self.k_max = k_max
        self._ewma = EWMAAcceptRate(alpha=ewma_alpha, init=0.7)
        self.skip_draft = False

    def record_and_update(self, proposed: int, accepted: int) -> None:
        rate = self._ewma.update(accepted, proposed)

        if rate < 0.10:
            self.skip_draft = True
            logger.debug(f"DynamicK: rate={rate:.2f} → skip draft")
        elif self.skip_draft and rate > 0.30:
            self.skip_draft = False
            self._k = 1  # restart conservatively
            logger.debug(f"DynamicK: rate={rate:.2f} → resume draft at k=1")
        elif not self.skip_draft:
            if rate < 0.30:
                self._k = max(self.k_min, self._k - 1)
            elif rate > 0.95:
                self._k = min(self.k_max, self._k + 2)
            elif rate > 0.80:
                self._k = min(self.k_max, self._k + 1)
            logger.debug(f"DynamicK: rate={rate:.2f} → k={self._k}")

    # keep old interface for backward compat
    def record(self, proposed: int, accepted: int) -> None:
        self.record_and_update(proposed, accepted)

    def update_k(self) -> None:
        pass  # integrated into record_and_update

    @property
    def current_k(self) -> int:
        return self._k


class LayerSkipMonitor:
    """
    Monitors hidden state changes across layers and decides whether a layer
    can be skipped based on cosine similarity to the previous state.
    """

    def __init__(self, threshold: float = 0.01, skip_window: int = 4) -> None:
        self.threshold = threshold
        self.skip_window = skip_window
        self._prev_h: Optional[torch.Tensor] = None
        self._skip_count: int = 0

    def should_skip(self, h_i: torch.Tensor) -> bool:
        """
        Return True if this layer can be skipped.
        Uses cosine similarity delta between current and previous hidden state.
        """
        if self.threshold <= 0.0:
            return False

        if self._prev_h is None:
            self._prev_h = h_i.detach()
            return False

        # Flatten to 2D for cosine similarity
        h_flat = h_i.detach().view(-1)
        prev_flat = self._prev_h.view(-1)

        cos_sim = F.cosine_similarity(h_flat.unsqueeze(0), prev_flat.unsqueeze(0)).item()
        delta = 1.0 - cos_sim

        self._prev_h = h_i.detach()

        if delta < self.threshold and self._skip_count < self.skip_window:
            self._skip_count += 1
            return True

        self._skip_count = 0
        return False

    def reset(self) -> None:
        self._prev_h = None
        self._skip_count = 0


class AcceleratedInference:
    """
    Speculative decoding inference engine.

    Uses a small draft model to propose K tokens at a time, then
    verifies them with one full forward pass through the larger verifier
    (ShardedModel). Accepts/rejects tokens using the speculative sampling rule.

    Features:
    - Persistent draft KV cache across decode steps (avoids redundant recomputation)
    - DynamicKController to adapt K based on acceptance rate
    """

    def __init__(
        self,
        verifier: "ShardedModel",
        draft_model_name: Optional[str],
        inference_config: InferenceConfig,
        scheduler: ShardScheduler,
        probe: AdaptiveShardProbe,
    ) -> None:
        self.verifier = verifier
        self.inference_config = inference_config
        self.scheduler = scheduler
        self.probe = probe

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load draft model
        self.draft_model: Optional[torch.nn.Module] = None
        if draft_model_name is not None:
            # Validate tokenizer compatibility before loading the model weights
            validate_draft_tokenizer(verifier.tokenizer, draft_model_name)

            try:
                logger.info(f"Loading draft model: {draft_model_name}")
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    draft_model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                )
                self.draft_model.eval()
                logger.info("Draft model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load draft model '{draft_model_name}': {e}. "
                               "Falling back to greedy decoding.")
                self.draft_model = None

        # Dynamic K controller
        if inference_config.dynamic_k:
            self.k_controller = DynamicKController(
                k_init=inference_config.speculative_k,
                k_min=inference_config.k_min,
                k_max=inference_config.k_max,
            )
        else:
            self.k_controller = None

        # Persistent draft KV cache
        self.draft_past_kv = None
        self.draft_context_len: int = 0

    def reset_draft_cache(self) -> None:
        """Clear the draft model's KV cache for a new generation."""
        self.draft_past_kv = None
        self.draft_context_len = 0

    def _draft_step(self, context: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate k draft tokens from the draft model using persistent KV cache.

        Args:
            context: full token sequence (batch, seq_len)
            k: number of draft tokens to generate

        Returns:
            draft_tokens: (batch, k) proposed token ids
            draft_logits: (batch, k, vocab) corresponding logits
        """
        assert self.draft_model is not None

        context = context.to(self.device)

        # Only pass new tokens to the draft model when cache is warm
        if self.draft_past_kv is not None:
            new_tokens = context[:, self.draft_context_len:]
        else:
            new_tokens = context

        draft_tokens_list: List[torch.Tensor] = []
        draft_logits_list: List[torch.Tensor] = []

        current_input = new_tokens

        with torch.no_grad():
            for _ in range(k):
                out = self.draft_model(
                    input_ids=current_input,
                    past_key_values=self.draft_past_kv,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :]  # (batch, vocab)
                self.draft_past_kv = out.past_key_values
                self.draft_context_len = context.shape[1] + len(draft_tokens_list)

                # Sample next token
                next_token = _sample_token(logits, temperature=self.inference_config.temperature)
                draft_tokens_list.append(next_token)
                draft_logits_list.append(logits.unsqueeze(1))

                # Feed predicted token back
                current_input = next_token

        draft_tokens = torch.cat(draft_tokens_list, dim=1)  # (batch, k)
        draft_logits = torch.cat(draft_logits_list, dim=1)  # (batch, k, vocab)
        return draft_tokens, draft_logits

    def _verify_tokens(
        self,
        context: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run one full forward pass through the verifier on (context + draft_tokens).

        Returns:
            verify_logits: (batch, k+1, vocab) — logits for each draft position + bonus
        """
        full_input = torch.cat([context, draft_tokens], dim=1)
        logits, _ = self.verifier.forward(full_input)
        # We want the logits for positions corresponding to the draft tokens
        # position [-k-1:-1] predict draft tokens, position [-1] is bonus
        k = draft_tokens.shape[1]
        verify_logits = logits[:, -(k + 1):, :]  # (batch, k+1, vocab)
        return verify_logits

    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.

        If no draft model is available, falls back to autoregressive greedy decoding.
        """
        input_ids = input_ids.to(self.device)
        self.reset_draft_cache()

        if self.draft_model is None:
            return self._greedy_generate(input_ids, max_new_tokens, temperature)

        generated = input_ids.clone()
        tokens_generated = 0
        eos_id = getattr(self.verifier.tokenizer, "eos_token_id", None)
        # KV sliding window: evict old context when sequence grows too long
        kv_window = 512

        while tokens_generated < max_new_tokens:
            skip_draft = (
                self.k_controller.skip_draft
                if (self.k_controller and self.draft_model is not None)
                else (self.draft_model is None)
            )

            if skip_draft:
                # Pure autoregressive with big model — one token per step
                verify_logits = self._verify_tokens(generated, generated[:, :0])  # empty draft
                next_tok = _sample_token(verify_logits[:, -1, :], temperature=temperature).squeeze(1)
                generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
                tokens_generated += 1
                if self.k_controller is not None:
                    self.k_controller.record_and_update(proposed=0, accepted=0)
            else:
                k = self.k_controller.current_k if self.k_controller else self.inference_config.speculative_k
                k = min(k, max_new_tokens - tokens_generated)
                if k <= 0:
                    break

                # Draft k tokens (pure VRAM — draft model is pinned, no I/O)
                draft_tokens, draft_logits = self._draft_step(generated, k)

                # Verify with one full forward pass through big sharded model
                verify_logits = self._verify_tokens(generated, draft_tokens)

                # Accept/reject loop
                accepted = 0
                accepted_tokens: List[torch.Tensor] = []

                for i in range(k):
                    draft_tok = draft_tokens[:, i]
                    p_verify = F.softmax(verify_logits[:, i, :] / max(temperature, 1e-8), dim=-1)
                    p_draft = F.softmax(draft_logits[:, i, :] / max(temperature, 1e-8), dim=-1)

                    tok_idx = draft_tok.unsqueeze(-1)
                    pv = p_verify.gather(1, tok_idx).squeeze(1)
                    pd = p_draft.gather(1, tok_idx).squeeze(1)

                    accept_prob = (pv / (pd + 1e-8)).clamp(max=1.0)
                    if (torch.rand_like(accept_prob) <= accept_prob).all():
                        accepted_tokens.append(draft_tok)
                        accepted += 1
                    else:
                        corrected_probs = (p_verify - p_draft).clamp(min=0)
                        corrected_probs = corrected_probs / (corrected_probs.sum(dim=-1, keepdim=True) + 1e-8)
                        corrected_tok = torch.multinomial(corrected_probs, num_samples=1).squeeze(1)
                        accepted_tokens.append(corrected_tok)
                        break

                if accepted == k:
                    bonus_tok = _sample_token(verify_logits[:, -1, :], temperature=temperature)
                    accepted_tokens.append(bonus_tok.squeeze(1))

                if accepted_tokens:
                    new_toks = torch.stack(accepted_tokens, dim=1)
                    generated = torch.cat([generated, new_toks], dim=1)
                    tokens_generated += new_toks.shape[1]
                    self.reset_draft_cache()

                if self.k_controller is not None:
                    self.k_controller.record_and_update(proposed=k, accepted=accepted)

            # Sliding-window KV eviction: keep only last kv_window tokens in context
            if generated.shape[1] > kv_window:
                generated = generated[:, -kv_window:]
                self.reset_draft_cache()

            if eos_id is not None and (generated[:, -1] == eos_id).any():
                break

        return generated

    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        Autoregressive greedy generation using CPU-offloaded KV cache.
        Step 0 processes the full prompt; subsequent steps pass only the last token.
        """
        generated = input_ids.clone()
        eos_id = getattr(self.verifier.tokenizer, "eos_token_id", None)

        self.verifier.forward_engine.reset_kv_cache()

        for tokens_generated in range(max_new_tokens):
            # Prefill on first step, single-token decode thereafter
            input_for_step = generated if tokens_generated == 0 else generated[:, -1:]
            logits = self.verifier.forward_with_kvcache(input_for_step, step=tokens_generated)

            next_logits = logits[:, -1, :]
            next_tok = _sample_token(next_logits, temperature=temperature).squeeze(1)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)

            if eos_id is not None and (next_tok == eos_id).any():
                break

        return generated


def _sample_token(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample next token from logits. If temperature is very low, use argmax.
    Returns (batch, 1) token ids.
    """
    if temperature < 1e-4:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
