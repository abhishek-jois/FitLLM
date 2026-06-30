from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .model import ShardedModel

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """Format seconds into h:mm:ss or m:ss."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _grad_norm(model: "ShardedModel") -> float:
    """Compute L2 norm of all LoRA parameter gradients."""
    total = 0.0
    if model.qlora_manager is None or model.qlora_manager.peft_model is None:
        return 0.0
    for p in model.qlora_manager.peft_model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def _cosine_lr_scale(step: int, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup then cosine decay. Returns multiplier in [0.0, 1.0]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _clip_grad_norm(model: "ShardedModel", max_norm: float) -> float:
    """Clip LoRA gradients in-place. Returns pre-clip grad norm.
    Uses a single CUDA sync point (.item() called once) to avoid stalling
    the GPU pipeline 64+ times per optimizer step.
    """
    if model.qlora_manager is None or model.qlora_manager.peft_model is None:
        return 0.0
    params = [p for p in model.qlora_manager.peft_model.parameters()
              if p.grad is not None]
    if not params:
        return 0.0
    # Stack all squared norms on GPU, sum, sqrt — single .item() sync
    norms_sq = torch.stack([p.grad.detach().float().norm() ** 2 for p in params])
    total_norm = norms_sq.sum().sqrt().item()
    if max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in params:
                p.grad.detach().mul_(clip_coef)
    return total_norm


def _clip_lora_accum_grads(backward_engine, max_norm: float) -> float:
    """Clip grads stored in backward_engine._lora_grad_accum (shard-stationary mode).

    In shard-stationary mode param.grad is zeroed before each mini-batch, so the
    real accumulated gradients live in _lora_grad_accum, not param.grad.  This
    function computes the global L2 norm across all accumulated tensors and clips
    them in-place, then returns the pre-clip norm for logging.
    """
    accum = backward_engine._lora_grad_accum
    all_grads = [g for layer_grads in accum.values() for g in layer_grads.values()]
    if not all_grads:
        return 0.0
    total_norm = sum(g.float().norm().item() ** 2 for g in all_grads) ** 0.5
    if max_norm > 0 and total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for layer_grads in accum.values():
            for name in layer_grads:
                layer_grads[name].mul_(clip_coef)
    return total_norm


# ── Dataset format auto-detection & normalization ────────────────────────────
#
# Supported dataset schemas (auto-detected from column names):
#
#  Format              Columns detected                        Examples
#  ──────────────────  ──────────────────────────────────────  ─────────────────────────────
#  alpaca              instruction, output [, input]           tatsu-lab/alpaca
#  dolly               instruction, response [, context]       databricks/databricks-dolly-15k
#  chat (messages)     messages  (list of {role, content})     HuggingFaceH4/ultrachat_200k
#  chat (conversations)conversations (list of {from, value})   Open-Orca/OpenOrca, ShareGPT
#  prompt+completion   prompt, completion                      openai/summarize_from_feedback
#  prompt+response     prompt, response                        Anthropic/hh-rlhf
#  question+answer     question, answer                        truthful_qa, trivia_qa
#  question+best_ans   question, best_answer                   truthful_qa (mc1)
#  context+question+ans context, question, answer              squad, quac
#  input+output        input, output  (no instruction col)     generic instruct
#  text (raw)          text                                    any pretraining / SFT corpus
#  content (raw)       content                                 some crawled corpora
#  raw pre-tokenized   input_ids                               passed through unchanged
#
# After normalization all rows become a single "text" string.
# Response-only loss masking looks for the response separator in the text.

_RESPONSE_MARKER = "### Response:\n"
_ASSISTANT_MARKER = "<|assistant|>\n"


def _detect_format(columns) -> str:
    cols = set(columns)
    if "input_ids" in cols:
        return "pretokenized"
    if "instruction" in cols and "output" in cols:
        return "alpaca"
    if "instruction" in cols and "response" in cols:
        return "dolly"
    if "messages" in cols:
        return "chat_messages"
    if "conversations" in cols:
        return "chat_conversations"
    if "prompt" in cols and "completion" in cols:
        return "prompt_completion"
    if "prompt" in cols and "response" in cols:
        return "prompt_response"
    if "context" in cols and "question" in cols and "answer" in cols:
        return "context_question_answer"
    if "question" in cols and "best_answer" in cols:
        return "question_best_answer"
    if "question" in cols and "answer" in cols:
        return "question_answer"
    if "input" in cols and "output" in cols:
        return "input_output"
    if "text" in cols:
        return "text"
    if "content" in cols:
        return "content"
    return "unknown"


def _normalize_to_text(examples: dict, fmt: str) -> dict:
    """Convert any supported dataset format to a list of text strings."""
    n = len(next(iter(examples.values())))
    texts = []

    for i in range(n):
        if fmt == "alpaca":
            inst = examples["instruction"][i] or ""
            inp  = (examples.get("input") or [""] * n)[i] or ""
            out  = examples["output"][i] or ""
            if inp.strip():
                text = (
                    "Below is an instruction that describes a task, paired with an input "
                    "that provides further context. Write a response that appropriately "
                    f"completes the request.\n\n### Instruction:\n{inst}\n\n"
                    f"### Input:\n{inp}\n\n{_RESPONSE_MARKER}{out}"
                )
            else:
                text = (
                    "Below is an instruction that describes a task. Write a response "
                    f"that appropriately completes the request.\n\n"
                    f"### Instruction:\n{inst}\n\n{_RESPONSE_MARKER}{out}"
                )

        elif fmt == "dolly":
            inst = examples["instruction"][i] or ""
            ctx  = (examples.get("context") or [""] * n)[i] or ""
            resp = examples["response"][i] or ""
            if ctx.strip():
                text = (
                    "Below is an instruction that describes a task, paired with an input "
                    "that provides further context. Write a response that appropriately "
                    f"completes the request.\n\n### Instruction:\n{inst}\n\n"
                    f"### Input:\n{ctx}\n\n{_RESPONSE_MARKER}{resp}"
                )
            else:
                text = (
                    "Below is an instruction that describes a task. Write a response "
                    f"that appropriately completes the request.\n\n"
                    f"### Instruction:\n{inst}\n\n{_RESPONSE_MARKER}{resp}"
                )

        elif fmt == "chat_messages":
            # [{role: "user"|"assistant"|"system", content: "..."}]
            turns = examples["messages"][i]
            parts = []
            for turn in (turns if isinstance(turns, list) else []):
                role    = (turn.get("role") or "user").lower()
                content = turn.get("content") or ""
                if role == "system":
                    parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    parts.append(f"{_ASSISTANT_MARKER}{content}")
            text = "\n".join(parts)

        elif fmt == "chat_conversations":
            # [{from: "human"|"gpt"|"system", value: "..."}]
            turns = examples["conversations"][i]
            parts = []
            for turn in (turns if isinstance(turns, list) else []):
                speaker = (turn.get("from") or "human").lower()
                content = turn.get("value") or ""
                if speaker == "system":
                    parts.append(f"<|system|>\n{content}")
                elif speaker in ("human", "user"):
                    parts.append(f"<|user|>\n{content}")
                else:
                    parts.append(f"{_ASSISTANT_MARKER}{content}")
            text = "\n".join(parts)

        elif fmt == "prompt_completion":
            p = examples["prompt"][i] or ""
            c = examples["completion"][i] or ""
            text = f"{p}{_RESPONSE_MARKER}{c}"

        elif fmt == "prompt_response":
            p = examples["prompt"][i] or ""
            r = examples["response"][i] or ""
            text = f"{p}{_RESPONSE_MARKER}{r}"

        elif fmt == "question_answer":
            q = examples["question"][i] or ""
            # answer may be a string or a dict {"text": [...]}
            raw_a = examples["answer"][i]
            if isinstance(raw_a, dict):
                a = (raw_a.get("text") or [""])[0]
            elif isinstance(raw_a, list):
                a = raw_a[0] if raw_a else ""
            else:
                a = raw_a or ""
            text = f"### Question:\n{q}\n\n{_RESPONSE_MARKER}{a}"

        elif fmt == "question_best_answer":
            q = examples["question"][i] or ""
            a = examples["best_answer"][i] or ""
            text = f"### Question:\n{q}\n\n{_RESPONSE_MARKER}{a}"

        elif fmt == "context_question_answer":
            ctx = examples["context"][i] or ""
            q   = examples["question"][i] or ""
            raw_a = examples["answer"][i]
            if isinstance(raw_a, dict):
                a = (raw_a.get("text") or [""])[0]
            elif isinstance(raw_a, list):
                a = raw_a[0] if raw_a else ""
            else:
                a = raw_a or ""
            text = f"### Context:\n{ctx}\n\n### Question:\n{q}\n\n{_RESPONSE_MARKER}{a}"

        elif fmt == "input_output":
            inp = examples["input"][i] or ""
            out = examples["output"][i] or ""
            text = f"### Instruction:\n{inp}\n\n{_RESPONSE_MARKER}{out}"

        elif fmt == "content":
            text = examples["content"][i] or ""

        else:  # "text" or fallback
            text = examples["text"][i] or ""

        texts.append(text)

    return {"text": texts}


def _normalize_dataset(dataset, fmt: str):
    """Apply _normalize_to_text to the whole dataset, removing original columns."""
    keep = [c for c in dataset.column_names if c == "text"]
    remove = [c for c in dataset.column_names if c not in ("text",)]
    return dataset.map(
        lambda ex: _normalize_to_text(ex, fmt),
        batched=True,
        remove_columns=remove,
        desc=f"Normalizing ({fmt})",
    )


def _preprocess_batch(batch: dict, tokenizer, config) -> tuple:
    """Mask prompt tokens, apply causal shift. Returns (input_ids_fwd, attention_mask, labels)."""
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    full_labels = _mask_prompt_tokens(input_ids, tokenizer)
    # Mask padding positions — attention_mask=0 means padding token.
    # Without this, hundreds of pad tokens per example inflate the loss to ~47
    # instead of the expected ~3-4 for a pre-trained model.
    if attention_mask is not None:
        full_labels[attention_mask == 0] = -100
    labels = full_labels[:, 1:].contiguous()
    input_ids_fwd = input_ids[:, :-1].contiguous()
    if attention_mask is not None:
        attention_mask = attention_mask[:, :-1].contiguous()
    return input_ids_fwd, attention_mask, labels


def _mask_prompt_tokens(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Build labels where prompt tokens are -100 (masked from loss).
    Searches for the response separator marker and masks everything before it.
    Tries both _RESPONSE_MARKER and _ASSISTANT_MARKER to cover all formats.
    """
    labels = input_ids.clone()

    for marker in (_RESPONSE_MARKER, _ASSISTANT_MARKER):
        response_ids = tokenizer.encode(marker, add_special_tokens=False)
        resp_len = len(response_ids)
        if resp_len == 0:
            continue

        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            found = -1
            for i in range(len(ids) - resp_len):
                if ids[i:i + resp_len] == response_ids:
                    found = i + resp_len
                    break
            if found != -1:
                labels[b, :found] = -100
                break  # marker found, stop trying other markers
        else:
            continue
        break

    return labels


class LoRATrainer:
    """
    Trains a ShardedModel using LoRA adapters with gradient accumulation.
    Features: cosine LR schedule with warmup, gradient clipping, mixed precision
    (bfloat16 on A100), alpaca instruction formatting, response-only loss masking.
    """

    def __init__(self, model: ShardedModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if config.log_wandb:
            try:
                import wandb
                wandb.init(project="fitllm", config=config.__dict__)
                self._wandb = wandb
                logger.info("WandB logging enabled")
            except ImportError:
                logger.warning("wandb not installed; disabling WandB logging")
                self._wandb = None
                self.config.log_wandb = False
        else:
            self._wandb = None

    def train(self, dataset, start_step: int = 0, start_sample: int = 0) -> None:
        """
        Main training loop.

        Args:
            dataset: A HuggingFace dataset or iterable of dicts with
                     'input_ids' and optionally 'attention_mask'.
            start_step: Resume global_step from this value (for LR schedule continuity).
            start_sample: Number of samples already consumed; fast-forwards the iterator.
        """
        tokenizer = self.model.tokenizer
        config = self.config

        # ── Auto-detect dataset format and normalize to text ─────────────────
        if hasattr(dataset, "column_names"):
            fmt = _detect_format(dataset.column_names)
            logger.info(f"Dataset format detected: '{fmt}' (columns: {list(dataset.column_names)})")

            if fmt == "unknown":
                raise ValueError(
                    f"Unrecognised dataset columns: {list(dataset.column_names)}. "
                    "FitLLM supports: alpaca, chat (messages/conversations), "
                    "prompt+completion, prompt+response, question+answer, "
                    "context+question+answer, input+output, text, content, "
                    "or pre-tokenized (input_ids)."
                )

            if fmt != "pretokenized":
                # Normalize any format → "text" column
                if fmt != "text":
                    dataset = _normalize_dataset(dataset, fmt)

                # Tokenize text → input_ids
                def tokenize_fn(examples):
                    return tokenizer(
                        examples["text"],
                        truncation=True,
                        max_length=config.max_seq_len,
                        padding="max_length",
                        return_tensors=None,
                    )
                dataset = dataset.map(
                    tokenize_fn, batched=True,
                    remove_columns=dataset.column_names,
                    desc="Tokenizing",
                )

        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,             # randomize order for better generalization
            num_workers=2,            # async prefetch — data never blocks GPU
            pin_memory=True,          # faster CPU→GPU for input_ids/labels
            persistent_workers=True,  # workers stay alive between resets
        )

        data_iter = iter(dataloader)
        global_step = start_step
        total_samples_seen = start_sample

        # Fast-forward past already-seen samples so we don't repeat training data.
        # The dataloader is shuffled so exact order differs, but position in the
        # epoch is preserved — avoids reusing the first N samples from the new shuffle.
        if start_sample > 0:
            skip = start_sample % len(dataset)
            logger.info(f"Fast-forwarding data iterator by {skip} samples (epoch position)")
            for _ in range(skip):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)

        accum_loss = 0.0
        accum_count = 0

        loss_window: deque = deque(maxlen=20)
        step_times: deque = deque(maxlen=10)
        train_start = time.monotonic()
        step_start = time.monotonic()

        # Mixed precision: bfloat16 on A100 (native support), float16 elsewhere
        use_amp = config.mixed_precision and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        print()
        print("=" * 84)
        print(f"  FitLLM Training  |  model: {getattr(tokenizer, 'name_or_path', '?')}")
        print(f"  steps={config.max_steps}  grad_accum={config.grad_accum}  "
              f"lr={config.lr}  seq_len={config.max_seq_len}")
        print(f"  warmup={config.warmup_steps} steps  clip={config.max_grad_norm}  "
              f"amp={'bfloat16' if amp_dtype == torch.bfloat16 else 'float16' if use_amp else 'off'}")
        print("=" * 84)
        print(f"{'Step':>8}  {'Loss':>8}  {'Smooth':>8}  {'GradNorm':>9}  {'LR':>8}  "
              f"{'Tok/s':>7}  {'Elapsed':>8}  {'ETA':>8}")
        print("-" * 84)

        pbar = tqdm(total=config.max_steps, initial=start_step, desc="Training", leave=False)

        # Enable dynamic weight cache — sizes itself to available RAM automatically.
        # user_cap_gb=0 means "use all available RAM minus safety margin".
        scheduler = getattr(self.model, "scheduler", None)
        shard_cfg = getattr(self.model, "shard_config", None)
        user_cap_gb = getattr(shard_cfg, "weight_cache_gb", 0.0)
        cpu_safety_margin = getattr(shard_cfg, "cpu_safety_margin_gb", 2.0)
        if scheduler is not None:
            scheduler.enable_weight_cache(
                cpu_safety_margin_gb=cpu_safety_margin,
                user_cap_gb=user_cap_gb,
            )

        use_shard_stationary = getattr(config, "shard_stationary", False)

        while global_step < config.max_steps:

            # ── SHARD-STATIONARY PATH ──────────────────────────────────────────
            # Each shard loaded once; all grad_accum mini-batches processed through
            # it before eviction.  Reduces shard transfers from 2048 → 128 per step.
            if use_shard_stationary:
                _batch_ids:    List[torch.Tensor]           = []
                _batch_masks:  List[Optional[torch.Tensor]] = []
                _batch_labels: List[torch.Tensor]           = []
                tokens_this_step = 0

                for _ in range(config.grad_accum):
                    try:
                        _raw = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        _raw = next(data_iter)
                    _ids, _mask, _lbl = _preprocess_batch(_raw, tokenizer, config)
                    _batch_ids.append(_ids)
                    _batch_masks.append(_mask)
                    _batch_labels.append(_lbl)
                    tokens_this_step += _ids.shape[1]
                    total_samples_seen += 1

                # Forward: shard 0→63, each loaded once, all batches through it
                _logits_all, _activations_all = \
                    self.model.forward_engine.shard_stationary_forward(
                        _batch_ids, _batch_masks,
                    )

                # Loss for logging (no grad needed here — backward handles grads)
                accum_loss = 0.0
                for bi, (_lg, _lb) in enumerate(zip(_logits_all, _batch_labels)):
                    _b, _s, _v = _lg.shape
                    _flat_lb = _lb.view(_b * _s)
                    _flat_lg = _lg.float().view(_b * _s, _v)
                    _ce = F.cross_entropy(_flat_lg, _flat_lb, ignore_index=-100).item()
                    accum_loss += _ce
                    if global_step == 0 and bi == 0:
                        _n_unmasked = (_flat_lb != -100).sum().item()
                        logger.info(
                            f"[LOSS-DIAG] step=0 batch=0 ce={_ce:.4f} "
                            f"n_unmasked={_n_unmasked}/{_b*_s} "
                            f"label_dtype={_flat_lb.dtype} "
                            f"logit_max={_flat_lg.max():.2f} logit_min={_flat_lg.min():.2f}"
                        )
                avg_loss = accum_loss / config.grad_accum

                # Backward: shard 63→0, each loaded once, all batches backpropped.
                # loss_scale=1/grad_accum ensures mean gradient, not sum.
                self.model.backward_engine.shard_stationary_backward(
                    _activations_all,
                    _batch_labels,
                    loss_scale=1.0 / config.grad_accum,
                )

                tokens_this_batch = tokens_this_step  # total tokens this optimizer step

            # ── ORIGINAL PER-BATCH PATH (unchanged) ───────────────────────────
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)

                full_labels = _mask_prompt_tokens(input_ids, tokenizer)
                if attention_mask is not None:
                    full_labels[attention_mask == 0] = -100
                labels = full_labels[:, 1:].contiguous()
                input_ids_fwd = input_ids[:, :-1].contiguous()
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :-1].contiguous()

                tokens_this_batch = input_ids_fwd.shape[1]

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits, activations = self.model.forward(input_ids_fwd, attention_mask)
                        b, s, v = logits.shape
                        loss = F.cross_entropy(
                            logits.float().view(b * s, v),
                            labels.view(b * s).to(logits.device),
                            ignore_index=-100,
                        )
                else:
                    logits, activations = self.model.forward(input_ids_fwd, attention_mask)
                    b, s, v = logits.shape
                    loss = F.cross_entropy(
                        logits.view(b * s, v),
                        labels.view(b * s).to(logits.device),
                        ignore_index=-100,
                    )

                accum_loss += loss.item()
                accum_count += 1
                total_samples_seen += 1

                loss_scaled = loss / config.grad_accum
                self.model.backward(loss_scaled, activations, labels)

                if accum_count < config.grad_accum:
                    continue  # accumulate more batches before optimizer step

                avg_loss = accum_loss / accum_count

            # ── SHARED: optimizer step + LR schedule + logging ─────────────────
            # In shard-stationary mode the real accumulated grads live in
            # _lora_grad_accum (param.grad is zeroed before each mini-batch).
            if use_shard_stationary:
                gnorm = _clip_lora_accum_grads(
                    self.model.backward_engine, config.max_grad_norm
                )
            else:
                gnorm = _clip_grad_norm(self.model, config.max_grad_norm)

            lr_scale = _cosine_lr_scale(global_step, config.warmup_steps, config.max_steps)
            current_lr = config.lr * lr_scale
            self.model.optimizer.set_lr(current_lr)

            lora_grads = self.model.backward_engine.get_and_zero_grads()
            self.model.optimizer.step_from_memory(lora_grads)
            self.model.optimizer.zero_grad()

            if scheduler is not None:
                scheduler.clear_weight_cache()
                scheduler.warm_cache_async(self.model.num_layers)

            loss_window.append(avg_loss)
            smooth_loss = sum(loss_window) / len(loss_window)
            global_step += 1

            now = time.monotonic()
            step_duration = now - step_start
            step_times.append(tokens_this_batch / max(step_duration, 1e-6))
            step_start = now
            elapsed = now - train_start
            tok_per_sec = sum(step_times) / len(step_times) if step_times else 0.0
            steps_left = config.max_steps - global_step
            eta = (elapsed / global_step) * steps_left if global_step > 0 else 0.0

            log_line = (
                f"{global_step:>6}/{config.max_steps:<6} "
                f"{avg_loss:>8.4f}  "
                f"{smooth_loss:>8.4f}  "
                f"{gnorm:>9.4f}  "
                f"{current_lr:>8.2e}  "
                f"{tok_per_sec:>7.1f}  "
                f"{_format_time(elapsed):>8}  "
                f"{_format_time(eta):>8}"
            )
            print(log_line)
            logger.info(
                f"step={global_step} loss={avg_loss:.4f} smooth={smooth_loss:.4f} "
                f"gnorm={gnorm:.4f} lr={current_lr:.2e} tok/s={tok_per_sec:.1f}"
            )

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "smooth": f"{smooth_loss:.4f}",
                "gnorm": f"{gnorm:.4f}",
                "lr": f"{current_lr:.1e}",
            })
            pbar.update(1)

            if self._wandb is not None:
                self._wandb.log({
                    "loss": avg_loss,
                    "loss_smooth": smooth_loss,
                    "grad_norm": gnorm,
                    "learning_rate": current_lr,
                    "tokens_per_sec": tok_per_sec,
                    "step": global_step,
                })

            accum_loss = 0.0
            accum_count = 0

            if global_step % config.save_every == 0:
                ckpt = self.save_checkpoint(global_step, total_samples_seen)
                print(f"  ↳ checkpoint saved → {ckpt}")

            if global_step >= config.max_steps:
                break

        pbar.close()

        total_time = time.monotonic() - train_start
        print("-" * 84)
        if loss_window:
            print(f"  Training complete in {_format_time(total_time)}  |  final loss={loss_window[-1]:.4f}")
        else:
            print(f"  Training complete in {_format_time(total_time)}")
        print("=" * 84)
        print()

        self.save_checkpoint(global_step, total_samples_seen)
        logger.info("Training complete")

        if self._wandb is not None:
            self._wandb.finish()

    def save_checkpoint(self, step: int, sample_index: int = 0) -> Path:
        """Save LoRA weights, optimizer state, step, and config. Prunes old checkpoints."""
        ckpt_path = self.checkpoint_dir / f"checkpoint_step_{step:06d}.pt"

        checkpoint: dict = {
            "step": step,
            "sample_index": sample_index,
            "optimizer_state": self.model.optimizer.state_dict_cpu(),
            "config": self.config.__dict__,
        }

        if (
            self.model.qlora_manager is not None
            and self.model.qlora_manager.peft_model is not None
        ):
            checkpoint["lora_state_dict"] = {
                k: v.cpu()
                for k, v in self.model.qlora_manager.peft_model.state_dict().items()
                if "lora_A" in k or "lora_B" in k
            }

        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint → {ckpt_path}")
        self._prune_checkpoints(self.config.keep_checkpoints)
        return ckpt_path

    def resume_from_checkpoint(self, path: str) -> tuple:
        """
        Restore training state from a checkpoint file.
        Returns (step, sample_index).
        """
        ckpt = torch.load(path, map_location="cpu")

        if "lora_state_dict" in ckpt and self.model.qlora_manager is not None:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(
                self.model.qlora_manager.peft_model, ckpt["lora_state_dict"]
            )
            logger.info("Restored LoRA adapter weights from checkpoint")

        if "optimizer_state" in ckpt:
            self.model.optimizer.load_state_dict_cpu(ckpt["optimizer_state"])
            logger.info("Restored optimizer state from checkpoint")

        step = ckpt.get("step", 0)
        sample_index = ckpt.get("sample_index", 0)
        logger.info(f"Resuming from step {step}, sample {sample_index}")
        return step, sample_index

    def _prune_checkpoints(self, keep: int) -> None:
        """Delete oldest checkpoints beyond the keep limit."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        while len(checkpoints) > keep:
            oldest = checkpoints.pop(0)
            oldest.unlink(missing_ok=True)
            logger.debug(f"Pruned old checkpoint: {oldest}")
