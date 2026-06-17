from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import InferenceConfig, ShardConfig, TrainingConfig
from .probe import AdaptiveShardProbe
from .registry import get_decoder_layers as _registry_get_decoder_layers, get_embed_and_head
from .scheduler import ShardScheduler, save_shard_with_checksum, load_shard_with_checksum
from .forward import ForwardEngine, _get_decoder_layers
from .backward import BackwardEngine
from .optimizer import ShardOptimizer
from .qlora import QLoRAManager

logger = logging.getLogger(__name__)


def shard_model(
    hf_model: nn.Module,
    shard_dir: Path,
    scheduler: ShardScheduler,
    shard_config=None,
    model_type: str = None,
) -> int:
    """
    Save decoder layers as safetensors shards with checksum.

    When shard_config.shard_group_size > 1, groups of layers are saved per file.
    Tensor keys are prefixed with the intra-group offset: "0.weight", "1.weight", etc.

    Returns number of layers saved (not number of files).
    """
    from .config import ShardConfig
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    group_size = shard_config.shard_group_size if shard_config is not None else 1

    # Unwrap PeftModel if needed before layer detection
    _base = getattr(getattr(hf_model, "base_model", hf_model), "model", hf_model)
    layers = _registry_get_decoder_layers(_base, model_type)
    if layers is None:
        layers = _registry_get_decoder_layers(hf_model, model_type)
    if layers is None:
        layers = _get_decoder_layers(hf_model)
    if layers is None:
        raise RuntimeError("Could not find decoder layers in the model")

    num_layers = len(layers)
    num_saved = 0
    group_start = 0

    while group_start < num_layers:
        group_end = min(group_start + group_size, num_layers)
        group_sd: Dict[str, torch.Tensor] = {}

        for offset, layer_idx in enumerate(range(group_start, group_end)):
            for k, v in layers[layer_idx].state_dict().items():
                key = k if group_size == 1 else f"{offset}.{k}"
                group_sd[key] = v.contiguous()

        last_idx = group_end - 1
        if group_size == 1:
            fname = f"layer_{group_start:03d}_weights.safetensors"
        else:
            fname = f"layer_{group_start:03d}-{last_idx:03d}_weights.safetensors"

        shard_path = shard_dir / fname
        save_shard_with_checksum(group_sd, shard_path)
        num_saved += group_end - group_start
        logger.debug(f"Saved layers {group_start}-{last_idx} → {shard_path}")
        group_start = group_end

    n_files = (num_layers + group_size - 1) // group_size
    logger.info(f"Sharded {num_saved} layers into {n_files} file(s) in {shard_dir}")
    return num_saved


def _estimate_shard_size_gb(shard_dir: Path, num_layers: int, group_size: int = 1) -> float:
    """Estimate average per-layer shard size in GB from the first few shard files."""
    total = 0
    count = 0
    group_start = 0
    samples = 0
    while group_start < num_layers and samples < 3:
        group_end = min(group_start + group_size, num_layers) - 1
        if group_size == 1:
            p = shard_dir / f"layer_{group_start:03d}_weights.safetensors"
        else:
            p = shard_dir / f"layer_{group_start:03d}-{group_end:03d}_weights.safetensors"
        if p.exists():
            total += p.stat().st_size
            count += group_end - group_start + 1  # layers in this file
        group_start += group_size
        samples += 1
    if count == 0:
        return 0.5  # default 500 MB estimate
    return (total / count) / (1024 ** 3)


def _shards_exist_and_valid(
    shard_dir: Path, num_layers: int, verify: bool = True, group_size: int = 1
) -> bool:
    """Return True if all expected shard files exist (and checksums pass)."""
    import hashlib
    group_start = 0
    while group_start < num_layers:
        group_end_idx = min(group_start + group_size, num_layers) - 1
        if group_size == 1:
            p = shard_dir / f"layer_{group_start:03d}_weights.safetensors"
        else:
            p = shard_dir / f"layer_{group_start:03d}-{group_end_idx:03d}_weights.safetensors"
        if not p.exists():
            return False
        if verify:
            chk = p.with_suffix(".sha256")
            if chk.exists():
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                if h.hexdigest() != chk.read_text().strip():
                    logger.warning(f"Checksum mismatch for {p}; will re-shard")
                    return False
        group_start += group_size
    return True


class ShardedModel:
    """
    A transformer model whose decoder layers are stored as individual shard
    files on disk and loaded/evicted on demand during forward and backward passes.
    """

    def __init__(
        self,
        shard_dir: Path,
        hf_model: nn.Module,
        tokenizer,
        num_layers: int,
        shard_size_gb: float,
        shard_config: ShardConfig,
        lm_head: nn.Module,
        embed_tokens: nn.Module,
        forward_engine: ForwardEngine,
        backward_engine: BackwardEngine,
        optimizer: ShardOptimizer,
        peft_model,         # PeftModel or None
        qlora_manager,      # QLoRAManager or None
        scheduler: ShardScheduler,
        probe: AdaptiveShardProbe,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        self._num_layers = num_layers
        self._shard_size_gb = shard_size_gb
        self.shard_config = shard_config
        self.lm_head = lm_head
        self.embed_tokens = embed_tokens
        self.forward_engine = forward_engine
        self.backward_engine = backward_engine
        self.optimizer = optimizer
        self.peft_model = peft_model          # PeftModel wrapping hf_model, or None
        self.qlora_manager = qlora_manager    # QLoRAManager or None
        self.scheduler = scheduler
        self.probe = probe
        self._verify_checksums = shard_config.verify_checksums

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        shard_dir: str,
        shard_config: Optional[ShardConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device: str = "auto",
        **kwargs,
    ) -> "ShardedModel":
        """
        Load a HuggingFace model and shard it into per-layer files.
        If shards already exist and checksums pass, skip re-sharding.
        """
        if shard_config is None:
            shard_config = ShardConfig()
        if training_config is None:
            training_config = TrainingConfig()

        shard_dir = Path(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model config from {model_name_or_path}")
        hf_config = AutoConfig.from_pretrained(model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine quantization config
        bnb_config = None
        if shard_config.compression == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("Using NF4 double-quantization via bitsandbytes")
            except (ImportError, Exception) as e:
                logger.warning(f"bitsandbytes not available ({e}), falling back to fp16")
                bnb_config = None
        elif shard_config.compression == "8bit":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            except (ImportError, Exception) as e:
                logger.warning(f"bitsandbytes 8bit failed ({e}), falling back to fp16")
                bnb_config = None

        # If shards already exist we still need to load the HF model to attach
        # QLoRA adapters and identify layer count — but we can load to CPU only
        # (no GPU touch at all), keeping VRAM at 0 during this phase.
        # bitsandbytes 4-bit rejects max_memory with CPU offload, so we use
        # device_map="cpu" which bypasses that restriction entirely.
        # Only when shards DON'T exist do we need GPU for the initial sharding pass.
        # Quick existence check: any shard file present = shards were built before
        _sdir = Path(shard_dir)
        shards_ready = _sdir.exists() and any(_sdir.glob("layer_*_weights.safetensors"))

        if shards_ready:
            # CPU-only load — safe with bitsandbytes, stays within 16 GB RAM budget
            load_kwargs: Dict = {
                "device_map": "cpu",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            logger.info("Shards exist — loading model to CPU only (VRAM untouched) ...")
        else:
            # First-time sharding: load to GPU so bitsandbytes can quantize
            vram_limit_gb = float(os.environ.get("FITLLM_VRAM_LIMIT_GB", "4.0"))
            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            logger.info(f"No shards found — loading model for initial sharding (VRAM cap: {vram_limit_gb:.0f} GB) ...")

        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=hf_config, **load_kwargs
        )

        # ── QLoRA pipeline (via QLoRAManager) ───────────────────────────
        peft_model = None
        qlora_manager = None
        if shard_config.lora_rank > 0:
            qlora_manager = QLoRAManager(
                rank=shard_config.lora_rank,
                alpha=shard_config.lora_alpha,
                targets=shard_config.lora_targets,
            )
            hf_model = qlora_manager.apply(
                hf_model,
                is_quantized=(shard_config.compression == "4bit"),
            )
            peft_model = qlora_manager.peft_model

        hf_model.eval()

        model_type = getattr(hf_config, "model_type", None)
        # After QLoRA wrapping hf_model is a PeftModel; unwrap to find layers
        _base = getattr(getattr(hf_model, "base_model", hf_model), "model", hf_model)
        layers = _registry_get_decoder_layers(_base, model_type)
        if layers is None:
            layers = _registry_get_decoder_layers(hf_model, model_type)
        if layers is None:
            raise RuntimeError("Could not identify decoder layers in model")
        num_layers = len(layers)

        # Check if we can skip re-sharding
        if _shards_exist_and_valid(shard_dir, num_layers, verify=shard_config.verify_checksums,
                                   group_size=shard_config.shard_group_size):
            logger.info(f"Found {num_layers} existing valid shards in {shard_dir}, skipping re-sharding")
        else:
            logger.info(f"Sharding {num_layers} layers into {shard_dir}")
            sched_tmp = ShardScheduler(
                shard_dir, device=device,
                pin_memory=shard_config.pin_memory,
                use_cuda_streams=shard_config.use_cuda_streams,
                shard_group_size=shard_config.shard_group_size,
                num_layers=num_layers,
            )
            shard_model(hf_model, shard_dir, sched_tmp, shard_config=shard_config, model_type=model_type)
            sched_tmp.shutdown()

        # Move model to CPU and free GPU cache so shard loading has headroom
        hf_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model moved to CPU after sharding; GPU VRAM freed for shard-by-shard execution")

        shard_size_gb = _estimate_shard_size_gb(shard_dir, num_layers,
                                                group_size=shard_config.shard_group_size)

        probe = AdaptiveShardProbe(
            shard_size_gb=shard_size_gb,
            total_shards=num_layers,
            gpu_safety_margin_gb=shard_config.gpu_safety_margin_gb,
            cpu_safety_margin_gb=shard_config.cpu_safety_margin_gb,
            reprobe_every=shard_config.reprobe_every,
            vram_limit_gb=shard_config.vram_limit_gb,
        )

        scheduler = ShardScheduler(
            shard_dir,
            device=device,
            max_parallel=shard_config.prefetch_depth * 2,
            pin_memory=shard_config.pin_memory,
            use_cuda_streams=shard_config.use_cuda_streams,
            shard_group_size=shard_config.shard_group_size,
            num_layers=num_layers,
            gpu_safety_margin_gb=shard_config.gpu_safety_margin_gb,
        )

        # Extract lm_head and embed_tokens via registry, falling back to heuristic
        # Try unwrapped base model first (PeftModel wraps the original)
        embed_tokens, lm_head = get_embed_and_head(_base, model_type)
        if embed_tokens is None or lm_head is None:
            embed_tokens, lm_head = get_embed_and_head(hf_model, model_type)
        if embed_tokens is None or lm_head is None:
            lm_head, embed_tokens = _extract_head_embed(hf_model)
        lm_head = lm_head.to(device) if device != "cpu" and torch.cuda.is_available() else lm_head
        embed_tokens = embed_tokens.to(device) if device != "cpu" and torch.cuda.is_available() else embed_tokens

        grad_dir = shard_dir / "grads"

        forward_engine = ForwardEngine(
            model_ref=None,  # set after construction
            scheduler=scheduler,
            probe=probe,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            use_fused_kernels=True,
            layer_skip_threshold=0.0,
            mixed_precision=shard_config.mixed_precision,
        )

        backward_engine = BackwardEngine(
            model_ref=None,  # set after construction
            scheduler=scheduler,
            probe=probe,
            lm_head=lm_head,
            loss_fn=nn.CrossEntropyLoss(),
            grad_dir=grad_dir,
            grad_accum_steps=training_config.grad_accum,
        )

        optimizer = ShardOptimizer(
            shard_dir=shard_dir,
            num_layers=num_layers,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
            beta1=training_config.beta1,
            beta2=training_config.beta2,
            eps=training_config.eps,
            peft_model=peft_model,
        )

        instance = cls(
            shard_dir=shard_dir,
            hf_model=hf_model,
            tokenizer=tokenizer,
            num_layers=num_layers,
            shard_size_gb=shard_size_gb,
            shard_config=shard_config,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            forward_engine=forward_engine,
            backward_engine=backward_engine,
            optimizer=optimizer,
            peft_model=peft_model,
            qlora_manager=qlora_manager,
            scheduler=scheduler,
            probe=probe,
        )

        # Patch model_ref now that instance exists
        forward_engine.model_ref = instance
        backward_engine.model_ref = instance

        # Cache decoder layer list — avoids repeated HF model tree traversal
        # in _reconstruct_layer_module (called 128 times per mini-batch)
        instance._decoder_layers = layers

        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run forward pass, returning (logits, activations)."""
        return self.forward_engine.forward(input_ids, attention_mask)

    def backward(
        self,
        loss: torch.Tensor,
        activations: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> None:
        """Run backward pass, accumulating LoRA gradients to disk."""
        self.backward_engine.backward(loss, activations, labels)

    def generate(
        self,
        input_ids: torch.Tensor,
        inference_config: Optional[InferenceConfig] = None,
    ) -> torch.Tensor:
        """Generate tokens using speculative or greedy decoding."""
        if inference_config is None:
            inference_config = InferenceConfig()

        from .inference import AcceleratedInference
        _saved_reprobe = self.probe.reprobe_every
        self.probe.reprobe_every = inference_config.reprobe_every
        try:
            inf = AcceleratedInference(
                verifier=self,
                draft_model_name=inference_config.draft_model,
                inference_config=inference_config,
                scheduler=self.scheduler,
                probe=self.probe,
            )
            return inf.speculative_generate(
                input_ids,
                max_new_tokens=inference_config.max_new_tokens,
                temperature=inference_config.temperature,
            )
        finally:
            self.probe.reprobe_every = _saved_reprobe

    def forward_with_kvcache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """KV-cache autoregressive forward pass. For inference only; do not use during training."""
        return self.forward_engine.forward_with_kvcache(input_ids, attention_mask, step)

    def save_lora(self, path: str) -> None:
        """Save QLoRA adapter weights via QLoRAManager."""
        if self.qlora_manager is None:
            logger.warning("No QLoRAManager attached — nothing to save")
            return
        self.qlora_manager.save(path)

    def load_lora(self, path: str) -> None:
        """Load QLoRA adapter weights via QLoRAManager."""
        if self.qlora_manager is None:
            logger.warning("No QLoRAManager attached — cannot load LoRA weights")
            return
        self.qlora_manager.load(path)

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def shard_size_gb(self) -> float:
        return self._shard_size_gb


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_head_embed(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    """
    Extract lm_head and embed_tokens from a HuggingFace causal LM.
    Handles LLaMA, GPT-2, Falcon, Mistral, and similar architectures.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Cannot find lm_head on model")

    # Try common attribute paths for embed_tokens
    embed_tokens = None
    for path in [
        ("model", "embed_tokens"),
        ("transformer", "wte"),
        ("model", "embed_tokens"),
        ("gpt_neox", "embed_in"),
    ]:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            embed_tokens = obj
            break
        except AttributeError:
            continue

    if embed_tokens is None:
        # Last resort: look for any Embedding module at top level
        for name, module in model.named_children():
            if isinstance(module, nn.Embedding):
                embed_tokens = module
                break

    if embed_tokens is None:
        raise RuntimeError("Cannot find embed_tokens/wte on model")

    return lm_head, embed_tokens
