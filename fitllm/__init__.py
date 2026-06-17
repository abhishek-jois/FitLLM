"""
FitLLM: Layer-sharding LLM inference and training system.

Built on top of AirLLM concepts with adaptive sharding, LoRA training,
speculative decoding, and fused kernel support.
"""

from .config import InferenceConfig, ShardConfig, TrainingConfig
from .model import ShardedModel
from .trainer import LoRATrainer

__all__ = [
    "ShardedModel",
    "LoRATrainer",
    "TrainingConfig",
    "ShardConfig",
    "InferenceConfig",
]

__version__ = "0.1.0"
