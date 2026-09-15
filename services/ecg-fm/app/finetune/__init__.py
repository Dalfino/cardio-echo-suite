"""ECG-FM fine-tuning scaffold (LoRA)."""
from .train import (
    ECGDataset,
    ECGExample,
    build_lora_config,
    train,
)

__all__ = ["ECGDataset", "ECGExample", "build_lora_config", "train"]
