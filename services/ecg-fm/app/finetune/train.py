"""LoRA fine-tuning scaffold for ECG-FM.

Reuses the same pattern as EchoPrime's fine-tune module. Dataset is a JSONL
of {ecg_path, label_dict} where label_dict can include:
  - "arrhythmia": {"afib": 1, "rbbb": 0, ...}
  - "intervals":  {"pr_interval": 160.0, "qrs_duration": 100.0, ...}
  - "stemi":      {"anterior": 0, "inferior": 1, "lateral": 0}

Usage:
    python -m app.finetune.train \\
        --train-data /data/ecg_train.jsonl \\
        --output-dir /checkpoints/ecgfm-lora \\
        --lora-r 16 --epochs 10
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class ECGExample:
    ecg_path: str
    arrhythmia: Dict[str, int] = field(default_factory=dict)
    intervals: Dict[str, float] = field(default_factory=dict)
    stemi: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ECGExample":
        return cls(
            ecg_path=d["ecg_path"],
            arrhythmia=d.get("arrhythmia", {}),
            intervals=d.get("intervals", {}),
            stemi=d.get("stemi", {}),
            extra={k: v for k, v in d.items()
                   if k not in {"ecg_path", "arrhythmia", "intervals", "stemi"}},
        )


class ECGDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.path = Path(jsonl_path)
        self.examples: List[ECGExample] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(ECGExample.from_dict(json.loads(line)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "ecg_path": ex.ecg_path,
            "arrhythmia": ex.arrhythmia,
            "intervals": ex.intervals,
            "stemi": ex.stemi,
        }


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Build a PEFT LoraConfig for ECG-FM's transformer blocks."""
    from peft import LoraConfig, TaskType  # type: ignore

    target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )


def train(
    train_jsonl: Path,
    val_jsonl: Optional[Path] = None,
    output_dir: Path = Path("checkpoints/ecgfm-lora"),
    base_model_id: str = "bman03/ECG-FM",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
):
    """End-to-end LoRA fine-tuning entrypoint. See module docstring for usage."""
    from transformers import AutoModel  # type: ignore
    from peft import get_peft_model  # type: ignore
    from ..ingestion import load_ecg

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = AutoModel.from_pretrained(base_model_id, trust_remote_code=True).to(device)
    model = get_peft_model(base, build_lora_config(r=lora_r, alpha=lora_alpha))
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_ds = ECGDataset(train_jsonl)

    for epoch in range(epochs):
        model.train()
        for i, ex in enumerate(train_ds):
            try:
                ecg, _ = load_ecg(ex["ecg_path"])
                ecg = ecg.to(device)
                # NOTE: Replace with a real loss function for your downstream task.
                out = model(ecg)
                loss = out.loss if hasattr(out, "loss") else torch.tensor(0.0, requires_grad=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % 50 == 0:
                    print(f"epoch={epoch} step={i} loss={float(loss):.4f}")
            except Exception as e:
                print(f"epoch={epoch} step={i} skipped: {e}")

        model.save_pretrained(output_dir / f"epoch-{epoch}")

    model.save_pretrained(output_dir / "final")
    print(f"Adapter weights saved to {output_dir / 'final'}")
