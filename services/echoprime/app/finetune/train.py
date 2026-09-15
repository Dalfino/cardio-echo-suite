"""LoRA fine-tuning scaffold for EchoPrime.

This module provides a thin wrapper around PEFT (Parameter-Efficient
Fine-Tuning) to fine-tune EchoPrime on your institution's echo reports.

Dataset format
--------------
A JSONL file where each line is:

    {
      "video_path": "/data/echo/study123.mp4",
      "view": "A4C",
      "ef": 58.0,
      "report_text": "Normal LV systolic function. EF = 58%..."
    }

Run
---
    python -m app.finetune.train \
        --train-data /data/reports_train.jsonl \
        --val-data   /data/reports_val.jsonl \
        --output-dir /checkpoints/echoprime-lora \
        --lora-r 16 --lora-alpha 32 --epochs 5

The resulting adapter weights can be merged into the base model via
`app.finetune.merge` or loaded dynamically with `peft.PeftModel.from_pretrained`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

@dataclass
class EchoReportExample:
    video_path: str
    view: str
    ef: float
    report_text: str
    extra: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EchoReportExample":
        return cls(
            video_path=d["video_path"],
            view=d.get("view", "unknown"),
            ef=float(d.get("ef", float("nan"))),
            report_text=d.get("report_text", ""),
            extra={k: v for k, v in d.items()
                   if k not in {"video_path", "view", "ef", "report_text"}},
        )


class EchoReportDataset(Dataset):
    """JSONL-backed dataset for EchoPrime fine-tuning."""

    def __init__(self, jsonl_path: Path):
        self.path = Path(jsonl_path)
        self.examples: List[EchoReportExample] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(EchoReportExample.from_dict(json.loads(line)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "video_path": ex.video_path,
            "view": ex.view,
            "ef": ex.ef,
            "report_text": ex.report_text,
            "extra": ex.extra,
        }


# ---------------------------------------------------------------
# LoRA config builder
# ---------------------------------------------------------------

def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
):
    """Build a PEFT LoraConfig suitable for EchoPrime's transformer blocks.

    Default target modules cover typical attention/MLP layer names used by
    HuggingFace vision-language models (q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, down_proj). Override with `target_modules` for
    model-specific layer names.
    """
    from peft import LoraConfig, TaskType  # type: ignore

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


# ---------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------

def train(
    train_jsonl: Path,
    val_jsonl: Optional[Path] = None,
    output_dir: Path = Path("checkpoints/echoprime-lora"),
    base_model_id: str = "digital-echo/EchoPrime",
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
):
    """End-to-end LoRA fine-tuning entrypoint.

    This is a reference scaffold — adapt the loss/forward pass to your
    institution's exact report schema. The skeleton handles:
      1. Loading the base model from HF Hub.
      2. Wrapping it with PEFT LoRA adapters.
      3. Iterating over your JSONL dataset.
      4. Saving the adapter weights to `output_dir`.
    """
    from transformers import AutoModel, AutoProcessor  # type: ignore
    from peft import get_peft_model  # type: ignore

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = AutoModel.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    lora_cfg = build_lora_config(r=lora_r, alpha=lora_alpha)
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_ds = EchoReportDataset(train_jsonl)
    val_ds = EchoReportDataset(val_jsonl) if val_jsonl else None

    for epoch in range(epochs):
        model.train()
        for i, ex in enumerate(train_ds):
            # NOTE: Replace with a real loss function adapted to your report schema.
            # For example: forward the model on (video, "Draft a structured echo
            # report") and compute cross-entropy against `ex["report_text"]`.
            try:
                inputs = processor(
                    videos=ex["video_path"],
                    text=["Draft a structured echo report based on this video."],
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(**inputs, labels=inputs.get("input_ids"))
                loss = out.loss if hasattr(out, "loss") else out["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % 10 == 0:
                    print(f"epoch={epoch} step={i} loss={float(loss):.4f}")
            except Exception as e:
                print(f"epoch={epoch} step={i} skipped: {e}")

        # Save adapter weights after each epoch
        model.save_pretrained(output_dir / f"epoch-{epoch}")

    # Save final adapter
    model.save_pretrained(output_dir / "final")
    print(f"Adapter weights saved to {output_dir / 'final'}")
