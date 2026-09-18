#!/usr/bin/env python3
"""Adapter A: Multi-domain LoRA training on public datasets.

Trains LoRA adapters on public cardiac datasets to create a multi-domain
ensemble. This is for RESEARCH ONLY (public data licenses forbid commercial use).

Usage:
    # Train echo adapter on EchoNet-Dynamic (small subset for demo)
    python scripts/train_adapter_a.py \\
        --dataset echonet \\
        --data-dir /tmp/my-project/echonet-data/EchoNet-Dynamic \\
        --output /home/z/my-project/download/research/adapters/echonet \\
        --max-n 20 \\
        --epochs 3

    # Train ECG adapter on PTB-XL
    python scripts/train_adapter_a.py \\
        --dataset ptbxl \\
        --data-dir /tmp/my-project/ptbxl/records100/00000 \\
        --labels /tmp/my-project/ptbxl/ptbxl_database.csv \\
        --output /home/z/my-project/download/research/adapters/ptbxl \\
        --max-n 20 \\
        --epochs 3

Note: This runs on CPU by default. For real training, use Google Colab
(free T4 GPU) or a local GPU. CPU is ~10× slower but works for proof of concept.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import cv2

logger = logging.getLogger("adapter-a")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_echonet_train_data(
    data_dir: Path,
    max_n: int = 20,
) -> List[Dict[str, Any]]:
    """Load EchoNet-Dynamic TRAIN split for LoRA fine-tuning."""
    filelist = data_dir / "FileList.csv"
    videos_dir = data_dir / "Videos"

    studies = []
    with filelist.open() as f:
        for row in csv.DictReader(f):
            if row.get("Split", "").strip() == "TRAIN":
                filename = row["FileName"].strip() + ".avi"
                filepath = videos_dir / filename
                if filepath.exists() and filepath.stat().st_size > 1000:  # Skip empty/partial files
                    studies.append({
                        "video_path": str(filepath),
                        "ef": float(row["EF"]),
                        "filename": filename,
                    })
                    if len(studies) >= max_n:
                        break

    logger.info("Loaded %d TRAIN videos from EchoNet-Dynamic", len(studies))
    return studies


def load_ptbxl_train_data(
    data_dir: Path,
    labels_path: Path,
    max_n: int = 20,
) -> List[Dict[str, Any]]:
    """Load PTB-XL for ECG adapter training."""
    import wfdb

    studies = []
    with labels_path.open() as f:
        for row in csv.DictReader(f):
            ecg_id = row.get("ecg_id", "")
            filename = row.get("filename_lr", "")
            if not filename:
                continue

            # Check if file exists
            hea_path = data_dir / Path(filename).name + ".hea"
            hea_path = data_dir / (Path(filename).stem + ".hea")
            if not hea_path.exists():
                continue

            studies.append({
                "hea_path": str(hea_path).replace(".hea", ""),
                "ecg_id": ecg_id,
                "scp_codes": row.get("scp_codes", ""),
            })
            if len(studies) >= max_n:
                break

    logger.info("Loaded %d PTB-XL ECGs", len(studies))
    return studies


def load_video_tensor(video_path: str, size: int = 224, clip_len: int = 16) -> torch.Tensor:
    """Load video as (1, 3, T, H, W) tensor, ImageNet-normalized."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames from {video_path}")

    if len(frames) >= clip_len:
        idx = np.linspace(0, len(frames) - 1, clip_len).astype(int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < clip_len:
            frames.append(frames[-1])

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float()
    return tensor


def train_echo_adapter(
    data_dir: Path,
    output_dir: Path,
    max_n: int = 20,
    epochs: int = 3,
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """Train LoRA adapter on EchoNet-Dynamic."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "cardio-echo-ml"))

    # Load training data
    studies = load_echonet_train_data(data_dir, max_n)
    if not studies:
        return {"error": "No training data found"}

    # Load PanEcho model
    logger.info("Loading PanEcho model...")
    model = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", pretrained=True, trust_repo=True)
    model.eval()

    # Wrap with LoRA
    from cardio_echo_ml.lora import LoRAFineTuner, HospitalDataset, _default_video_loader

    # Create dataset
    examples = []
    for s in studies:
        examples.append(type("E", (), {
            "video_path": s["video_path"],
            "ef": s["ef"],
            "patient_id": "echonet",
            "extra": {"dataset_name": "EchoNet-Dynamic"},
        })())
    dataset = HospitalDataset(examples, _default_video_loader)

    # Create fine-tuner
    tuner = LoRAFineTuner(
        model=model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=1,  # CPU — batch size 1
    )

    # Evaluate before fine-tuning
    logger.info("Evaluating base model...")
    mae_before = tuner._evaluate_mae(model, dataset)
    logger.info("Base model MAE: %.2f EF%%", mae_before)

    # Fine-tune
    logger.info("Starting LoRA fine-tuning (%d epochs, %d examples)...", epochs, len(studies))
    t0 = time.perf_counter()
    adapter_path = tuner.fit(dataset, output_dir)
    elapsed = time.perf_counter() - t0

    # Evaluate after fine-tuning
    logger.info("Evaluating fine-tuned model...")
    tuned_model = tuner.load_adapter(adapter_path)
    mae_after = tuner._evaluate_mae(tuned_model, dataset)

    improvement = mae_before - mae_after

    result = {
        "dataset": "EchoNet-Dynamic TRAIN",
        "n_examples": len(studies),
        "epochs": epochs,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "mae_before": round(mae_before, 2),
        "mae_after": round(mae_after, 2),
        "improvement": round(improvement, 2),
        "improvement_pct": round(improvement / mae_before * 100, 1) if mae_before > 0 else 0,
        "elapsed_s": round(elapsed, 1),
        "adapter_path": str(adapter_path),
        "license": "research_only",
    }

    logger.info("=" * 60)
    logger.info("✅ LoRA training complete")
    logger.info("   MAE before: %.2f EF%%", mae_before)
    logger.info("   MAE after:  %.2f EF%%", mae_after)
    logger.info("   Improvement: %s EF%% (%.1f%%)", round(improvement, 2), result["improvement_pct"])
    logger.info("   Time: %.1f seconds", elapsed)
    logger.info("=" * 60)

    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train Adapter A (multi-domain LoRA on public data)")
    p.add_argument("--dataset", choices=["echonet", "ptbxl"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-n", type=int, default=20)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    args = p.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "echonet":
        result = train_echo_adapter(
            data_dir=args.data_dir,
            output_dir=args.output,
            max_n=args.max_n,
            epochs=args.epochs,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
    else:
        result = {"error": "PTB-XL adapter training not yet implemented (needs ECG-FM weights)"}

    # Save results
    results_path = args.output.parent / f"{args.output.name}_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Results saved to %s", results_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
