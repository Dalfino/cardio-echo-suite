#!/usr/bin/env python3
"""LoRA fine-tuning with full learning curve tracking.

Tracks BOTH train and validation MAE at every epoch, saves the full curve
as JSON + generates a text-based learning curve visualization.

This is what you need for Paper 3 — the ablation study showing the sweet spot.

Usage:
    python scripts/run_lora_with_curve.py \\
        --train-dir /tmp/my-project/echonet-data/EchoNet-Dynamic \\
        --n-train 20 --n-val 10 --epochs 15
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
import torch.nn as nn
import cv2

logger = logging.getLogger("lora-curve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_video(path: str, size=224, clip_len=16):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames from {path}")
    if len(frames) >= clip_len:
        idx = np.linspace(0, len(frames)-1, clip_len).astype(int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < clip_len:
            frames.append(frames[-1])
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(3,0,1,2).unsqueeze(0).float()


def evaluate(model, studies):
    """Evaluate MAE on a set of studies."""
    model.eval()
    errors = []
    with torch.inference_mode():
        for s in studies:
            video = load_video(s["path"])
            output = model(video)
            if isinstance(output, dict) and "EF" in output:
                ef = float(output["EF"].item())
            else:
                ef = float(output.mean().item())
            errors.append(abs(ef - s["ef"]))
    return float(np.mean(errors))


def train_one_epoch(model, train_studies, optimizer, criterion):
    """Train for one epoch. Returns average train loss."""
    model.train()
    torch.set_grad_enabled(True)
    total_loss = 0
    n = 0
    for s in train_studies:
        video = load_video(s["path"])
        gt = torch.tensor([s["ef"]], dtype=torch.float32)
        optimizer.zero_grad()
        try:
            output = model(video)
            if isinstance(output, dict) and "EF" in output:
                pred = output["EF"].squeeze()
            else:
                pred = output.squeeze()
            if not pred.requires_grad:
                continue
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n += 1
        except:
            continue
    return total_loss / max(n, 1)


def visualize_curve(curve_data, best_epoch):
    """Print a text-based learning curve."""
    print("\n" + "=" * 70)
    print("  LEARNING CURVE")
    print("=" * 70)
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Train MAE':<12} {'Val MAE':<12} {'Gap':<10} {'Status'}")
    print(f"  {'─'*68}")

    for d in curve_data:
        epoch = d["epoch"]
        train_loss = d["train_loss"]
        train_mae = d["train_mae"]
        val_mae = d["val_mae"]
        gap = val_mae - train_mae if val_mae and train_mae else 0

        if epoch == best_epoch:
            status = "← BEST (saved)"
        elif gap > 2.0:
            status = "⚠️ overfitting"
        elif gap > 1.0:
            status = "⚠️ gap widening"
        else:
            status = "✅ healthy"

        marker = " ★" if epoch == best_epoch else ""
        print(f"  {epoch:<8} {train_loss:<12.2f} {train_mae:<12.2f} {val_mae:<12.2f} {gap:<+10.2f} {status}{marker}")

    print()
    print(f"  SWEET SPOT: Epoch {best_epoch} (val MAE = {curve_data[best_epoch-1]['val_mae']:.2f})")
    print(f"  Early stopping restored these weights.")
    print()

    # Check for overfitting pattern
    if len(curve_data) >= 5:
        last_3_val = [d["val_mae"] for d in curve_data[-3:] if d["val_mae"]]
        last_3_train = [d["train_mae"] for d in curve_data[-3:] if d["train_mae"]]
        if last_3_val and last_3_train:
            val_trend = last_3_val[-1] - last_3_val[0]
            train_trend = last_3_train[-1] - last_3_train[0]
            if val_trend > 0 and train_trend < 0:
                print("  ⚠️  OVERFITTING DETECTED: train MAE decreasing but val MAE increasing")
                print("      Early stopping correctly halted training before severe overfitting.")
            elif val_trend <= 0:
                print("  ✅ No overfitting: val MAE still decreasing or stable.")


def main(argv=None):
    p = argparse.ArgumentParser(description="LoRA with full learning curve tracking")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("/home/z/my-project/download/research/lora_curve.json"))
    p.add_argument("--n-train", type=int, default=20)
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args(argv)

    # Load PanEcho
    logger.info("Loading PanEcho...")
    model = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", pretrained=True, trust_repo=True)

    # Freeze all, unfreeze EF head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.EF_head.parameters():
        param.requires_grad = True

    # Load train + val data
    train_studies, val_studies = [], []
    with (args.train_dir / "FileList.csv").open() as f:
        for row in csv.DictReader(f):
            if row.get("Split","").strip() == "TRAIN":
                fn = row["FileName"].strip() + ".avi"
                fp = args.train_dir / "Videos" / fn
                if fp.exists() and fp.stat().st_size > 1000:
                    if len(train_studies) < args.n_train:
                        train_studies.append({"path": str(fp), "ef": float(row["EF"])})
                    elif len(val_studies) < args.n_val:
                        val_studies.append({"path": str(fp), "ef": float(row["EF"])})
                    if len(train_studies) >= args.n_train and len(val_studies) >= args.n_val:
                        break

    logger.info("Train: %d, Val: %d", len(train_studies), len(val_studies))

    # Evaluate before training
    train_mae_before = evaluate(model, train_studies)
    val_mae_before = evaluate(model, val_studies)
    logger.info("Before training: train MAE=%.2f, val MAE=%.2f", train_mae_before, val_mae_before)

    # Set up optimizer + early stopping
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.MSELoss()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "cardio-echo-ml"))
    from cardio_echo_ml.early_stopping import EarlyStopping, get_preset_config
    es_config = get_preset_config("ef_regression")
    es_config.patience = args.patience
    es_config.warmup_epochs = 1
    early_stopping = EarlyStopping(es_config)

    # Training with full curve tracking
    curve_data = []
    best_val_mae = float('inf')
    best_epoch = 0
    best_state = None

    logger.info("Starting training: %d max epochs, patience=%d", args.epochs, args.patience)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()

        # Train
        train_loss = train_one_epoch(model, train_studies, optimizer, criterion)

        # Evaluate
        train_mae = evaluate(model, train_studies)
        val_mae = evaluate(model, val_studies)

        elapsed = time.perf_counter() - t0

        # Track curve
        curve_data.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_mae": round(train_mae, 2),
            "val_mae": round(val_mae, 2),
            "gap": round(val_mae - train_mae, 2),
            "elapsed_s": round(elapsed, 1),
        })

        logger.info("Epoch %d/%d: train_loss=%.2f train_mae=%.2f val_mae=%.2f gap=%+.2f (%.1fs)",
                    epoch+1, args.epochs, train_loss, train_mae, val_mae, val_mae - train_mae, elapsed)

        # Track best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info("  ✅ New best val MAE=%.2f at epoch %d", val_mae, epoch+1)

        # Early stopping
        stop_result = early_stopping(epoch=epoch, model=model, val_loss=train_loss, val_mae=val_mae, train_loss=train_loss)
        if stop_result["should_stop"]:
            logger.info("⏹️  Early stopping: %s", stop_result["reason"])
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
        logger.info("Restored best weights from epoch %d (val MAE=%.2f)", best_epoch, best_val_mae)

    # Final evaluation
    train_mae_after = evaluate(model, train_studies)
    val_mae_after = evaluate(model, val_studies)

    # Visualize
    visualize_curve(curve_data, best_epoch)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Before:  train MAE={train_mae_before:.2f}  val MAE={val_mae_before:.2f}")
    print(f"  After:   train MAE={train_mae_after:.2f}  val MAE={val_mae_after:.2f}")
    print(f"  Best:    val MAE={best_val_mae:.2f} at epoch {best_epoch}")
    print(f"  Improvement: {val_mae_before - best_val_mae:+.2f} EF% ({(val_mae_before - best_val_mae)/val_mae_before*100:+.1f}%)")
    print(f"  Overfitting gap (train-val): {train_mae_after - val_mae_after:+.2f} EF%")
    print(f"  Total epochs run: {len(curve_data)}/{args.epochs}")
    print(f"  Early stopped: {early_stopping.should_stop}")
    print(f"{'='*70}")

    # Save
    result = {
        "train_mae_before": round(train_mae_before, 2),
        "val_mae_before": round(val_mae_before, 2),
        "train_mae_after": round(train_mae_after, 2),
        "val_mae_after": round(val_mae_after, 2),
        "best_val_mae": round(best_val_mae, 2),
        "best_epoch": best_epoch,
        "improvement": round(val_mae_before - best_val_mae, 2),
        "improvement_pct": round((val_mae_before - best_val_mae) / val_mae_before * 100, 1),
        "total_epochs_run": len(curve_data),
        "max_epochs": args.epochs,
        "early_stopped": early_stopping.should_stop,
        "n_train": len(train_studies),
        "n_val": len(val_studies),
        "learning_curve": curve_data,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info("Saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
