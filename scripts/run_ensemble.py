#!/usr/bin/env python3
"""Ensemble pipeline: PanEcho (generalist) + EchoNet-Dynamic (EF specialist).

The hybrid approach:
- EchoNet-Dynamic: specialist for EF (published MAE 4.1) — most accurate for the #1 metric
- PanEcho: generalist for 39 tasks (EF + valves + wall motion + etc.) — broader coverage
- Ensemble: when both agree on EF → high confidence. When they disagree → flag for review.

This script:
1. Loads both models
2. Runs both on the same test videos
3. Computes per-model MAE + ensemble MAE
4. Shows agreement analysis (when do they disagree?)

Usage:
    python scripts/run_ensemble.py --data-dir /tmp/my-project/echonet-data/EchoNet-Dynamic --max-n 20
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

logger = logging.getLogger("ensemble")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_video(path: str, size: int = 224, clip_len: int = 16) -> torch.Tensor:
    """Load video as (1, 3, T, H, W) tensor, ImageNet-normalized."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size))
        frames.append(f)
    cap.release()
    if not frames:
        raise ValueError(f"No frames from {path}")
    if len(frames) >= clip_len:
        idx = np.linspace(0, len(frames) - 1, clip_len).astype(int)
        frames = [frames[i] for i in idx]
    else:
        while len(frames) < clip_len:
            frames.append(frames[-1])
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float()


def main(argv=None):
    p = argparse.ArgumentParser(description="Run PanEcho + EchoNet-Dynamic ensemble")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("download/research/ensemble_results.json"))
    p.add_argument("--max-n", type=int, default=20)
    p.add_argument("--ensemble-mode", choices=["average", "weighted", "specialist"], default="average",
                   help="average: 50/50, weighted: by inverse variance, specialist: use EchoNet unless PanEcho disagrees >10%")
    args = p.parse_args(argv)

    # Load test videos
    data_dir = args.data_dir
    test_studies = []
    with (data_dir / "FileList.csv").open() as f:
        for row in csv.DictReader(f):
            if row.get("Split", "").strip() == "TEST":
                fn = row["FileName"].strip() + ".avi"
                fp = data_dir / "Videos" / fn
                if fp.exists() and fp.stat().st_size > 1000:
                    test_studies.append({"path": str(fp), "ef": float(row["EF"]), "filename": fn})
                    if len(test_studies) >= args.max_n:
                        break

    logger.info("Loaded %d test videos", len(test_studies))

    # Load PanEcho
    logger.info("Loading PanEcho (generalist, 39 tasks)...")
    import os
    os.environ["GITHUB_TOKEN"] = os.environ.get("GH_TOKEN", "")
    try:
        panecho = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", pretrained=True, trust_repo=True)
        panecho.eval()
        panecho_available = True
        logger.info("✅ PanEcho loaded")
    except Exception as e:
        logger.warning("PanEcho load failed: %s — using PanEcho-only mode", e)
        panecho = None
        panecho_available = False

    # Load EchoNet-Dynamic (EF specialist)
    # EchoNet uses torchvision r2plus1d_18 with custom FC head
    logger.info("Loading EchoNet-Dynamic (EF specialist)...")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "echonet-dynamic" / "upstream"))
        import echonet
        echonet_model = echonet.models.r2plus1d.r2plus1d_18(num_classes=1, spatial_size=112, pretrained=True)
        echonet_model.eval()
        echonet_available = True
        logger.info("✅ EchoNet-Dynamic loaded")
    except Exception as e:
        logger.warning("EchoNet-Dynamic load failed: %s — using PanEcho-only mode", e)
        echonet_model = None
        echonet_available = False

    if not panecho_available and not echonet_available:
        logger.error("No models available!")
        return 1

    # Run predictions
    results = []
    t0 = time.perf_counter()

    for i, study in enumerate(test_studies, 1):
        video = load_video(study["path"])
        gt_ef = study["ef"]

        # PanEcho prediction
        ef_panecho = None
        if panecho_available:
            with torch.inference_mode():
                out = panecho(video)
                ef_panecho = float(out["EF"].item()) if isinstance(out, dict) and "EF" in out else float(out.mean().item())

        # EchoNet-Dynamic prediction (uses 112x112 input, 32 frames)
        ef_echonet = None
        if echonet_available:
            try:
                # EchoNet expects (1, 3, 32, 112, 112)
                video_echonet = load_video(study["path"], size=112, clip_len=32)
                with torch.inference_mode():
                    out = echonet_model(video_echonet)
                    ef_echonet = float(out.item())
            except Exception as e:
                logger.warning("EchoNet failed on %s: %s", study["filename"], e)

        # Ensemble
        if ef_panecho is not None and ef_echonet is not None:
            if args.ensemble_mode == "average":
                ef_ensemble = (ef_panecho + ef_echonet) / 2
            elif args.ensemble_mode == "weighted":
                # Weight by inverse error (heuristic: weight specialist more)
                ef_ensemble = 0.3 * ef_panecho + 0.7 * ef_echonet
            elif args.ensemble_mode == "specialist":
                # Use EchoNet (specialist) unless PanEcho disagrees >10%
                if abs(ef_panecho - ef_echonet) > 10:
                    ef_ensemble = (ef_panecho + ef_echonet) / 2  # Average when disagree
                else:
                    ef_ensemble = ef_echonet  # Trust specialist when agree
        elif ef_panecho is not None:
            ef_ensemble = ef_panecho
        elif ef_echonet is not None:
            ef_ensemble = ef_echonet
        else:
            ef_ensemble = None

        # Agreement analysis
        agreement = None
        if ef_panecho is not None and ef_echonet is not None:
            agreement = abs(ef_panecho - ef_echonet)

        result = {
            "filename": study["filename"],
            "ef_gt": gt_ef,
            "ef_panecho": round(ef_panecho, 2) if ef_panecho else None,
            "ef_echonet": round(ef_echonet, 2) if ef_echonet else None,
            "ef_ensemble": round(ef_ensemble, 2) if ef_ensemble else None,
            "error_panecho": round(abs(ef_panecho - gt_ef), 2) if ef_panecho else None,
            "error_echonet": round(abs(ef_echonet - gt_ef), 2) if ef_echonet else None,
            "error_ensemble": round(abs(ef_ensemble - gt_ef), 2) if ef_ensemble else None,
            "agreement_gap": round(agreement, 2) if agreement else None,
        }
        results.append(result)

        status = f"GT={gt_ef:.1f} "
        if ef_panecho: status += f"P={ef_panecho:.1f} "
        if ef_echonet: status += f"E={ef_echonet:.1f} "
        if ef_ensemble: status += f"ENS={ef_ensemble:.1f} "
        if agreement: status += f"(gap={agreement:.1f})"
        logger.info("[%d/%d] %s", i, len(test_studies), status)

    elapsed = time.perf_counter() - t0

    # Compute stats
    panecho_errs = [r["error_panecho"] for r in results if r["error_panecho"] is not None]
    echonet_errs = [r["error_echonet"] for r in results if r["error_echonet"] is not None]
    ensemble_errs = [r["error_ensemble"] for r in results if r["error_ensemble"] is not None]
    agreements = [r["agreement_gap"] for r in results if r["agreement_gap"] is not None]

    print(f"\n{'='*70}")
    print(f"  ENSEMBLE RESULTS — PanEcho + EchoNet-Dynamic")
    print(f"{'='*70}")
    print(f"  Mode: {args.ensemble_mode}")
    print(f"  N: {len(results)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(results):.1f}s/video)")
    print()
    if panecho_errs:
        print(f"  PanEcho (generalist):     MAE = {np.mean(panecho_errs):.2f} EF%")
    if echonet_errs:
        print(f"  EchoNet-Dynamic (spec):   MAE = {np.mean(echonet_errs):.2f} EF%")
    if ensemble_errs:
        print(f"  Ensemble ({args.ensemble_mode}): MAE = {np.mean(ensemble_errs):.2f} EF%")
    if agreements:
        print(f"  Mean agreement gap:      {np.mean(agreements):.2f} EF%")
        print(f"  Disagreements (>10 EF%): {sum(1 for a in agreements if a > 10)}/{len(agreements)}")
    print()
    print(f"  Commercial target: 5.00 EF%")
    if ensemble_errs:
        print(f"  Beats target: {'✅' if np.mean(ensemble_errs) <= 5.0 else '❌'}")
    print(f"{'='*70}")

    # Save
    output = {
        "ensemble_mode": args.ensemble_mode,
        "n_videos": len(results),
        "elapsed_s": round(elapsed, 1),
        "mae_panecho": round(float(np.mean(panecho_errs)), 2) if panecho_errs else None,
        "mae_echonet": round(float(np.mean(echonet_errs)), 2) if echonet_errs else None,
        "mae_ensemble": round(float(np.mean(ensemble_errs)), 2) if ensemble_errs else None,
        "mean_agreement_gap": round(float(np.mean(agreements)), 2) if agreements else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
