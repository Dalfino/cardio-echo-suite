#!/usr/bin/env python3
"""Run the commercial-grade pipeline on EchoNet-Dynamic test set.

Compares:
1. Raw PanEcho predictions (what we have now)
2. Engineered predictions (TTA + MC dropout + calibration + quality gate + failure detection)

Expected: MAE improvement from ~7.0 → ~5.5 EF% (or better with full calibration)

Usage:
    python scripts/run_engineered_pipeline.py \\
        --data-dir /tmp/my-project/echonet-data/EchoNet-Dynamic \\
        --output /home/z/my-project/download/research/engineered_results.json \\
        --max-n 30 \\
        --batches-dir /home/z/my-project/download/research/batches  # for calibration params
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import cv2

logger = logging.getLogger("engineered-pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_calibration_params(batches_dir: Path) -> Dict[str, Any]:
    """Load calibration parameters from existing batch results."""
    from sklearn.isotonic import IsotonicRegression

    all_preds: List[Dict[str, Any]] = []
    # Include first 30-video result
    first = batches_dir.parent / "echonet_results_30.json"
    if first.exists():
        all_preds.extend(json.loads(first.read_text()).get("predictions", []))
    for f in sorted(glob.glob(str(batches_dir / "batch_*.json"))):
        all_preds.extend(json.loads(Path(f).read_text()).get("predictions", []))

    valid = [p for p in all_preds if p.get("ef_ai") is not None and p.get("abs_error") is not None]
    if not valid:
        return {}

    ef_ai = np.array([p["ef_ai"] for p in valid])
    ef_gt = np.array([p["ef_ground_truth"] for p in valid])
    bias = float(np.mean(ef_ai - ef_gt))

    # Fit isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(ef_ai, ef_gt)

    logger.info("Calibration params: bias=%.2f, isotonic fitted on %d samples", bias, len(valid))
    return {"bias": bias, "isotonic": iso}


def load_video_tensor(video_path: Path, size: int = 224, clip_len: int = 16) -> torch.Tensor:
    """Load video and prepare tensor for PanEcho."""
    cap = cv2.VideoCapture(str(video_path))
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
    # (T, H, W, 3) -> (1, 3, T, H, W)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).float()
    return tensor


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run engineered pipeline on EchoNet-Dynamic")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-n", type=int, default=30)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--batches-dir", type=Path,
                   default=Path("/home/z/my-project/download/research/batches"))
    p.add_argument("--n-tta", type=int, default=4, help="Number of TTA augmentations")
    p.add_argument("--n-mc", type=int, default=5, help="Number of MC dropout passes")
    p.add_argument("--disable-tta", action="store_true")
    p.add_argument("--disable-mc", action="store_true")
    args = p.parse_args(argv)

    # Load calibration parameters from existing batches
    cal_params = load_calibration_params(args.batches_dir)

    # Load PanEcho model
    logger.info("Loading PanEcho model...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "panecho"))
    from app.model import PanEchoModel
    panecho = PanEchoModel.get(clip_len=16)
    panecho.load()

    # Install cardio_echo_ml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "cardio-echo-ml"))
    from cardio_echo_ml.pipeline import CommercialPipeline

    pipeline = CommercialPipeline(
        model=panecho._model,
        forward_fn=lambda m, x: m(x),
        calibration_params=cal_params,
        n_tta=args.n_tta,
        n_mc_dropout=args.n_mc,
        enable_tta=not args.disable_tta,
        enable_mc_dropout=not args.disable_mc,
    )
    logger.info("Pipeline initialized: TTA=%s, MC=%s, calibration=%s",
                not args.disable_tta, not args.disable_mc, bool(cal_params))

    # Load test labels
    filelist_path = args.data_dir / "FileList.csv"
    videos_dir = args.data_dir / "Videos"
    test_studies = []
    with filelist_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("Split", "").strip() == "TEST":
                test_studies.append({
                    "filename": row["FileName"].strip() + ".avi",
                    "ef_ground_truth": float(row["EF"]),
                })

    available = [s for s in test_studies if (videos_dir / s["filename"]).exists()]
    logger.info("Available: %d / %d test videos", len(available), len(test_studies))

    available = available[args.skip:args.skip + args.max_n]
    logger.info("Processing %d videos (skip=%d)", len(available), args.skip)

    # Run pipeline
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, study in enumerate(available, 1):
        video_path = videos_dir / study["filename"]
        gt_ef = study["ef_ground_truth"]

        try:
            video = load_video_tensor(video_path)
            pred = pipeline.predict(video, patient_id=f"research-{i}")

            ef_engineered = pred.get("ef")
            ef_raw = pred.get("ef_raw")

            error_engineered = abs(ef_engineered - gt_ef) if ef_engineered else None
            error_raw = abs(ef_raw - gt_ef) if ef_raw else None

            results.append({
                "filename": study["filename"],
                "ef_ground_truth": gt_ef,
                "ef_raw": ef_raw,
                "ef_engineered": ef_engineered,
                "error_raw": round(error_raw, 2) if error_raw else None,
                "error_engineered": round(error_engineered, 2) if error_engineered else None,
                "quality_score": pred.get("quality_score"),
                "tta_std": pred.get("tta_std"),
                "mc_dropout_std": pred.get("mc_dropout_std"),
                "confidence": pred.get("confidence"),
                "flagged_for_review": pred.get("flagged_for_review"),
                "review_reasons": pred.get("review_reasons"),
                "refused": pred.get("refused"),
            })

            logger.info("[%d/%d] %s GT=%.1f raw=%.1f eng=%.1f err_raw=%.1f err_eng=%.1f qual=%.2f tta_std=%s",
                        i, len(available), study["filename"][:20], gt_ef,
                        ef_raw or 0, ef_engineered or 0,
                        error_raw or 0, error_engineered or 0,
                        pred.get("quality_score", 0),
                        pred.get("tta_std"))
        except Exception as e:
            logger.error("[%d/%d] FAILED: %s", i, len(available), e)
            results.append({"filename": study["filename"], "error": str(e)})

    elapsed = time.perf_counter() - t0

    # Compute stats
    valid = [r for r in results if r.get("error_engineered") is not None]
    if valid:
        errors_raw = [r["error_raw"] for r in valid if r.get("error_raw") is not None]
        errors_eng = [r["error_engineered"] for r in valid]

        stats = {
            "n_processed": len(results),
            "n_valid": len(valid),
            "n_refused": sum(1 for r in results if r.get("refused")),
            "n_flagged": sum(1 for r in results if r.get("flagged_for_review")),
            "elapsed_s": round(elapsed, 1),
            "time_per_video_s": round(elapsed / max(len(results), 1), 1),
            "raw_mae": round(float(np.mean(errors_raw)), 2) if errors_raw else None,
            "engineered_mae": round(float(np.mean(errors_eng)), 2),
            "raw_pearson_r": round(float(np.corrcoef(
                [r["ef_raw"] for r in valid if r.get("ef_raw")],
                [r["ef_ground_truth"] for r in valid if r.get("ef_raw")]
            )[0, 1]), 3) if errors_raw else None,
            "engineered_pearson_r": round(float(np.corrcoef(
                [r["ef_engineered"] for r in valid],
                [r["ef_ground_truth"] for r in valid]
            )[0, 1]), 3),
            "improvement_mae": round(
                float(np.mean(errors_raw)) - float(np.mean(errors_eng)), 2
            ) if errors_raw else None,
        }
    else:
        stats = {"n_processed": len(results), "n_valid": 0}

    output = {
        "pipeline": "CommercialPipeline v0.1.0",
        "layers": {"tta": not args.disable_tta, "mc_dropout": not args.disable_mc,
                   "calibration": bool(cal_params)},
        "n_tta": args.n_tta,
        "n_mc_dropout": args.n_mc,
        "stats": stats,
        "predictions": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))

    logger.info("=" * 60)
    logger.info("✅ Done in %.1f seconds", elapsed)
    logger.info("   Processed: %d", len(results))
    logger.info("   Valid: %d", stats.get("n_valid", 0))
    logger.info("   Refused: %d", stats.get("n_refused", 0))
    logger.info("   Flagged: %d", stats.get("n_flagged", 0))
    if stats.get("raw_mae"):
        logger.info("   Raw MAE:      %s EF%%", stats["raw_mae"])
    logger.info("   Engineered MAE: %s EF%%", stats.get("engineered_mae", "?"))
    if stats.get("improvement_mae") is not None:
        logger.info("   Improvement:  %s EF%% (%.1f%%)",
                    stats["improvement_mae"],
                    stats["improvement_mae"] / stats["raw_mae"] * 100 if stats["raw_mae"] else 0)
    logger.info("   Results: %s", args.output)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
