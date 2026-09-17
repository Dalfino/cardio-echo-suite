#!/usr/bin/env python3
"""Research shakedown: run cardio-echo-suite on EchoNet-Dynamic test set
and compare AI EF predictions to ground truth EF labels.

This is for RESEARCH ONLY — not for regulatory validation.
EchoNet-Dynamic was used to train the upstream models, so this is circular.

Usage:
    python scripts/run_research_shakedown.py \\
        --data-dir /tmp/my-project/echonet-data/EchoNet-Dynamic \\
        --output /tmp/research_results.json \\
        --max-n 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("research-shakedown")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_test_labels(filelist_path: Path) -> List[Dict[str, Any]]:
    """Load EchoNet-Dynamic FileList.csv and return test split rows."""
    studies = []
    with filelist_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Split", "").strip() == "TEST":
                studies.append({
                    "filename": row["FileName"].strip() + ".avi",
                    "ef_ground_truth": float(row["EF"]),
                    "split": row["Split"].strip(),
                })
    return studies


def run_panecho_dryrun(video_path: Path, patient_id: str = "research", dry_run: bool = True) -> Dict[str, Any]:
    """Run PanEcho CLI on a single video. Returns the pre-read report."""
    import subprocess
    cmd = [
        sys.executable, "-m", "app.cli", "predict",
        str(video_path),
        "--patient-id", patient_id,
        "-o", "/tmp/_preread_tmp.json",
    ]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1] / "services" / "panecho"),
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "services" / "panecho")},
        timeout=120,
    )
    if result.returncode != 0:
        return {"error": result.stderr[:500]}
    try:
        return json.loads(Path("/tmp/_preread_tmp.json").read_text())
    except Exception as e:
        return {"error": f"Failed to read result: {e}"}


def extract_ef_from_preread(preread: Dict[str, Any]) -> float | None:
    """Extract the AI-predicted EF from a PanEcho pre-read report."""
    if "error" in preread:
        return None
    for finding in preread.get("findings", []):
        if finding.get("task") == "EF" and finding.get("type") == "regression":
            return float(finding["value"])
    return None


def compute_stats(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute MAE, Pearson r, and other stats."""
    import numpy as np

    ef_ai = [p["ef_ai"] for p in predictions if p.get("ef_ai") is not None]
    ef_gt = [p["ef_ground_truth"] for p in predictions if p.get("ef_ai") is not None]
    errors = [abs(a - g) for a, g in zip(ef_ai, ef_gt)]

    if not errors:
        return {"n": 0, "error": "no valid predictions"}

    mae = float(np.mean(errors))
    std = float(np.std(errors))

    # Pearson r
    if len(ef_ai) > 1:
        r = float(np.corrcoef(ef_ai, ef_gt)[0, 1])
    else:
        r = 0.0

    # Bland-Altman
    bias = float(np.mean([a - g for a, g in zip(ef_ai, ef_gt)]))
    loa = 1.96 * float(np.std([a - g for a, g in zip(ef_ai, ef_gt)]))

    return {
        "n": len(errors),
        "mae_ef": round(mae, 2),
        "std_ef": round(std, 2),
        "min_error": round(min(errors), 2),
        "max_error": round(max(errors), 2),
        "pearson_r": round(r, 3),
        "bland_altman_bias": round(bias, 2),
        "bland_altman_loa": round(loa, 2),
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research shakedown on EchoNet-Dynamic")
    p.add_argument("--data-dir", type=Path, required=True, help="Path to EchoNet-Dynamic/ (containing Videos/ and FileList.csv)")
    p.add_argument("--output", type=Path, default=Path("/tmp/research_results.json"))
    p.add_argument("--max-n", type=int, default=50, help="Max videos to process (default 50)")
    p.add_argument("--skip", type=int, default=0, help="Skip first N available videos (for batching)")
    p.add_argument("--real", action="store_true", help="Run real model inference (not dry-run). Downloads ~150MB weights on first run.")
    args = p.parse_args(argv)

    filelist_path = args.data_dir / "FileList.csv"
    videos_dir = args.data_dir / "Videos"

    logger.info("Loading test labels from %s", filelist_path)
    test_studies = load_test_labels(filelist_path)
    logger.info("Found %d test studies in FileList.csv", len(test_studies))

    # Filter to only videos that exist on disk
    available = [s for s in test_studies if (videos_dir / s["filename"]).exists()]
    logger.info("Available on disk: %d / %d", len(available), len(test_studies))

    if args.max_n > 0:
        available = available[args.skip:args.skip + args.max_n]
    elif args.skip > 0:
        available = available[args.skip:]
    logger.info("Will process %d videos (skip=%d)", len(available), args.skip)

    predictions: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, study in enumerate(available, 1):
        video_path = videos_dir / study["filename"]
        patient_id = f"echonet-{study['filename'].replace('.avi', '')}"

        logger.info("[%d/%d] %s (GT EF: %.1f%%)", i, len(available),
                    study["filename"], study["ef_ground_truth"])

        preread = run_panecho_dryrun(video_path, patient_id, dry_run=not args.real)
        ef_ai = extract_ef_from_preread(preread)

        predictions.append({
            "filename": study["filename"],
            "patient_id": patient_id,
            "ef_ground_truth": study["ef_ground_truth"],
            "ef_ai": ef_ai,
            "abs_error": abs(ef_ai - study["ef_ground_truth"]) if ef_ai is not None else None,
            "preread_summary": preread.get("summary", ""),
            "error": preread.get("error"),
        })

        if ef_ai is not None:
            logger.info("  → AI EF: %.1f%% (error: %.1f%%)", ef_ai, abs(ef_ai - study["ef_ground_truth"]))
        else:
            logger.warning("  → AI EF: failed — %s", preread.get("error", "unknown"))

    elapsed = time.perf_counter() - t0
    stats = compute_stats(predictions)

    result = {
        "warning": "RESEARCH ONLY — not for regulatory validation. EchoNet-Dynamic was used to train upstream models.",
        "dataset": "EchoNet-Dynamic test split",
        "n_processed": len(predictions),
        "n_success": sum(1 for p in predictions if p["ef_ai"] is not None),
        "n_failed": sum(1 for p in predictions if p["ef_ai"] is None),
        "elapsed_s": round(elapsed, 1),
        "stats": stats,
        "predictions": predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info("=" * 60)
    logger.info("✅ Done in %.1f seconds", elapsed)
    logger.info("   Processed: %d", len(predictions))
    logger.info("   Success: %d", result["n_success"])
    logger.info("   Failed: %d", result["n_failed"])
    logger.info("   EF MAE: %s EF%%", stats.get("mae_ef", "?"))
    logger.info("   Pearson r: %s", stats.get("pearson_r", "?"))
    logger.info("   Bland-Altman bias: %s (LoA: ±%s)",
                stats.get("bland_altman_bias", "?"), stats.get("bland_altman_loa", "?"))
    logger.info("   Results: %s", args.output)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
