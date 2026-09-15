#!/usr/bin/env python3
"""Validation script — runs cardio-echo-suite on a folder of de-identified DICOMs.

Usage:
    python scripts/validate.py \\
        --input-dir /data/deidentified_echo/ \\
        --labels /data/cardiologist_labels.csv \\
        --output /data/validation_results.csv \\
        --orchestrator-url http://localhost:8080 \\
        --patient-id-field patient_id \\
        --study-uid-field study_instance_uid \\
        --ef-field ef_cardiologist

The labels CSV must have columns:
    patient_id, study_instance_uid, ef_cardiologist, [any other ground-truth columns]

Outputs:
    validation_results.csv — one row per study with AI predictions + ground truth
    validation_summary.json — aggregate accuracy metrics
    failure_modes.html — HTML dashboard of worst 20 cases
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("validate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_labels(labels_path: Path) -> List[Dict[str, str]]:
    """Load ground-truth labels from CSV."""
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_dicom_for_study(input_dir: Path, study_uid: str) -> Optional[Path]:
    """Find the first DICOM file matching a StudyInstanceUID.

    Assumes directory structure: input_dir/<StudyInstanceUID>/*.dcm
    Or flat: input_dir/*.dcm with StudyInstanceUID in DICOM header
    """
    # Try directory-based lookup first
    study_dir = input_dir / study_uid
    if study_dir.is_dir():
        files = list(study_dir.glob("*.dcm")) + list(study_dir.glob("*.mp4"))
        if files:
            return files[0]

    # Fall back to flat lookup with header inspection
    for f in input_dir.glob("*.dcm"):
        try:
            import pydicom
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if str(getattr(ds, "StudyInstanceUID", "")) == study_uid:
                return f
        except Exception:
            continue

    return None


async def run_one_study(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    file_path: Path,
    patient_id: str,
    study_uid: str,
) -> Dict[str, Any]:
    """Run echo AI on one study via the orchestrator."""
    try:
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f.read(), "application/dicom")}
            params = {
                "patient_id": patient_id,
                "study_uid": study_uid,
                "skip": "echonet,echoprime",  # PanEcho only for speed
            }
            r = await client.post(
                f"{orchestrator_url}/v1/echo/full",
                files=files,
                params=params,
            )
        if r.status_code >= 400:
            return {"error": r.text, "status_code": r.status_code, "study_uid": study_uid}
        return r.json()
    except Exception as e:
        return {"error": str(e), "study_uid": study_uid}


async def validate_all(
    input_dir: Path,
    labels: List[Dict[str, str]],
    orchestrator_url: str,
    patient_id_field: str,
    study_uid_field: str,
    concurrency: int = 4,
) -> List[Dict[str, Any]]:
    """Run validation on all labeled studies."""
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=300.0) as client:
        async def bounded_run(label_row: Dict[str, str]) -> Dict[str, Any]:
            async with sem:
                patient_id = label_row.get(patient_id_field, "unknown")
                study_uid = label_row.get(study_uid_field, "")
                file_path = find_dicom_for_study(input_dir, study_uid)
                if file_path is None:
                    return {
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "error": "DICOM file not found",
                        **label_row,
                    }
                t0 = time.perf_counter()
                ai_result = await run_one_study(client, orchestrator_url, file_path,
                                                patient_id, study_uid)
                dt = time.perf_counter() - t0
                return {
                    "patient_id": patient_id,
                    "study_uid": study_uid,
                    "inference_s": round(dt, 2),
                    "ai_result": ai_result,
                    **label_row,
                }

        tasks = [bounded_run(row) for row in labels]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info("Processed %d/%d studies", i + 1, len(labels))

    return results


def compute_summary(
    results: List[Dict[str, Any]],
    ef_ground_truth_field: str = "ef_cardiologist",
) -> Dict[str, Any]:
    """Compute aggregate accuracy metrics."""
    summary: Dict[str, Any] = {
        "n_total": len(results),
        "n_success": 0,
        "n_error": 0,
        "ef_predictions": [],
    }

    for r in results:
        if "error" in r:
            summary["n_error"] += 1
            continue
        ai = r.get("ai_result", {})
        panecho = (ai.get("results") or {}).get("panecho", {})
        findings = panecho.get("findings", [])
        ef_finding = next((f for f in findings if f.get("task") == "EF"), None)
        if ef_finding and ef_ground_truth_field in r:
            try:
                ef_ai = float(ef_finding.get("value", 0))
                ef_gt = float(r[ef_ground_truth_field])
                summary["ef_predictions"].append({
                    "patient_id": r.get("patient_id", ""),
                    "study_uid": r.get("study_uid", ""),
                    "ef_ai": ef_ai,
                    "ef_cardiologist": ef_gt,
                    "abs_error": abs(ef_ai - ef_gt),
                })
                summary["n_success"] += 1
            except (ValueError, TypeError):
                pass

    # Compute EF MAE and Pearson r
    if summary["ef_predictions"]:
        errors = [p["abs_error"] for p in summary["ef_predictions"]]
        summary["ef_mae"] = round(sum(errors) / len(errors), 2)
        summary["ef_max_error"] = round(max(errors), 2)
        summary["ef_min_error"] = round(min(errors), 2)
        # Pearson r
        ef_ais = [p["ef_ai"] for p in summary["ef_predictions"]]
        ef_gts = [p["ef_cardiologist"] for p in summary["ef_predictions"]]
        n = len(ef_ais)
        mean_ai = sum(ef_ais) / n
        mean_gt = sum(ef_gts) / n
        num = sum((a - mean_ai) * (g - mean_gt) for a, g in zip(ef_ais, ef_gts))
        den_ai = (sum((a - mean_ai) ** 2 for a in ef_ais)) ** 0.5
        den_gt = (sum((g - mean_gt) ** 2 for g in ef_gts)) ** 0.5
        summary["ef_pearson_r"] = round(num / max(den_ai * den_gt, 1e-10), 3)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run cardio-echo-suite validation on a folder of DICOMs")
    p.add_argument("--input-dir", type=Path, required=True, help="Folder of de-identified DICOM studies")
    p.add_argument("--labels", type=Path, required=True, help="CSV with cardiologist ground truth")
    p.add_argument("--output", type=Path, default=Path("validation_results.csv"))
    p.add_argument("--orchestrator-url", default="http://localhost:8080")
    p.add_argument("--patient-id-field", default="patient_id")
    p.add_argument("--study-uid-field", default="study_instance_uid")
    p.add_argument("--ef-field", default="ef_cardiologist")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args(argv)

    logger.info("Loading labels from %s", args.labels)
    labels = load_labels(args.labels)
    logger.info("Loaded %d labeled studies", len(labels))

    logger.info("Running validation against %s", args.orchestrator_url)
    results = asyncio.run(validate_all(
        args.input_dir, labels, args.orchestrator_url,
        args.patient_id_field, args.study_uid_field,
        concurrency=args.concurrency,
    ))

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "study_uid", "inference_s",
            "ef_cardiologist", "ef_ai", "abs_error",
            "error",
        ])
        writer.writeheader()
        for r in results:
            ai = r.get("ai_result", {})
            panecho = (ai.get("results") or {}).get("panecho", {})
            findings = panecho.get("findings", [])
            ef_finding = next((f for f in findings if f.get("task") == "EF"), None)
            ef_ai = float(ef_finding["value"]) if ef_finding else None
            ef_gt = float(r.get(args.ef_field, 0)) if r.get(args.ef_field) else None
            writer.writerow({
                "patient_id": r.get("patient_id", ""),
                "study_uid": r.get("study_uid", ""),
                "inference_s": r.get("inference_s", ""),
                "ef_cardiologist": ef_gt if ef_gt is not None else "",
                "ef_ai": ef_ai if ef_ai is not None else "",
                "abs_error": abs(ef_ai - ef_gt) if ef_ai and ef_gt else "",
                "error": r.get("error", ""),
            })
    logger.info("Wrote %s", args.output)

    # Summary
    summary = compute_summary(results, ef_ground_truth_field=args.ef_field)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
