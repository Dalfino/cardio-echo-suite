#!/usr/bin/env python3
"""Public data shakedown script for cardio-echo-suite.

⚠️  WARNING — NOT FOR REGULATORY VALIDATION  ⚠️

This script runs cardio-echo-suite against PUBLIC datasets to:
- Verify the pipeline works end-to-end (smoke test)
- Fit calibration (temperature scaling) using labels (no hospital data needed)
- Discover edge cases and failure modes before real validation

THIS IS NOT A SUBSTITUTE FOR LOCAL CLINICAL VALIDATION. Public datasets
(EchoNet-Dynamic, PTB-XL, etc.) were used to TRAIN the upstream models.
"Validation" on training data is circular and will be rejected by FDA.

Public datasets are licensed for RESEARCH USE ONLY. Use of these datasets
in a COMMERCIAL PRODUCT is prohibited by their Data Use Agreements.

Usage:
    # Echo shakedown on EchoNet-Dynamic public test set
    python scripts/public_data_shakedown.py \\
        --dataset echonet-dynamic \\
        --data-dir /data/public/echonet-dynamic/test \\
        --labels /data/public/echonet-dynamic/FileList.csv \\
        --orchestrator-url http://localhost:8080 \\
        --output /tmp/shakedown_echo.json

    # ECG shakedown on PTB-XL
    python scripts/public_data_shakedown.py \\
        --dataset ptbxl \\
        --data-dir /data/public/ptbxl/records100 \\
        --labels /data/public/ptbxl/ptbxl_database.csv \\
        --orchestrator-url http://localhost:8080 \\
        --output /tmp/shakedown_ecg.json
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

logger = logging.getLogger("shakedown")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Big warning banner
SHAKEDOWN_WARNING = """
==============================================================
⚠️  PUBLIC DATA SHAKEDOWN — NOT FOR REGULATORY VALIDATION  ⚠️
==============================================================
This script uses PUBLIC datasets that were used to TRAIN the
upstream AI models. Results from this script CANNOT be used
for:
  - FDA 510(k) submission
  - Accuracy claims in marketing
  - Subgroup equity analysis (no demographics)
  - Clinical decision support claims

Use this script ONLY for:
  - Smoke testing the pipeline
  - Fitting calibration (temperature scaling)
  - Discovering edge cases / failure modes
  - Pre-validation shakedown

For real validation, see scripts/validate.py and
docs/irb/PROTOCOL_TEMPLATE.md.
==============================================================
"""


def load_echonet_labels(path: Path) -> List[Dict[str, str]]:
    """Load EchoNet-Dynamic FileList.csv."""
    studies = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            studies.append({
                "filename": row.get("FileName", "") + ".avi",
                "split": row.get("Split", ""),
                "ef": row.get("EF", ""),
                "patient_id": f"echonet-{row.get('FileName', '')}",
                "study_uid": "",
            })
    return [s for s in studies if s["split"] == "TEST"]


def load_ptbxl_labels(path: Path) -> List[Dict[str, str]]:
    """Load PTB-XL ptbxl_database.csv."""
    studies = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            studies.append({
                "filename": row.get("filename_lr", "") + ".dat",
                "patient_id": f"ptbxl-{row.get('ecg_id', '')}",
                "study_uid": "",
                "rhythm": row.get("rhythms", ""),
                "form": row.get("forms", ""),
            })
    return studies


async def shakedown_one(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    file_path: Path,
    patient_id: str,
    endpoint: str,
) -> Dict[str, Any]:
    """Run one shakedown prediction."""
    try:
        with file_path.open("rb") as f:
            content = f.read()
        files = {"file": (file_path.name, content)}
        params = {"patient_id": patient_id, "dry_run": "false"}
        r = await client.post(
            f"{orchestrator_url}{endpoint}",
            files=files,
            params=params,
            timeout=120.0,
        )
        if r.status_code >= 400:
            return {"error": r.text[:200], "status": r.status_code, "file": str(file_path)}
        return r.json()
    except Exception as e:
        return {"error": str(e), "file": str(file_path)}


async def shakedown_run(
    data_dir: Path,
    labels: List[Dict[str, str]],
    orchestrator_url: str,
    endpoint: str,
    max_n: int = 50,
    concurrency: int = 4,
) -> List[Dict[str, Any]]:
    """Run shakedown on up to max_n studies."""
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        async def bounded_run(label: Dict[str, str]) -> Dict[str, Any]:
            async with sem:
                file_path = data_dir / label["filename"]
                if not file_path.exists():
                    return {**label, "error": "file not found"}
                t0 = time.perf_counter()
                ai = await shakedown_one(client, orchestrator_url, file_path,
                                         label["patient_id"], endpoint)
                dt = time.perf_counter() - t0
                return {**label, "ai_result": ai, "inference_s": round(dt, 2)}

        tasks = [bounded_run(l) for l in labels[:max_n]]
        for i, task in enumerate(asyncio.as_completed(tasks)):
            r = await task
            results.append(r)
            if (i + 1) % 5 == 0:
                logger.info("Shakedown: %d/%d done", i + 1, min(max_n, len(labels)))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    print(SHAKEDOWN_WARNING)
    time.sleep(2)  # Force the user to read the warning

    p = argparse.ArgumentParser(description="Public data shakedown (NOT for validation)")
    p.add_argument("--dataset", choices=["echonet-dynamic", "ptbxl"], required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--orchestrator-url", default="http://localhost:8080")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-n", type=int, default=50, help="Max studies to process (default 50)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--i-understand-this-is-not-validation", action="store_true",
                   help="Required flag to confirm you understand the limitations")
    args = p.parse_args(argv)

    if not args.i_understand_this_is_not_validation:
        print("ERROR: You must pass --i-understand-this-is-not-validation to confirm you")
        print("understand that this script's output CANNOT be used for FDA validation.")
        return 1

    # Load labels
    if args.dataset == "echonet-dynamic":
        labels = load_echonet_labels(args.labels)
        endpoint = "/v1/echo/full"
        # Skip echonet + echoprime for speed (PanEcho dry-run only)
        params_extra = {"skip": "echonet,echoprime", "dry_run": "true"}
    else:  # ptbxl
        labels = load_ptbxl_labels(args.labels)
        endpoint = "/v1/ecg/full"
        params_extra = {"dry_run": "true"}

    logger.info("Loaded %d labels from %s", len(labels), args.labels)
    logger.info("Running shakedown on first %d (max)", args.max_n)

    # Run
    async def run():
        return await shakedown_run(args.data_dir, labels, args.orchestrator_url,
                                    endpoint, args.max_n, args.concurrency)
    results = asyncio.run(run())

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "warning": "NOT FOR REGULATORY VALIDATION — see script header",
        "dataset": args.dataset,
        "n_processed": len(results),
        "n_success": sum(1 for r in results if "error" not in r),
        "n_error": sum(1 for r in results if "error" in r),
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Wrote %s", args.output)
    logger.info("Success: %d, Error: %d", output["n_success"], output["n_error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
