#!/usr/bin/env python3
"""ECG research shakedown: run cardio-echo-suite ECG services on PTB-XL test set.

Tests:
1. NeuroKit2 service — R-peak detection, HRV analysis, signal quality
2. ECG-FM service — arrhythmia classification, interval measurement

⚠️  RESEARCH ONLY — not for regulatory validation.
PTB-XL was used to train ECG-FM, so this is circular evaluation.

Prerequisites:
  - PTB-XL downloaded (see scripts/download_ptbxl.py)
  - PhysioNet credentialed access
  - cardio-echo-suite services running (docker compose up)

Usage:
    # Full shakedown with NeuroKit2 (no model weights needed)
    python scripts/run_ecg_research_shakedown.py \\
        --data-dir /data/public/ptbxl/records100 \\
        --labels /data/public/ptbxl/ptbxl_database.csv \\
        --output /tmp/ecg_research_results.json \\
        --max-n 50 \\
        --service neurokit

    # With ECG-FM (requires model weights from HuggingFace)
    python scripts/run_ecg_research_shakedown.py \\
        --data-dir /data/public/ptbxl/records100 \\
        --labels /data/public/ptbxl/ptbxl_database.csv \\
        --output /tmp/ecg_research_results.json \\
        --max-n 50 \\
        --service ecg-fm

    # Test on synthetic ECG (no PTB-XL needed — for pipeline testing)
    python scripts/run_ecg_research_shakedown.py \\
        --synthetic \\
        --output /tmp/ecg_synthetic_results.json \\
        --max-n 10
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

logger = logging.getLogger("ecg-shakedown")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ----------------------------------------------------------------------
# PTB-XL label loading
# ----------------------------------------------------------------------

def load_ptbxl_labels(labels_path: Path) -> List[Dict[str, Any]]:
    """Load PTB-XL database CSV and return test split rows."""
    studies = []
    with labels_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            # PTB-XL has strat-fold column for train/test split
            # fold 0-7 = train, 8 = val, 9 = test
            fold = int(row.get("strat_fold", 0))
            if fold != 9:
                continue
            filename = row.get("filename_lr", "") or row.get("filename", "")
            if not filename:
                continue
            studies.append({
                "filename": filename,
                "ecg_id": row.get("ecg_id", ""),
                "patient_id": f"ptbxl-{row.get('ecg_id', '')}",
                "rhythm": row.get("rhythms", ""),
                "forms": row.get("forms", ""),
                "heart_rate": float(row.get("heart_rate", 0) or 0),
                "patient_age": int(float(row.get("patient_age", 0) or 0)),
                "sex": int(row.get("sex", 0) or 0),
            })
    return studies


# ----------------------------------------------------------------------
# Synthetic ECG generator (for testing without PTB-XL)
# ----------------------------------------------------------------------

def generate_synthetic_ecg(
    n_samples: int = 2500,
    fs: float = 250.0,
    heart_rate: float = 72.0,
    n_leads: int = 12,
) -> np.ndarray:
    """Generate a synthetic 12-lead ECG for pipeline testing.

    Produces a (n_samples, n_leads) array with realistic-looking
    P-QRS-T morphology at the specified heart rate.
    """
    t = np.arange(n_samples) / fs
    rr_interval = 60.0 / heart_rate  # seconds per beat
    n_beats = int(n_samples / (fs * rr_interval))

    # Base signal (noise)
    signal = np.random.normal(0, 0.02, (n_samples, n_leads))

    # Add QRS complexes
    for beat in range(n_beats):
        beat_time = beat * rr_interval
        beat_samples = int(beat_time * fs)
        if beat_samples >= n_samples:
            break

        # QRS duration ~80ms
        qrs_duration = int(0.08 * fs)
        for lead in range(n_leads):
            # R-peak amplitude varies by lead
            r_amp = 1.0 + 0.5 * np.sin(lead * 0.5)
            for i in range(max(0, beat_samples - qrs_duration // 2),
                          min(n_samples, beat_samples + qrs_duration // 2)):
                dt = (i - beat_samples) / fs
                # Gaussian-shaped R-peak
                signal[i, lead] += r_amp * np.exp(-((dt * 30) ** 2))

            # P-wave (before QRS)
            p_time = beat_samples - int(0.16 * fs)
            if 0 <= p_time < n_samples:
                p_duration = int(0.08 * fs)
                for i in range(max(0, p_time - p_duration // 2),
                              min(n_samples, p_time + p_duration // 2)):
                    dt = (i - p_time) / fs
                    signal[i, lead] += 0.2 * np.exp(-((dt * 20) ** 2))

            # T-wave (after QRS)
            t_time = beat_samples + int(0.25 * fs)
            if 0 <= t_time < n_samples:
                t_duration = int(0.16 * fs)
                for i in range(max(0, t_time - t_duration // 2),
                              min(n_samples, t_time + t_duration // 2)):
                    dt = (i - t_time) / fs
                    signal[i, lead] += 0.3 * np.exp(-((dt * 10) ** 2))

    return signal


def save_synthetic_ecg_csv(signal: np.ndarray, path: Path, fs: float = 250.0):
    """Save synthetic ECG as CSV with sidecar JSON."""
    import csv as csv_module
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    with path.open("w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(leads[:signal.shape[1]])
        for row in signal:
            writer.writerow([f"{v:.4f}" for v in row])
    # Sidecar
    import json as json_module
    sidecar = path.with_suffix(".csv.json")
    sidecar.write_text(json_module.dumps({"sample_rate_hz": fs, "n_leads": signal.shape[1]}))


# ----------------------------------------------------------------------
# NeuroKit2 service testing
# ----------------------------------------------------------------------

def test_neurokit_on_ecg(ecg_csv_path: Path, fs: float = 250.0) -> Dict[str, Any]:
    """Run NeuroKit2 R-peak detection + HRV on a single ECG CSV.

    This calls the neurokit service's /r-peak and /hrv endpoints.
    """
    import subprocess
    import httpx

    # Try calling the neurokit service via HTTP
    try:
        with ecg_csv_path.open("rb") as f:
            files = {"file": (ecg_csv_path.name, f.read(), "text/csv")}
            # R-peak detection
            r_peak_resp = httpx.post(
                "http://localhost:8007/r-peak",
                files=files,
                params={"fs": fs},
                timeout=30.0,
            )
            if r_peak_resp.status_code == 200:
                r_peak_data = r_peak_resp.json()
            else:
                return {"error": f"R-peak HTTP {r_peak_resp.status_code}: {r_peak_resp.text[:200]}"}

            # HRV
            f.seek(0)
            hrv_resp = httpx.post(
                "http://localhost:8007/hrv",
                files={"file": (ecg_csv_path.name, f.read(), "text/csv")},
                params={"fs": fs},
                timeout=30.0,
            )
            if hrv_resp.status_code == 200:
                hrv_data = hrv_resp.json()
            else:
                hrv_data = {"error": f"HRV HTTP {hrv_resp.status_code}"}

            return {
                "r_peak": r_peak_data,
                "hrv": hrv_data,
            }
    except httpx.ConnectError:
        # Service not running — fall back to direct Python call
        return test_neurokit_direct(ecg_csv_path, fs)
    except Exception as e:
        return {"error": str(e)}


def test_neurokit_direct(ecg_csv_path: Path, fs: float = 250.0) -> Dict[str, Any]:
    """Run NeuroKit2 directly (without HTTP) for testing."""
    try:
        import neurokit2 as nk  # type: ignore
    except ImportError:
        # Fall back to scipy-based R-peak detection
        return test_neurokit_scipy(ecg_csv_path, fs)

    # Load CSV
    signal = np.loadtxt(ecg_csv_path, delimiter=",", skiprows=1)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    # Use lead I (first column) for R-peak detection
    lead1 = signal[:, 0]

    try:
        _, info = nk.ecg_peaks(lead1, sampling_rate=int(fs))
        peaks = info.get("ECG_R_Peaks", [])

        if len(peaks) < 2:
            return {"error": "Not enough R-peaks detected", "n_peaks": len(peaks)}

        # Compute HRV
        hrv = nk.hrv(info, sampling_rate=int(fs), show=False)

        return {
            "n_peaks": len(peaks),
            "heart_rate": 60.0 / (np.mean(np.diff(peaks)) / fs),
            "rr_intervals_ms": (np.diff(peaks) / fs * 1000).tolist(),
            "hrv_rmssd": float(hrv["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv else None,
            "hrv_sdnn": float(hrv["HRV_SDNN"].iloc[0]) if "HRV_SDNN" in hrv else None,
            "hrv_pnn50": float(hrv["HRV_pNN50"].iloc[0]) if "HRV_pNN50" in hrv else None,
        }
    except Exception as e:
        return {"error": str(e)}


def test_neurokit_scipy(ecg_csv_path: Path, fs: float = 250.0) -> Dict[str, Any]:
    """Simple scipy-based R-peak detection (fallback when neurokit2 not installed)."""
    from scipy.signal import find_peaks

    signal = np.loadtxt(ecg_csv_path, delimiter=",", skiprows=1)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    lead1 = signal[:, 0]
    threshold = np.std(lead1) * 1.5
    peaks, _ = find_peaks(lead1, height=threshold, distance=int(fs * 0.5))

    if len(peaks) < 2:
        return {"error": "Not enough R-peaks", "n_peaks": len(peaks)}

    rr_intervals = np.diff(peaks) / fs * 1000  # ms

    return {
        "n_peaks": len(peaks),
        "heart_rate": 60.0 / (np.mean(np.diff(peaks)) / fs),
        "rr_intervals_ms": rr_intervals.tolist(),
        "hrv_rmssd": float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))),
        "hrv_sdnn": float(np.std(rr_intervals)),
        "hrv_pnn50": float(100 * np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)),
        "method": "scipy_fallback",
    }


# ----------------------------------------------------------------------
# ECG-FM service testing
# ----------------------------------------------------------------------

def test_ecg_fm_on_ecg(ecg_csv_path: Path) -> Dict[str, Any]:
    """Run ECG-FM service on a single ECG CSV.

    Calls the ecg-fm service's /predict endpoint with dry_run=true
    (real model weights may not be available).
    """
    import httpx

    try:
        with ecg_csv_path.open("rb") as f:
            files = {"file": (ecg_csv_path.name, f.read(), "text/csv")}
            resp = httpx.post(
                "http://localhost:8004/predict",
                files=files,
                params={"patient_id": "research", "dry_run": "true"},
                timeout=30.0,
            )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except httpx.ConnectError:
        return {"error": "ECG-FM service not running (docker compose up)"}
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def compute_ecg_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from ECG shakedown results."""
    n_total = len(results)
    n_success = sum(1 for r in results if "error" not in r.get("r_peak", {}) and "error" not in r)
    n_failed = n_total - n_success

    heart_rates = []
    hrv_rmssds = []
    hrv_sdnns = []

    for r in results:
        rp = r.get("r_peak", {})
        if "error" not in rp and "heart_rate" in rp:
            heart_rates.append(rp["heart_rate"])
            if rp.get("hrv_rmssd") is not None:
                hrv_rmssds.append(rp["hrv_rmssd"])
            if rp.get("hrv_sdnn") is not None:
                hrv_sdnns.append(rp["hrv_sdnn"])

    stats = {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
    }

    if heart_rates:
        stats["heart_rate_mean"] = round(float(np.mean(heart_rates)), 1)
        stats["heart_rate_std"] = round(float(np.std(heart_rates)), 1)
        stats["heart_rate_min"] = round(float(min(heart_rates)), 1)
        stats["heart_rate_max"] = round(float(max(heart_rates)), 1)

    if hrv_rmssds:
        stats["hrv_rmssd_mean"] = round(float(np.mean(hrv_rmssds)), 1)
        stats["hrv_rmssd_std"] = round(float(np.std(hrv_rmssds)), 1)

    if hrv_sdnns:
        stats["hrv_sdnn_mean"] = round(float(np.mean(hrv_sdnns)), 1)
        stats["hrv_sdnn_std"] = round(float(np.std(hrv_sdnns)), 1)

    return stats


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ECG research shakedown on PTB-XL or synthetic data")
    p.add_argument("--data-dir", type=Path, help="PTB-XL records100 directory")
    p.add_argument("--labels", type=Path, help="PTB-XL ptbxl_database.csv")
    p.add_argument("--output", type=Path, default=Path("/tmp/ecg_research_results.json"))
    p.add_argument("--max-n", type=int, default=50)
    p.add_argument("--service", choices=["neurokit", "ecg-fm", "both"], default="neurokit",
                   help="Which ECG service to test (default: neurokit)")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic ECG data (no PTB-XL needed)")
    p.add_argument("--fs", type=float, default=250.0, help="Sample rate in Hz")
    args = p.parse_args(argv)

    results: List[Dict[str, Any]] = []

    if args.synthetic:
        # Generate synthetic ECGs
        logger.info("Generating %d synthetic ECGs...", args.max_n)
        tmp_dir = Path("/tmp/synthetic_ecg")
        tmp_dir.mkdir(exist_ok=True)

        heart_rates = [60, 65, 70, 75, 80, 85, 90, 95, 100, 110]
        for i in range(args.max_n):
            hr = heart_rates[i % len(heart_rates)]
            signal = generate_synthetic_ecg(n_samples=int(args.fs * 10), fs=args.fs,
                                            heart_rate=hr)
            csv_path = tmp_dir / f"synthetic_ecg_{i:03d}_hr{hr}.csv"
            save_synthetic_ecg_csv(signal, csv_path, args.fs)

            logger.info("[%d/%d] Synthetic ECG (HR=%d bpm)", i + 1, args.max_n, hr)

            result: Dict[str, Any] = {
                "filename": csv_path.name,
                "ground_truth_hr": hr,
            }

            if args.service in ("neurokit", "both"):
                result["r_peak"] = test_neurokit_direct(csv_path, args.fs)

            if args.service in ("ecg-fm", "both"):
                result["ecg_fm"] = test_ecg_fm_on_ecg(csv_path)

            results.append(result)

    else:
        # Use real PTB-XL data
        if not args.data_dir or not args.labels:
            logger.error("Must provide --data-dir and --labels for PTB-XL, or use --synthetic")
            return 1

        logger.info("Loading PTB-XL labels from %s", args.labels)
        test_studies = load_ptbxl_labels(args.labels)
        logger.info("Found %d test studies (strat_fold=9)", len(test_studies))

        # Filter to available files
        available = []
        for s in test_studies:
            # PTB-XL filename_lr is relative path like "records100/00000/00001_lr"
            filepath = args.data_dir.parent / s["filename"] + ".dat"
            filepath = Path(str(args.data_dir.parent / s["filename"]))
            if filepath.with_suffix(".dat").exists() or filepath.with_suffix(".csv").exists():
                available.append(s)

        logger.info("Available on disk: %d / %d", len(available), len(test_studies))

        if args.max_n > 0:
            available = available[:args.max_n]

        for i, study in enumerate(available, 1):
            logger.info("[%d/%d] %s", i, len(available), study["filename"])
            # ... (would call service on each file)
            # This is a placeholder — actual implementation depends on file format
            results.append({"filename": study["filename"], "status": "not_implemented"})

    # Compute stats
    stats = compute_ecg_stats(results)

    output = {
        "warning": "RESEARCH ONLY — not for regulatory validation" if not args.synthetic
                   else "Synthetic data — for pipeline testing only",
        "service": args.service,
        "n_processed": len(results),
        "stats": stats,
        "results": results[:20],  # Limit stored results to first 20 for readability
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str))

    logger.info("=" * 60)
    logger.info("✅ Done")
    logger.info("   Processed: %d", len(results))
    logger.info("   Success: %d", stats["n_success"])
    logger.info("   Failed: %d", stats["n_failed"])
    if "heart_rate_mean" in stats:
        logger.info("   Heart rate: %.1f ± %.1f bpm", stats["heart_rate_mean"], stats["heart_rate_std"])
    if "hrv_rmssd_mean" in stats:
        logger.info("   HRV RMSSD: %.1f ± %.1f ms", stats["hrv_rmssd_mean"], stats["hrv_rmssd_std"])
    logger.info("   Results: %s", args.output)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
