"""FastAPI app for NeuroKit2 utility service."""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("neurokit")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NeuroKit2 utility service",
    description="ECG signal processing — HRV, R-peak detection, signal quality. Wraps neuropsychology/NeuroKit (MIT).",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _load_signal_from_csv(path: Path, fs: float = 250.0) -> tuple[np.ndarray, float]:
    """Load a single-lead ECG from CSV. Returns (signal, sample_rate)."""
    import csv
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Detect header
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    if rows and not all(is_number(c) for c in rows[0] if c):
        rows = rows[1:]
    # Take first column as signal
    signal = np.array([float(row[0]) for row in rows if row], dtype=np.float32)
    # Check sidecar for sample rate
    sidecar = Path(str(path) + ".json")
    if sidecar.exists():
        import json
        with sidecar.open() as f:
            meta = json.load(f)
            fs = float(meta.get("sample_rate_hz", fs))
    return signal, fs


class HRVResult(BaseModel):
    heart_rate_bpm: float
    n_nni: int
    mean_nni_ms: float
    sdnn_ms: float  # standard deviation of NN intervals
    rmssd_ms: float  # root mean square of successive differences
    pnn50_percent: float  # percentage of successive NN intervals > 50ms
    lf_ms2: Optional[float]  # low-frequency power (0.04-0.15 Hz)
    hf_ms2: Optional[float]  # high-frequency power (0.15-0.4 Hz)
    lf_hf_ratio: Optional[float]
    inference_ms: float
    fhir: dict


class RPeakResult(BaseModel):
    n_peaks: int
    peak_indices: List[int]
    peak_times_ms: List[float]
    rr_intervals_ms: List[float]
    inference_ms: float


class QualityResult(BaseModel):
    overall_quality: float  # 0-1
    baseline_wander: float
    powerline_interference: float
    signal_to_noise_db: float
    inference_ms: float


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        import neurokit2  # type: ignore # noqa
        return {"status": "ready", "library": "neurokit2"}
    except ImportError:
        return {"status": "degraded", "library": "neurokit2 not installed"}


@app.post("/r-peak", response_model=RPeakResult)
async def detect_r_peaks(
    file: UploadFile = File(...),
    fs: float = Query(250.0, description="Sample rate in Hz"),
):
    """Detect R-peaks in an ECG signal."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        signal, detected_fs = _load_signal_from_csv(tmp_path, fs)
    except Exception as e:
        raise HTTPException(422, f"Could not parse ECG: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    try:
        import neurokit2 as nk  # type: ignore
        _, info = nk.ecg_peaks(signal, sampling_rate=int(detected_fs))
        peaks = info.get("ECG_R_Peaks", [])
    except ImportError:
        # Fallback: simple threshold-based peak detection
        from scipy.signal import find_peaks  # type: ignore
        threshold = signal.std() * 1.5
        peaks, _ = find_peaks(signal, height=threshold, distance=int(detected_fs * 0.5))
        peaks = peaks.tolist()
    except Exception as e:
        raise HTTPException(500, f"R-peak detection failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    peak_times_ms = [int(p / detected_fs * 1000) for p in peaks]
    rr_intervals_ms = [peak_times_ms[i+1] - peak_times_ms[i]
                       for i in range(len(peak_times_ms) - 1)]
    return RPeakResult(
        n_peaks=len(peaks),
        peak_indices=peaks,
        peak_times_ms=peak_times_ms,
        rr_intervals_ms=rr_intervals_ms,
        inference_ms=round(dt_ms, 1),
    )


@app.post("/hrv", response_model=HRVResult)
async def compute_hrv(
    file: UploadFile = File(...),
    fs: float = Query(250.0),
):
    """Compute Heart Rate Variability metrics from an ECG signal."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        signal, detected_fs = _load_signal_from_csv(tmp_path, fs)
    except Exception as e:
        raise HTTPException(422, f"Could not parse ECG: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    try:
        import neurokit2 as nk  # type: ignore
        _, info = nk.ecg_peaks(signal, sampling_rate=int(detected_fs))
        nni = nk.ecg_intervalrelated(info, sampling_rate=int(detected_fs))
        # nni is a DataFrame; extract values
        hrv = nk.hrv(info, sampling_rate=int(detected_fs), show=False)

        heart_rate = 60.0 / (np.mean(np.diff(info["ECG_R_Peaks"]) / detected_fs) or 1.0)
        mean_nni = float(hrv["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv else 0.0
        sdnn = float(hrv["HRV_SDNN"].iloc[0]) if "HRV_SDNN" in hrv else 0.0
        rmssd = float(hrv["HRV_RMSSD"].iloc[0]) if "HRV_RMSSD" in hrv else 0.0
        pnn50 = float(hrv["HRV_pNN50"].iloc[0]) if "HRV_pNN50" in hrv else 0.0
        lf = float(hrv["HRV_LF"].iloc[0]) if "HRV_LF" in hrv else None
        hf = float(hrv["HRV_HF"].iloc[0]) if "HRV_HF" in hrv else None
        lf_hf = float(hrv["HRV_LFHF"].iloc[0]) if "HRV_LFHF" in hrv else None
    except ImportError:
        # Fallback: compute basic time-domain HRV from peak detection
        from scipy.signal import find_peaks  # type: ignore
        peaks, _ = find_peaks(signal, height=signal.std() * 1.5,
                              distance=int(detected_fs * 0.5))
        if len(peaks) < 2:
            raise HTTPException(422, "Not enough R-peaks for HRV")
        rr = np.diff(peaks) / detected_fs * 1000  # ms
        heart_rate = 60.0 / (np.mean(rr) / 1000.0)
        mean_nni = float(np.mean(rr))
        sdnn = float(np.std(rr))
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        pnn50 = float(100 * np.sum(np.abs(np.diff(rr)) > 50) / len(rr))
        lf = hf = lf_hf = None
    except Exception as e:
        raise HTTPException(500, f"HRV computation failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    # Build FHIR Observation
    import uuid
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    fhir = {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "8889-8",
                "display": "Heart Rate Variability",
            }],
            "text": "HRV (AI-computed from ECG)",
        },
        "effectiveDateTime": ts,
        "issued": ts,
        "performer": [{"display": "NeuroKit2 service"}],
        "component": [
            {"code": {"text": "Mean NN interval"}, "valueQuantity": {"value": round(mean_nni, 1), "unit": "ms"}},
            {"code": {"text": "SDNN"}, "valueQuantity": {"value": round(sdnn, 1), "unit": "ms"}},
            {"code": {"text": "RMSSD"}, "valueQuantity": {"value": round(rmssd, 1), "unit": "ms"}},
            {"code": {"text": "pNN50"}, "valueQuantity": {"value": round(pnn50, 1), "unit": "%"}},
        ],
        "note": [{"text": "Computed by NeuroKit2. Clinical interpretation required."}],
    }
    if lf is not None:
        fhir["component"].append({"code": {"text": "LF power"}, "valueQuantity": {"value": round(lf, 1), "unit": "ms2"}})
    if hf is not None:
        fhir["component"].append({"code": {"text": "HF power"}, "valueQuantity": {"value": round(hf, 1), "unit": "ms2"}})
    if lf_hf is not None:
        fhir["component"].append({"code": {"text": "LF/HF ratio"}, "valueQuantity": {"value": round(lf_hf, 2), "unit": "1"}})

    return HRVResult(
        heart_rate_bpm=round(heart_rate, 1),
        n_nni=len(peaks) if 'peaks' in locals() else 0,
        mean_nni_ms=round(mean_nni, 1),
        sdnn_ms=round(sdnn, 1),
        rmssd_ms=round(rmssd, 1),
        pnn50_percent=round(pnn50, 1),
        lf_ms2=lf,
        hf_ms2=hf,
        lf_hf_ratio=lf_hf,
        inference_ms=round(dt_ms, 1),
        fhir=fhir,
    )


@app.post("/quality", response_model=QualityResult)
async def assess_quality(
    file: UploadFile = File(...),
    fs: float = Query(250.0),
):
    """Assess ECG signal quality."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        signal, detected_fs = _load_signal_from_csv(tmp_path, fs)
    except Exception as e:
        raise HTTPException(422, f"Could not parse ECG: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    try:
        # Compute quality metrics directly
        # 1. Baseline wander (low-frequency content < 0.5 Hz)
        from scipy.signal import welch  # type: ignore
        f, Pxx = welch(signal, fs=detected_fs, nperseg=min(len(signal), 1024))
        baseline_power = float(np.sum(Pxx[f < 0.5]))
        total_power = float(np.sum(Pxx))
        baseline_wander = baseline_power / max(total_power, 1e-10)

        # 2. Powerline interference (50/60 Hz peak)
        powerline_50 = float(Pxx[np.argmin(np.abs(f - 50.0))]) if len(Pxx) > 0 else 0.0
        powerline_60 = float(Pxx[np.argmin(np.abs(f - 60.0))]) if len(Pxx) > 0 else 0.0
        powerline_interference = (powerline_50 + powerline_60) / max(total_power, 1e-10)

        # 3. Signal-to-noise ratio
        signal_power = float(np.mean(signal ** 2))
        noise_power = float(np.var(np.diff(signal)))  # high-freq noise proxy
        snr_db = 10 * np.log10(max(signal_power, 1e-10) / max(noise_power, 1e-10))

        # 4. Overall quality (heuristic)
        overall = max(0.0, min(1.0, 1.0 - baseline_wander * 5 - powerline_interference * 10
                               + (snr_db + 20) / 60))
        overall = max(0.0, min(1.0, overall))
    except Exception as e:
        raise HTTPException(500, f"Quality assessment failed: {e}")
    dt_ms = (time.perf_counter() - t0) * 1000

    return QualityResult(
        overall_quality=round(float(overall), 3),
        baseline_wander=round(float(baseline_wander), 4),
        powerline_interference=round(float(powerline_interference), 4),
        signal_to_noise_db=round(float(snr_db), 1),
        inference_ms=round(dt_ms, 1),
    )


@app.get("/")
def root():
    return {
        "service": "neurokit",
        "upstream": "https://github.com/neuropsychology/NeuroKit",
        "endpoints": ["/healthz", "/readyz", "/r-peak", "/hrv", "/quality"],
        "docs": "/docs",
    }
