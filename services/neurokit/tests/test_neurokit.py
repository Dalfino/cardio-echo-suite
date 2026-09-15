"""Tests for NeuroKit2 service (signal processing only — no GPU/model needed)."""
import csv
import tempfile
from pathlib import Path

import numpy as np


def _make_ecg_csv(n_samples: int = 5000, fs: float = 250.0) -> Path:
    """Create a synthetic ECG CSV with R-peaks every ~1 second (HR ~60)."""
    t = np.arange(n_samples) / fs
    # Synthetic ECG: gaussian peak every 1s + noise
    signal = np.random.default_rng(42).normal(0, 0.05, n_samples)
    for i in range(1, n_samples):
        if i % int(fs) == 0:  # peak every 1s
            signal[i-3:i+3] += 1.0
    tmp = Path(tempfile.mkdtemp()) / "ecg.csv"
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ECG"])
        for v in signal:
            writer.writerow([f"{v:.4f}"])
    return tmp


def test_healthz():
    from app.api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz():
    from app.api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/readyz")
    assert r.status_code == 200
    assert "status" in r.json()


def test_r_peak_detection():
    from app.api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    csv_path = _make_ecg_csv(n_samples=5000, fs=250.0)
    with csv_path.open("rb") as f:
        files = {"file": ("ecg.csv", f.read(), "text/csv")}
        r = client.post("/r-peak", files=files, params={"fs": 250.0})
    assert r.status_code == 200
    data = r.json()
    # Should detect ~4-5 peaks in 20 seconds of signal at HR=60
    assert data["n_peaks"] >= 3, f"Expected >=3 peaks, got {data['n_peaks']}"
    assert len(data["peak_indices"]) == data["n_peaks"]
    assert len(data["rr_intervals_ms"]) == data["n_peaks"] - 1


def test_quality_assessment():
    from app.api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    csv_path = _make_ecg_csv(n_samples=5000, fs=250.0)
    with csv_path.open("rb") as f:
        files = {"file": ("ecg.csv", f.read(), "text/csv")}
        r = client.post("/quality", files=files, params={"fs": 250.0})
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["overall_quality"] <= 1.0
    assert "baseline_wander" in data
    assert "signal_to_noise_db" in data
