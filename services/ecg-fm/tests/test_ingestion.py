"""Tests for ECG ingestion (CSV loader only — no WFDB/DICOM/MUSE deps needed)."""
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from app.ingestion import load_ecg, STANDARD_LEADS


def _make_csv_ecg(n_samples: int = 2500, fs: float = 250.0) -> Path:
    """Create a 12-lead CSV ECG at the given sample rate."""
    tmp = Path(tempfile.mkdtemp()) / "test_ecg.csv"
    rng = np.random.default_rng(42)
    # Simulate 12-lead ECG: 250 Hz × 10 seconds = 2500 samples
    t = np.arange(n_samples) / fs
    # Add a synthetic QRS-like spike every 0.85s (HR ~70)
    qrs_mask = (t * (1 / 0.85)) % 1 < 0.02
    signal = rng.normal(0, 0.1, (n_samples, 12))
    signal[qrs_mask] += 1.0
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(STANDARD_LEADS)
        for row in signal:
            writer.writerow([f"{v:.4f}" for v in row])
    # Sidecar JSON for sample rate
    import json
    (Path(str(tmp) + ".json")).write_text(json.dumps({"sample_rate_hz": fs}))
    return tmp


def test_load_csv_returns_correct_shape():
    csv_path = _make_csv_ecg(n_samples=2500, fs=250.0)
    tensor, meta = load_ecg(csv_path)
    assert tensor.shape == (1, 12, 1000), f"Expected (1, 12, 1000), got {tensor.shape}"
    assert meta.format == "csv"
    assert meta.n_leads == 12
    assert meta.n_samples == 1000
    assert meta.sample_rate_hz == 250


def test_load_csv_resamples_to_250hz():
    # Create a 500 Hz ECG, should be resampled to 250 Hz × 1000 samples
    csv_path = _make_csv_ecg(n_samples=5000, fs=500.0)
    tensor, meta = load_ecg(csv_path)
    assert tensor.shape == (1, 12, 1000)
    assert meta.sample_rate_hz == 250


def test_load_csv_pads_short_signal():
    # 2 seconds @ 250 Hz = 500 samples — should be padded to 1000
    csv_path = _make_csv_ecg(n_samples=500, fs=250.0)
    tensor, _ = load_ecg(csv_path)
    assert tensor.shape == (1, 12, 1000)


def test_load_csv_crops_long_signal():
    # 10 seconds @ 250 Hz = 2500 samples — should be center-cropped to 1000
    csv_path = _make_csv_ecg(n_samples=2500, fs=250.0)
    tensor, _ = load_ecg(csv_path)
    assert tensor.shape == (1, 12, 1000)


def test_tensor_is_float32():
    csv_path = _make_csv_ecg()
    tensor, _ = load_ecg(csv_path)
    assert tensor.dtype == torch.float32


def test_tensor_is_zscored():
    csv_path = _make_csv_ecg()
    tensor, _ = load_ecg(csv_path)
    # Each lead should be approximately mean 0, std 1
    for lead_idx in range(12):
        lead = tensor[0, lead_idx].numpy()
        assert abs(lead.mean()) < 0.1, f"Lead {lead_idx} mean = {lead.mean()}"
        assert 0.5 < lead.std() < 1.5, f"Lead {lead_idx} std = {lead.std()}"


def test_standard_leads_count():
    assert len(STANDARD_LEADS) == 12
    assert STANDARD_LEADS[0] == "I"
    assert STANDARD_LEADS[-1] == "V6"
