"""ECG signal ingestion — WFDB, DICOM waveform, MUSE XML, CSV.

ECG-FM expects a (12, 1000) tensor at 250 Hz (4 seconds). This module loads
from the four common hospital ECG formats:

1. **WFDB** (.hea + .dat) — PhysioNet standard. Used by MIT-BIH, PTB-XL.
2. **DICOM Waveform** (.dcm) — 12-lead ECG stored as multi-frame DICOM.
   Common on GE MAC5500, Philips TC55.
3. **MUSE XML** (.xml) — GE MUSE / Philips XML export format.
4. **CSV** (.csv) — Generic 12-column CSV with optional header.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

PathLike = Union[str, Path]

# Standard 12-lead order (SCP-ECG / IEC 62364-1)
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ECG-FM's expected input: 12 leads × 1000 samples @ 250 Hz (4 seconds)
TARGET_FS = 250
TARGET_LEN = 1000


@dataclass
class ECGMetadata:
    format: str  # "wfdb" | "dicom" | "muse" | "csv"
    n_leads: int
    n_samples: int
    sample_rate_hz: float
    duration_s: float
    patient_id: Optional[str] = None
    study_instance_uid: Optional[str] = None
    acquisition_datetime: Optional[str] = None
    leads: list = None


# ----------------------------------------------------------------------
# WFDB loader
# ----------------------------------------------------------------------

def _load_wfdb(path: PathLike) -> Tuple[np.ndarray, ECGMetadata]:
    """Load a WFDB record. Requires the `wfdb` package."""
    try:
        import wfdb  # type: ignore
    except ImportError as e:
        raise RuntimeError("WFDB loader requires `pip install wfdb`") from e

    record = wfdb.rdrecord(str(path))
    # record.p_signal: (n_samples, n_leads)
    sig = record.p_signal.astype(np.float32)
    fs = float(record.fs)
    leads = record.sig_name if hasattr(record, "sig_name") else STANDARD_LEADS[: sig.shape[1]]
    return sig, ECGMetadata(
        format="wfdb",
        n_leads=sig.shape[1],
        n_samples=sig.shape[0],
        sample_rate_hz=fs,
        duration_s=sig.shape[0] / fs,
        leads=leads,
    )


# ----------------------------------------------------------------------
# DICOM Waveform loader
# ----------------------------------------------------------------------

def _load_dicom_waveform(path: PathLike) -> Tuple[np.ndarray, ECGMetadata]:
    """Load a 12-lead ECG stored as DICOM Waveform."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    if "WaveformSequence" not in ds:
        raise ValueError("DICOM file has no WaveformSequence — not an ECG waveform")

    wv = ds.WaveformSequence[0]
    n_channels = int(wv.NumberOfWaveformChannels)
    n_samples = int(wv.NumberOfWaveformSamples)
    fs = 1.0 / float(wv.SamplingFrequency) if "SamplingFrequency" in wv else 1.0 / 0.004  # default 250 Hz
    fs = 1.0 / fs  # convert sample period to frequency

    # Decode waveform data
    arr = np.frombuffer(wv.WaveformData, dtype=np.int16).astype(np.float32)
    arr = arr.reshape(n_samples, n_channels)

    # Channel definitions
    leads = []
    for ch_def in wv.ChannelDefinitionSeq if "ChannelDefinitionSeq" in wv else []:
        if "ChannelSourceSeq" in ch_def:
            src = ch_def.ChannelSourceSeq[0]
            leads.append(str(src.CodeMeaning) if "CodeMeaning" in src else f"ch{len(leads)}")

    return arr, ECGMetadata(
        format="dicom",
        n_leads=n_channels,
        n_samples=n_samples,
        sample_rate_hz=fs,
        duration_s=n_samples / fs,
        patient_id=str(getattr(ds, "PatientID", None)),
        study_instance_uid=str(getattr(ds, "StudyInstanceUID", None)),
        acquisition_datetime=str(getattr(ds, "AcquisitionDateTime", None)),
        leads=leads or STANDARD_LEADS[:n_channels],
    )


# ----------------------------------------------------------------------
# MUSE XML loader
# ----------------------------------------------------------------------

def _parse_muse_xml(path: PathLike) -> Tuple[np.ndarray, ECGMetadata]:
    """Parse GE MUSE XML ECG export."""
    tree = ET.parse(str(path))
    root = tree.getroot()

    # Find the waveform data — MUSE XML stores it under
    # <RestingECGMeasurements><Waveform><LeadData>...
    # The actual sample text is under <SampleType>, <LeadData>, etc.
    # Different MUSE versions use slightly different tags, so we go
    # pattern-based.
    lead_data = {}
    for lead_elem in root.iter("LeadData"):
        lead_name = lead_elem.findtext("Lead", "") or lead_elem.findtext("LeadName", "")
        samples_text = lead_elem.findtext("SampleType") or lead_elem.findtext("Samples") or ""
        # Samples are typically space-separated integers
        samples = [int(x) for x in re.split(r"[\s,]+", samples_text.strip()) if x]
        if lead_name and samples:
            lead_data[lead_name.strip().upper()] = samples

    if not lead_data:
        raise ValueError("No lead data found in MUSE XML")

    # Find sample rate / duration
    fs_elem = root.find(".//SampleRate")
    fs = float(fs_elem.text) if fs_elem is not None and fs_elem.text else 250.0

    # Resample to align all leads to the longest
    max_len = max(len(v) for v in lead_data.values())
    leads = list(lead_data.keys())
    arr = np.zeros((max_len, len(leads)), dtype=np.float32)
    for i, lead in enumerate(leads):
        v = lead_data[lead]
        arr[: len(v), i] = v

    # Get patient ID if present
    patient_id = root.findtext(".//PatientID") or root.findtext(".//Patient/ID")

    return arr, ECGMetadata(
        format="muse",
        n_leads=len(leads),
        n_samples=max_len,
        sample_rate_hz=fs,
        duration_s=max_len / fs,
        patient_id=patient_id,
        leads=leads,
    )


# ----------------------------------------------------------------------
# CSV loader
# ----------------------------------------------------------------------

def _load_csv(path: PathLike) -> Tuple[np.ndarray, ECGMetadata]:
    """Load a generic CSV with 12 columns (one per lead).

    Assumes:
    - First row is header (lead names) OR numeric data
    - Each subsequent row is one sample
    - Sample rate given via filename suffix `_250hz` or via sidecar JSON
    """
    import csv

    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Detect header: if first row contains non-numeric strings
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    has_header = not all(is_number(c) for c in rows[0] if c)
    if has_header:
        leads = [c.strip().upper() for c in rows[0]]
        data_rows = rows[1:]
    else:
        leads = STANDARD_LEADS[: len(rows[0])]
        data_rows = rows

    arr = np.array([[float(c) for c in row] for row in data_rows if row], dtype=np.float32)

    # Default to 250 Hz; can be overridden via sidecar .json
    sidecar = Path(str(path) + ".json")
    fs = 250.0
    if sidecar.exists():
        import json
        with sidecar.open() as f:
            meta = json.load(f)
            fs = float(meta.get("sample_rate_hz", 250.0))

    return arr, ECGMetadata(
        format="csv",
        n_leads=arr.shape[1],
        n_samples=arr.shape[0],
        sample_rate_hz=fs,
        duration_s=arr.shape[0] / fs,
        leads=leads,
    )


# ----------------------------------------------------------------------
# Resampling + lead normalization
# ----------------------------------------------------------------------

def _resample(signal: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """Linear resample along the time axis. signal shape: (N, C)."""
    if fs_in == fs_out:
        return signal
    n_out = int(round(signal.shape[0] * fs_out / fs_in))
    indices = np.linspace(0, signal.shape[0] - 1, n_out)
    x_old = np.arange(signal.shape[0])
    # np.interp doesn't support axis kwarg in older versions; do per-column
    out = np.zeros((n_out, signal.shape[1]), dtype=signal.dtype)
    for c in range(signal.shape[1]):
        out[:, c] = np.interp(indices, x_old, signal[:, c])
    return out


def _reorder_leads(signal: np.ndarray, leads: list) -> np.ndarray:
    """Reorder columns to match STANDARD_LEADS. Drops unknown leads."""
    if leads is None:
        return signal
    cols = []
    for std in STANDARD_LEADS:
        if std.upper() in [l.upper() for l in leads]:
            idx = [i for i, l in enumerate(leads) if l.upper() == std.upper()][0]
            cols.append(signal[:, idx])
    if not cols:
        return signal
    return np.stack(cols, axis=1)


def _pad_or_crop(signal: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Pad with zeros or center-crop to target length along axis 0."""
    n = signal.shape[0]
    if n == target_len:
        return signal
    if n < target_len:
        pad = np.zeros((target_len - n, signal.shape[1]), dtype=signal.dtype)
        return np.concatenate([signal, pad], axis=0)
    # Center crop
    start = (n - target_len) // 2
    return signal[start : start + target_len]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load_ecg(path: PathLike) -> Tuple[torch.Tensor, ECGMetadata]:
    """Load an ECG file in any supported format.

    Returns a (1, 12, 1000) float32 tensor at 250 Hz, ready for ECG-FM.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".hea":  # WFDB header — companion .dat expected
        signal, meta = _load_wfdb(path.with_suffix(""))
    elif suffix in (".dcm", ".dicom"):
        signal, meta = _load_dicom_waveform(path)
    elif suffix == ".xml":
        signal, meta = _parse_muse_xml(path)
    elif suffix == ".csv":
        signal, meta = _load_csv(path)
    else:
        # Try each loader in turn
        for loader in [_load_dicom_waveform, _parse_muse_xml, _load_csv, _load_wfdb]:
            try:
                signal, meta = loader(path)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not load ECG from {path}: unknown format")

    # Resample to 250 Hz
    signal = _resample(signal, meta.sample_rate_hz, TARGET_FS)

    # Reorder leads to standard 12-lead order
    if meta.leads and signal.shape[1] == 12:
        signal = _reorder_leads(signal, meta.leads)

    # Pad or crop to 1000 samples (4 seconds)
    signal = _pad_or_crop(signal, TARGET_LEN)

    # Normalize per-lead (z-score) — ECG-FM is robust to this
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True) + 1e-6
    signal = (signal - mean) / std

    # (N, C) -> (1, C, N) — batch × channels × samples
    tensor = torch.from_numpy(signal.T).unsqueeze(0).float()
    meta.n_samples = TARGET_LEN
    meta.sample_rate_hz = TARGET_FS
    meta.duration_s = TARGET_LEN / TARGET_FS
    return tensor, meta
