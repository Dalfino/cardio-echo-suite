"""Multi-vendor DICOM ingestion for EchoNet-Dynamic.

Upstream EchoNet-Dynamic consumes AVI videos. Hospital echo machines
(GE Vivid, Philips IE33/EPIQ, Siemens Acuson, Fujifilm) export multi-frame
DICOM files that need vendor-specific handling before being fed to the model.

This module:

1. Reads a DICOM file with pydicom.
2. Sniffs the vendor from DICOM tags (Manufacturer, ManufacturerModelName,
   StationName, SoftwareVersions).
3. Normalizes the pixel data into a (T, H, W, 3) uint8 RGB array.
4. Crops to the ultrasound sector (basic Otsu-based mask, similar to upstream
   `ConvertDICOMToAVI.ipynb` but vendor-aware).
5. Resizes to (112, 112) — the upstream input resolution.
6. Returns a torch tensor ready for the model.

References:
- Upstream notebook: upstream/docs/ConvertDICOMToAVI.ipynb (if present)
- DICOM transfer syntaxes: PS3.5 Annex A
- Vendor tag conventions: dicom.innolitics.com/ciods/us-image
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pydicom
import torch

PathLike = Union[str, Path]


# ----------------------------------------------------------------------
# Vendor detection
# ----------------------------------------------------------------------

VENDOR_PATTERNS = {
    "ge": [
        re.compile(r"\bGE\b", re.I),
        re.compile(r"\bvivid\b", re.I),
        re.compile(r"\blogiq\b", re.I),
    ],
    "philips": [
        re.compile(r"\bphilips\b", re.I),
        re.compile(r"\bie33\b", re.I),
        re.compile(r"\bepiq\b", re.I),
        re.compile(r"\baffiniti\b", re.I),
    ],
    "siemens": [
        re.compile(r"\bsiemens\b", re.I),
        re.compile(r"\bacuson\b", re.I),
        re.compile(r"\bs2000\b", re.I),
        re.compile(r"\bs3000\b", re.I),
    ],
    "fujifilm": [
        re.compile(r"\bfujifilm\b", re.I),
        re.compile(r"\bfuji\b", re.I),
        re.compile(r"\barietta\b", re.I),
    ],
}


@dataclass
class DicomMetadata:
    vendor: str
    manufacturer: Optional[str]
    model: Optional[str]
    station: Optional[str]
    software_versions: Optional[str]
    n_frames: int
    rows: int
    cols: int
    frame_rate: Optional[float]
    photometric: Optional[str]


def _tag(ds: pydicom.Dataset, name: str) -> Optional[str]:
    if name not in ds:
        return None
    val = ds.data_element(name).value
    if isinstance(val, (list, pydicom.multival.MultiValue)):
        return " ".join(str(v) for v in val).strip() or None
    return str(val).strip() or None


def sniff_vendor(ds: pydicom.Dataset) -> str:
    manufacturer = _tag(ds, "Manufacturer") or ""
    model = _tag(ds, "ManufacturerModelName") or ""
    station = _tag(ds, "StationName") or ""
    haystack = f"{manufacturer} {model} {station}"
    for vendor, patterns in VENDOR_PATTERNS.items():
        if any(p.search(haystack) for p in patterns):
            return vendor
    return "generic"


def extract_metadata(ds: pydicom.Dataset) -> DicomMetadata:
    return DicomMetadata(
        vendor=sniff_vendor(ds),
        manufacturer=_tag(ds, "Manufacturer"),
        model=_tag(ds, "ManufacturerModelName"),
        station=_tag(ds, "StationName"),
        software_versions=_tag(ds, "SoftwareVersions"),
        n_frames=int(getattr(ds, "NumberOfFrames", 1)),
        rows=int(getattr(ds, "Rows", 0)),
        cols=int(getattr(ds, "Columns", 0)),
        frame_rate=(float(ds.FrameTime) if "FrameTime" in ds else None),
        photometric=_tag(ds, "PhotometricInterpretation"),
    )


# ----------------------------------------------------------------------
# Pixel decoding
# ----------------------------------------------------------------------

def _decode_pixels(ds: pydicom.Dataset) -> np.ndarray:
    """Decode PixelData into a (T, H, W, 3) uint8 RGB array."""
    if "PixelData" not in ds:
        raise ValueError("DICOM file has no PixelData")

    arr = ds.pixel_array

    if arr.dtype != np.uint8:
        if "WindowCenter" in ds and "WindowWidth" in ds:
            wc = float(ds.WindowCenter)
            ww = float(ds.WindowWidth)
            arr = arr.astype(np.float32)
            arr = np.clip((arr - (wc - 0.5)) / (ww - 1.0) + 0.5, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        arr = arr[None, ..., None]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[None]
        else:
            arr = arr[..., None]
    elif arr.ndim == 4 and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    photo = _tag(ds, "PhotometricInterpretation") or ""
    if photo.upper() == "MONOCHROME1":
        arr = 255 - arr

    return arr


# ----------------------------------------------------------------------
# Sector cropping (vendor-aware)
# ----------------------------------------------------------------------

def _crop_sector(frame: np.ndarray, vendor: str, target: int = 112) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if vendor == "philips":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(frame, (target, target))

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 4
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(frame.shape[1] - x, w + 2 * pad)
    h = min(frame.shape[0] - y, h + 2 * pad)

    cropped = frame[y : y + h, x : x + w]
    return cv2.resize(cropped, (target, target), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load_dicom_echo(path: PathLike, target_size: int = 112, max_frames: int = 256) -> Tuple[torch.Tensor, DicomMetadata]:
    ds = pydicom.dcmread(str(path))
    meta = extract_metadata(ds)
    frames = _decode_pixels(ds)

    T = frames.shape[0]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        frames = frames[idx]

    out = np.stack([_crop_sector(f, meta.vendor, target_size) for f in frames], axis=0)

    tensor = torch.from_numpy(out).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    return tensor, meta


def load_video_file(path: PathLike, target_size: int = 112, max_frames: int = 256) -> Tuple[torch.Tensor, DicomMetadata]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Could not read any frames from {path}")

    out = np.stack(frames, axis=0)
    T = out.shape[0]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames).astype(int)
        out = out[idx]

    tensor = torch.from_numpy(out).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    meta = DicomMetadata(
        vendor="generic",
        manufacturer=None,
        model=None,
        station=None,
        software_versions=None,
        n_frames=int(out.shape[0]),
        rows=target_size,
        cols=target_size,
        frame_rate=None,
        photometric=None,
    )
    return tensor, meta


def load_any(path: PathLike, target_size: int = 112, max_frames: int = 256) -> Tuple[torch.Tensor, DicomMetadata]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".dcm", ".dicom"):
        return load_dicom_echo(path, target_size, max_frames)
    if suffix in (".avi", ".mp4", ".mov", ".mkv"):
        return load_video_file(path, target_size, max_frames)
    try:
        return load_dicom_echo(path, target_size, max_frames)
    except Exception:
        return load_video_file(path, target_size, max_frames)
