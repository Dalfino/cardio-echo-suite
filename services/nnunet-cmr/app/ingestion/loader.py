"""CMR image ingestion — NIfTI/DICOM stack loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

PathLike = Union[str, Path]


@dataclass
class CMRMetadata:
    format: str
    shape: Tuple[int, ...]
    spacing_mm: Tuple[float, ...]
    n_slices: int
    patient_id: Optional[str] = None
    study_instance_uid: Optional[str] = None


def load_cmr(path: PathLike) -> Tuple[torch.Tensor, CMRMetadata]:
    """Load a cardiac MRI short-axis stack.

    Accepts:
    - NIfTI (.nii / .nii.gz) — 3D volume
    - Directory of DICOM files — sorted by InstanceNumber
    """
    path = Path(path)

    if path.is_dir():
        return _load_dicom_stack(path)
    if path.suffix.lower() in (".nii", ".gz"):
        return _load_nifti(path)
    if path.suffix.lower() in (".dcm", ".dicom"):
        return _load_dicom_single(path)

    raise ValueError(f"Unsupported CMR format: {path}")


def _load_nifti(path: Path) -> Tuple[torch.Tensor, CMRMetadata]:
    try:
        import nibabel as nib  # type: ignore
    except ImportError as e:
        raise RuntimeError("NIfTI loading requires `pip install nibabel`") from e

    img = nib.load(str(path))
    arr = img.get_fdata().astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    # (H, W, D) -> (1, 1, D, H, W)
    if arr.ndim == 3:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()

    spacing = tuple(float(s) for s in img.header.get_zooms())
    return tensor, CMRMetadata(
        format="nifti", shape=tensor.shape, spacing_mm=spacing,
        n_slices=tensor.shape[-1],
    )


def _load_dicom_stack(directory: Path) -> Tuple[torch.Tensor, CMRMetadata]:
    import pydicom

    files = sorted(directory.glob("*.dcm")) or sorted(directory.iterdir())
    if not files:
        raise ValueError(f"No DICOM files in {directory}")

    slices = []
    patient_id = None
    study_uid = None
    for f in files:
        ds = pydicom.dcmread(str(f))
        if "PixelData" not in ds:
            continue
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        slices.append(arr[..., 0])
        if patient_id is None and "PatientID" in ds:
            patient_id = str(ds.PatientID)
        if study_uid is None and "StudyInstanceUID" in ds:
            study_uid = str(ds.StudyInstanceUID)

    if not slices:
        raise ValueError("No valid DICOM slices found")

    # Stack along new axis: (H, W, N)
    arr = np.stack(slices, axis=-1)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)

    # (H, W, N) -> (1, 1, N, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()

    spacing = (1.0, 1.0, 1.0)  # placeholder; real value from DICOM PixelSpacing + SliceThickness
    return tensor, CMRMetadata(
        format="dicom_stack", shape=tensor.shape, spacing_mm=spacing,
        n_slices=len(slices), patient_id=patient_id, study_instance_uid=study_uid,
    )


def _load_dicom_single(path: Path) -> Tuple[torch.Tensor, CMRMetadata]:
    """Load a single-slice DICOM (degenerate case)."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    # (H, W) -> (1, 1, 1, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
    return tensor, CMRMetadata(
        format="dicom", shape=tensor.shape, spacing_mm=(1.0, 1.0, 1.0),
        n_slices=1, patient_id=str(getattr(ds, "PatientID", "") or "") or None,
        study_instance_uid=str(getattr(ds, "StudyInstanceUID", "") or "") or None,
    )
