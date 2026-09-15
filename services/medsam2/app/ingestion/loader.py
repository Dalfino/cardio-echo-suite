"""Image ingestion for MedSAM2 — DICOM/PNG/NIfTI loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

PathLike = Union[str, Path]


@dataclass
class ImageMetadata:
    format: str
    shape: Tuple[int, ...]
    spacing_mm: Optional[Tuple[float, ...]] = None
    patient_id: Optional[str] = None
    study_instance_uid: Optional[str] = None


def load_image(path: PathLike) -> Tuple[torch.Tensor, ImageMetadata]:
    """Load a medical image into a (1, C, H, W) or (1, C, D, H, W) tensor.

    Supports:
    - PNG/JPG (2D)
    - DICOM single-frame (2D)
    - NIfTI (.nii/.nii.gz, 3D)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        # (H, W, 3) -> (1, 3, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor, ImageMetadata(format="image", shape=tensor.shape)

    if suffix in (".dcm", ".dicom"):
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr.astype(np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        return tensor, ImageMetadata(
            format="dicom",
            shape=tensor.shape,
            patient_id=str(getattr(ds, "PatientID", None) or "") or None,
            study_instance_uid=str(getattr(ds, "StudyInstanceUID", None) or "") or None,
        )

    if suffix in (".nii", ".gz"):
        try:
            import nibabel as nib  # type: ignore
        except ImportError as e:
            raise RuntimeError("NIfTI loading requires `pip install nibabel`") from e
        img = nib.load(str(path))
        arr = img.get_fdata().astype(np.float32)
        # Normalize
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        # (D, H, W) or (H, W, D) -> (1, 1, D, H, W)
        if arr.ndim == 3:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        else:
            tensor = torch.from_numpy(arr).unsqueeze(0).float()
        spacing = tuple(float(s) for s in img.header.get_zooms())
        return tensor, ImageMetadata(format="nifti", shape=tensor.shape, spacing_mm=spacing)

    raise ValueError(f"Unsupported image format: {suffix}")
