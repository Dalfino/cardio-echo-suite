"""DICOM utilities shared across services.

- `extract_study_uid(ds)` — extract StudyInstanceUID from a pydicom Dataset
- `extract_patient_id(ds)` — extract PatientID
- `read_dicom_safe(path)` — read DICOM with error handling
- `is_dicom_file(path)` — quick check
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import pydicom

PathLike = Union[str, Path]


def is_dicom_file(path: PathLike) -> bool:
    """Quick check: does this look like a DICOM file?"""
    path = Path(path)
    if not path.is_file():
        return False
    # DICOM files have the magic "DICM" at offset 128
    try:
        with path.open("rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def read_dicom_safe(path: PathLike) -> Optional[pydicom.Dataset]:
    """Read a DICOM file, returning None on failure."""
    try:
        return pydicom.dcmread(str(path))
    except Exception:
        return None


def extract_study_uid(ds: pydicom.Dataset) -> Optional[str]:
    """Extract StudyInstanceUID from a DICOM dataset."""
    if "StudyInstanceUID" in ds:
        return str(ds.StudyInstanceUID)
    return None


def extract_series_uid(ds: pydicom.Dataset) -> Optional[str]:
    if "SeriesInstanceUID" in ds:
        return str(ds.SeriesInstanceUID)
    return None


def extract_sop_uid(ds: pydicom.Dataset) -> Optional[str]:
    if "SOPInstanceUID" in ds:
        return str(ds.SOPInstanceUID)
    return None


def extract_patient_id(ds: pydicom.Dataset) -> Optional[str]:
    """Extract PatientID, falling back to PatientName if absent."""
    if "PatientID" in ds:
        return str(ds.PatientID)
    if "PatientName" in ds:
        return str(ds.PatientName)
    return None


def extract_encounter_id(ds: pydicom.Dataset) -> Optional[str]:
    """Extract AccessionNumber as a proxy for encounter ID."""
    if "AccessionNumber" in ds:
        return str(ds.AccessionNumber)
    return None


def extract_metadata_summary(ds: pydicom.Dataset) -> dict:
    """Extract a summary of clinically relevant DICOM metadata."""
    return {
        "patient_id": extract_patient_id(ds),
        "study_instance_uid": extract_study_uid(ds),
        "series_instance_uid": extract_series_uid(ds),
        "sop_instance_uid": extract_sop_uid(ds),
        "accession_number": extract_encounter_id(ds),
        "study_description": str(getattr(ds, "StudyDescription", "") or ""),
        "study_date": str(getattr(ds, "StudyDate", "") or ""),
        "study_time": str(getattr(ds, "StudyTime", "") or ""),
        "modality": str(getattr(ds, "Modality", "") or ""),
        "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
        "manufacturer_model_name": str(getattr(ds, "ManufacturerModelName", "") or ""),
        "station_name": str(getattr(ds, "StationName", "") or ""),
    }


def load_dicom_with_metadata(path: PathLike) -> Tuple[Optional[pydicom.Dataset], dict]:
    """Read a DICOM file and return (dataset, metadata_summary).

    Returns (None, {}) if the file is not a valid DICOM.
    """
    ds = read_dicom_safe(path)
    if ds is None:
        return None, {}
    return ds, extract_metadata_summary(ds)
