# EchoNet-Dynamic service — quick tests that don't require GPU or model weights.
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from app.ingestion.dicom_loader import sniff_vendor, extract_metadata, VENDOR_PATTERNS


def _make_fake_dicom(vendor: str, frames: int = 4) -> FileDataset:
    """Build a tiny multi-frame DICOM in memory for testing."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"  # US Image
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR LE
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset("test.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "TEST-001"
    ds.Manufacturer = vendor
    ds.ManufacturerModelName = f"{vendor} Model X"
    ds.StationName = f"{vendor}-STATION-1"
    ds.SoftwareVersions = "1.0.0"
    ds.Modality = "US"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.NumberOfFrames = frames
    ds.Rows = 32
    ds.Columns = 32
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 0

    # Random pixel data (uint8, RGB)
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 255, size=(frames, 32, 32, 3), dtype=np.uint8)
    ds.PixelData = pixels.tobytes()
    return ds


def test_vendor_detection_ge():
    ds = _make_fake_dicom("GE")
    assert sniff_vendor(ds) == "ge"


def test_vendor_detection_philips():
    ds = _make_fake_dicom("Philips")
    assert sniff_vendor(ds) == "philips"


def test_vendor_detection_siemens():
    ds = _make_fake_dicom("Siemens")
    assert sniff_vendor(ds) == "siemens"


def test_vendor_detection_generic():
    ds = _make_fake_dicom("Acme Medical")
    assert sniff_vendor(ds) == "generic"


def test_metadata_extraction():
    ds = _make_fake_dicom("GE")
    meta = extract_metadata(ds)
    assert meta.vendor == "ge"
    assert meta.manufacturer == "GE"
    assert meta.n_frames == 4
    assert meta.rows == 32
    assert meta.cols == 32
    assert meta.photometric == "RGB"


def test_vendor_patterns_present():
    # Sanity: each vendor has at least one compiled pattern
    for vendor, patterns in VENDOR_PATTERNS.items():
        assert patterns, f"vendor {vendor} has no patterns"
