"""Minimal DICOM C-STORE gateway tests (no real PACS needed)."""
import pytest
from pathlib import Path
import tempfile


def test_modality_routes_complete():
    """All expected modalities have a route."""
    from app.main import MODALITY_ROUTES
    expected = {"US", "ECG", "MR", "CT"}
    assert expected.issubset(MODALITY_ROUTES.keys())


def test_study_buffer_add_file(tmp_path):
    """StudyBuffer should accumulate files by StudyInstanceUID."""
    from app.main import StudyBuffer
    buffer = StudyBuffer(tmp_path / "incoming")
    # Simulate a fake dataset
    class FakeDataset:
        StudyInstanceUID = "1.2.3.4.5"
        SeriesInstanceUID = "1.2.3.4.6"
        SOPInstanceUID = "1.2.3.4.7"
        PatientID = "PATIENT-001"
        Modality = "US"
        StudyDescription = "Echo"
        file_meta = type("M", (), {"MediaStorageSOPClassUID": "1.2.840.10008.5.1.4.1.1.3.1",
                                    "MediaStorageSOPInstanceUID": "1.2.3.4.7",
                                    "TransferSyntaxUID": "1.2.840.10008.1.2.1"})()
        def save_as(self, path):
            Path(path).write_bytes(b"FAKE_DICOM")

    buffer.add_file(FakeDataset())
    assert "1.2.3.4.5" in buffer._studies
    assert buffer._study_meta["1.2.3.4.5"]["patient_id"] == "PATIENT-001"
    assert buffer._study_meta["1.2.3.4.5"]["modality"] == "US"


def test_study_buffer_flush_clears_state(tmp_path):
    from app.main import StudyBuffer
    buffer = StudyBuffer(tmp_path / "incoming")
    buffer._studies = {"1.2.3": [Path("/tmp/x.dcm")]}
    buffer._study_meta = {"1.2.3": {"patient_id": "P1", "modality": "US"}}
    flushed = buffer.flush()
    assert "1.2.3" in flushed["meta"]
    assert "1.2.3" in flushed["files"]
    assert len(buffer._studies) == 0
    assert len(buffer._study_meta) == 0


def test_setup_ae():
    """AE should be configurable with the right AE title."""
    from app.main import setup_ae
    ae = setup_ae("TEST_AE")
    assert ae.ae_title == "TEST_AE"
