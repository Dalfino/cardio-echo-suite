"""Tests for Phase A additions: DICOM-SR converter, PHI redaction, accuracy tier 2."""
import torch
import numpy as np
from cardio_echo_core.dicom_sr import fhir_to_dicom_sr
from cardio_echo_core.phi_redaction import redact_phi
from cardio_echo_core.accuracy import (
    consistency_test,
    fit_temperature,
    apply_temperature,
    assess_image_quality,
    assess_video_quality,
)


# ----------------------------------------------------------------------
# DICOM-SR converter
# ----------------------------------------------------------------------

def test_dicom_sr_basic_diagnostic_report():
    fhir = {
        "resourceType": "DiagnosticReport",
        "id": "test-1",
        "status": "preliminary",
        "conclusion": "EF = 55%. Normal LV systolic function.",
        "contained": [
            {
                "resourceType": "Observation",
                "id": "obs-ef",
                "code": {"coding": [{"system": "http://loinc.org", "code": "10230-1", "display": "Ejection fraction"}]},
                "valueQuantity": {"value": 55.0, "unit": "%"},
            }
        ],
    }
    sr = fhir_to_dicom_sr(
        fhir_report=fhir,
        patient_id="PATIENT-001",
        patient_name="TEST^PATIENT",
        study_instance_uid="1.2.3.4.5",
    )
    assert sr.Modality == "SR"
    assert str(sr.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.88.33"  # Comprehensive SR
    assert sr.PatientID == "PATIENT-001"
    assert sr.StudyInstanceUID == "1.2.3.4.5"
    assert hasattr(sr, "ContentSequence")
    assert len(sr.ContentSequence) > 0


def test_dicom_sr_bundle_input():
    fhir = {
        "resourceType": "Bundle",
        "total": 2,
        "entry": [
            {"resource": {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
                "valueQuantity": {"value": 70.0, "unit": "/min"},
            }},
            {"resource": {
                "resourceType": "Observation",
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": "164873001", "display": "Atrial fibrillation"}]},
                "valueCodeableConcept": {"coding": [{"system": "http://snomed.info/sct", "code": "164873001", "display": "Atrial fibrillation"}]},
            }},
        ],
    }
    sr = fhir_to_dicom_sr(fhir_report=fhir, patient_id="P1")
    assert sr.Modality == "SR"
    assert len(sr.ContentSequence) >= 2


def test_dicom_sr_auto_generates_uids():
    sr = fhir_to_dicom_sr(
        fhir_report={"resourceType": "DiagnosticReport", "conclusion": "ok"},
        patient_id="P1",
    )
    # Should auto-generate UIDs if not provided
    assert str(sr.StudyInstanceUID)
    assert str(sr.SOPInstanceUID)
    assert str(sr.SeriesInstanceUID)


# ----------------------------------------------------------------------
# PHI redaction
# ----------------------------------------------------------------------

def test_redact_dicom_name():
    s = redact_phi("Patient name: SMITH^JOHN^A")
    assert "SMITH^JOHN^A" not in s
    assert "[REDACTED-PHI]" in s


def test_redact_iso_date():
    s = redact_phi("DOB: 1985-03-15")
    assert "1985-03-15" not in s
    assert "[REDACTED-PHI]" in s


def test_redact_dicom_date():
    s = redact_phi("Study date: 19850315")
    assert "19850315" not in s


def test_redact_ssn():
    s = redact_phi("SSN: 123-45-6789")
    assert "123-45-6789" not in s


def test_redact_phone():
    s = redact_phi("Phone: (555) 123-4567")
    assert "(555) 123-4567" not in s


def test_redact_email():
    s = redact_phi("Email: patient@example.com")
    assert "patient@example.com" not in s


def test_redact_long_numeric():
    s = redact_phi("MRN: 123456789")
    assert "123456789" not in s


def test_redact_preserves_non_phi():
    s = redact_phi("EF = 55.0%, normal study")
    assert "EF = 55.0%" in s
    assert "normal study" in s


def test_redact_handles_non_string():
    s = redact_phi(12345)
    assert isinstance(s, str)


def test_redact_handles_none():
    s = redact_phi(None)
    assert s == ""


# ----------------------------------------------------------------------
# Accuracy tier 2: consistency test
# ----------------------------------------------------------------------

def test_consistency_test_deterministic():
    """A deterministic function should pass consistency test."""
    result = consistency_test(
        predict_fn=lambda: torch.tensor([1.0, 2.0, 3.0]),
        n_runs=5,
        threshold=1e-6,
    )
    assert result.is_consistent
    assert result.max_variance < 1e-6
    assert result.n_runs == 5


def test_consistency_test_noisy():
    """A noisy function should fail consistency test."""
    rng = np.random.default_rng(42)
    result = consistency_test(
        predict_fn=lambda: torch.tensor(rng.normal(0, 1.0)),
        n_runs=20,
        threshold=0.01,
    )
    assert not result.is_consistent
    assert result.max_variance > 0.01


# ----------------------------------------------------------------------
# Accuracy tier 2: calibration
# ----------------------------------------------------------------------

def test_fit_temperature_reduces_nll():
    """Temperature scaling should not increase NLL."""
    torch.manual_seed(42)
    # 100 samples, 3 classes, slightly miscalibrated logits
    logits = torch.randn(100, 3) * 3  # overconfident
    labels = torch.randint(0, 3, (100,))
    result = fit_temperature(logits, labels)
    assert result.nll_after <= result.nll_before + 1e-3  # allow small numerical slack
    assert result.temperature > 0


def test_apply_temperature_outputs_probabilities():
    logits = torch.randn(5, 3)
    probs = apply_temperature(logits, temperature=2.0)
    assert probs.shape == (5, 3)
    # Probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)


# ----------------------------------------------------------------------
# Accuracy tier 2: image quality gating
# ----------------------------------------------------------------------

def test_quality_gate_accepts_normal_image():
    img = torch.rand(3, 64, 64) * 0.7 + 0.15  # mean ~0.5
    gate = assess_image_quality(img)
    assert gate.is_acceptable
    assert gate.quality_score > 0.5


def test_quality_gate_rejects_black_image():
    img = torch.zeros(3, 64, 64)
    gate = assess_image_quality(img)
    assert not gate.is_acceptable
    assert "Std too low" in " ".join(gate.reasons)


def test_quality_gate_rejects_white_image():
    img = torch.ones(3, 64, 64)
    gate = assess_image_quality(img)
    assert not gate.is_acceptable


def test_quality_gate_rejects_noisy_flat():
    img = torch.full((3, 64, 64), 0.5)  # constant image
    gate = assess_image_quality(img)
    assert not gate.is_acceptable


def test_video_quality_accepts_normal():
    video = torch.rand(1, 8, 3, 64, 64) * 0.7 + 0.15
    gate = assess_video_quality(video)
    assert gate.is_acceptable


def test_video_quality_rejects_all_black():
    video = torch.zeros(1, 8, 3, 64, 64)
    gate = assess_video_quality(video)
    assert not gate.is_acceptable
