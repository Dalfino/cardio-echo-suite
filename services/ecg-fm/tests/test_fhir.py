"""Tests for ECG-FM FHIR DiagnosticReport builder."""
from app.fhir import (
    ARRHYTHMIA_SNOMED,
    INTERVAL_LOINC,
    build_arrhythmia_observation,
    build_ecg_diagnostic_report,
    build_interval_observation,
)


def test_interval_observation_basic():
    obs = build_interval_observation("pr_interval", 165.0, patient_id="P1")
    assert obs["resourceType"] == "Observation"
    assert obs["code"]["coding"][0]["code"] == "44967-8"
    assert obs["valueQuantity"]["value"] == 165.0
    assert obs["valueQuantity"]["unit"] == "ms"
    assert obs["status"] == "preliminary"
    assert obs["subject"]["reference"] == "Patient/P1"


def test_arrhythmia_observation_predicted():
    obs = build_arrhythmia_observation("afib", 0.85, predicted=True, patient_id="P1")
    assert obs["valueCodeableConcept"]["coding"][0]["code"] == "164873001"
    assert obs["interpretation"][0]["coding"][0]["code"] == "POS"
    assert "85.00%" in obs["interpretation"][0]["text"]


def test_arrhythmia_observation_not_predicted():
    obs = build_arrhythmia_observation("afib", 0.05, predicted=False)
    assert obs["interpretation"][0]["coding"][0]["code"] == "NEG"


def test_diagnostic_report_normal_rhythm():
    report = build_ecg_diagnostic_report(
        intervals={"pr_interval": 165.0, "qrs_duration": 95.0, "heart_rate": 70.0},
        arrhythmia_predictions={"sinus_rhythm": 0.92, "afib": 0.02},
        patient_id="P1",
    )
    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "preliminary"
    assert report["code"]["coding"][0]["code"] == "11524-6"
    assert "Sinus rhythm" in report["conclusion"]
    # 3 interval obs + 2 arrhythmia obs = 5 contained
    assert len(report["contained"]) == 5


def test_diagnostic_report_afib_detected():
    report = build_ecg_diagnostic_report(
        intervals={"heart_rate": 95.0},
        arrhythmia_predictions={"sinus_rhythm": 0.20, "afib": 0.85, "rbbb": 0.05},
    )
    assert "Atrial fibrillation" in report["conclusion"]
    assert len(report["conclusionCode"]) >= 1
    assert report["conclusionCode"][0]["coding"][0]["code"] == "164873001"


def test_diagnostic_report_with_study_uid():
    report = build_ecg_diagnostic_report(
        intervals={"heart_rate": 70.0},
        arrhythmia_predictions={"sinus_rhythm": 0.95},
        study_instance_uid="1.2.840.113619.2.55.3.604688119.971",
    )
    assert any(i["value"] == "urn:oid:1.2.840.113619.2.55.3.604688119.971"
               for i in report["identifier"])


def test_arrhythmia_codes_present():
    # Sanity check: critical arrhythmias are in the dictionary
    expected = {"afib", "aflutter", "vt", "vfib", "stemi_anterior",
                "stemi_inferior", "av_block_3", "rbbb", "lbbb"}
    assert expected.issubset(ARRHYTHMIA_SNOMED.keys())


def test_interval_codes_present():
    expected = {"pr_interval", "qrs_duration", "qt_interval",
                "qtc_interval", "rr_interval", "heart_rate"}
    assert expected.issubset(INTERVAL_LOINC.keys())
