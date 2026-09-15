"""Tests for EchoPrime FHIR DiagnosticReport builder."""
from app.fhir import build_diagnostic_report, MEASUREMENT_LOINC, VIEW_SNOMED


def test_diagnostic_report_basic():
    report = build_diagnostic_report(
        views=["A4C", "A2C", "PLAX"],
        measurements={"ef": 58.0, "lvidd": 4.5, "e_e_ratio": 8.0},
        draft_report="Normal LV systolic function.",
        patient_id="P1",
    )
    assert report["resourceType"] == "DiagnosticReport"
    assert report["status"] == "preliminary"
    assert report["code"]["coding"][0]["code"] == "26520-0"
    assert report["subject"]["reference"] == "Patient/P1"
    assert report["conclusion"] == "Normal LV systolic function."
    assert len(report["contained"]) == 3
    assert any(c["id"] == "obs-ef" for c in report["contained"])
    assert len(report["conclusionCode"]) == 3


def test_view_codes_known():
    report = build_diagnostic_report(
        views=["A4C"],
        measurements={},
        draft_report="",
    )
    assert report["conclusionCode"][0]["coding"][0]["system"] == "http://snomed.info/sct"
    assert report["conclusionCode"][0]["coding"][0]["display"] == "Apical 4-chamber view"


def test_view_codes_unknown():
    report = build_diagnostic_report(
        views=["UNKNOWN_VIEW"],
        measurements={},
        draft_report="",
    )
    assert "coding" not in report["conclusionCode"][0]
    assert report["conclusionCode"][0]["text"] == "UNKNOWN_VIEW"


def test_measurement_loinc_present():
    expected = {"ef", "lvidd", "lvids", "ivsd", "pwdd", "la_size",
                "a_peak_v", "m_peak_v", "e_wave", "a_wave", "e_e_ratio", "tr_v"}
    assert expected.issubset(MEASUREMENT_LOINC.keys())


def test_study_instance_uid_adds_identifier_and_media():
    report = build_diagnostic_report(
        views=["A4C"],
        measurements={"ef": 60.0},
        draft_report="",
        study_instance_uid="1.2.840.113619.2.55.3.604688119.971",
    )
    assert any(i["system"] == "urn:dicom:uid" for i in report["identifier"])
    assert "media" in report and len(report["media"]) == 1


def test_encounter_reference():
    report = build_diagnostic_report(
        views=["A4C"],
        measurements={},
        draft_report="",
        encounter_id="E123",
    )
    assert report["encounter"]["reference"] == "Encounter/E123"
