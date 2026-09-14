"""Tests for PanEcho FHIR mapping (no model weights needed)."""
import torch

from app.fhir import (
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    build_preread_report,
    map_predictions_to_bundle,
)


def test_task_count():
    # PanEcho outputs 39+ reporting tasks per the JAMA paper (we map 42 explicitly).
    assert len(ALL_TASKS) >= 39, f"Expected >=39 tasks, got {len(ALL_TASKS)}"
    assert len(REGRESSION_TASKS) >= 12
    assert len(CLASSIFICATION_TASKS) >= 25


def test_regression_observation():
    preds = {"EF": torch.tensor([58.0])}
    bundle = map_predictions_to_bundle(preds, patient_id="P1")
    assert bundle["resourceType"] == "Bundle"
    assert bundle["total"] == 1
    obs = bundle["entry"][0]["resource"]
    assert obs["code"]["coding"][0]["code"] == "10230-1"
    assert obs["valueQuantity"]["value"] == 58.0
    assert obs["valueQuantity"]["unit"] == "%"


def test_classification_observation():
    preds = {"AVStenosis": torch.tensor([[0.85, 0.10, 0.04, 0.01]])}
    bundle = map_predictions_to_bundle(preds)
    obs = bundle["entry"][0]["resource"]
    assert obs["valueCodeableConcept"]["coding"][0]["code"] == "none"
    assert len(obs["component"]) == 4  # 4 classes


def test_bundle_with_multiple_tasks():
    preds = {
        "EF": torch.tensor([42.0]),
        "LVIDd": torch.tensor([5.2]),
        "MVRegurgitation": torch.tensor([[0.6, 0.3, 0.08, 0.02]]),
        "PericardialEffusion": torch.tensor([[0.9, 0.07, 0.02, 0.01]]),
    }
    bundle = map_predictions_to_bundle(preds)
    assert bundle["total"] == 4


def test_preread_report_critical_ef():
    preds = {"EF": torch.tensor([25.0])}
    report = build_preread_report(preds)
    assert any(c["task"] == "EF" for c in report["critical_findings"])
    assert "EF = 25.0%" in report["summary"]


def test_preread_report_critical_class():
    preds = {"AVStenosis": torch.tensor([[0.05, 0.10, 0.15, 0.70]])}  # severe
    report = build_preread_report(preds)
    assert any("AVStenosis" in c["task"] for c in report["critical_findings"])


def test_preread_report_no_critical():
    preds = {"EF": torch.tensor([60.0]), "AVStenosis": torch.tensor([[0.95, 0.03, 0.01, 0.01]])}
    report = build_preread_report(preds)
    assert len(report["critical_findings"]) == 0


def test_study_uid_added_to_bundle():
    preds = {"EF": torch.tensor([50.0])}
    bundle = map_predictions_to_bundle(preds, study_instance_uid="1.2.3.4.5")
    assert any(i["value"] == "urn:oid:1.2.3.4.5" for i in bundle["identifier"])
