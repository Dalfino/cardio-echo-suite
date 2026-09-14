"""Tests for the FHIR Observation builders (no model needed)."""
from app.fhir import build_ef_observation, build_segmentation_observation


def test_ef_observation_basic():
    obs = build_ef_observation(ef_percent=60.0, patient_id="P1")
    assert obs["resourceType"] == "Observation"
    assert obs["code"]["coding"][0]["code"] == "10230-1"
    assert obs["valueQuantity"]["value"] == 60.0
    assert obs["valueQuantity"]["unit"] == "%"
    assert obs["status"] == "final"
    assert obs["subject"]["reference"] == "Patient/P1"
    assert "Normal" in obs["interpretation"][0]["text"]


def test_ef_observation_reduced():
    obs = build_ef_observation(ef_percent=35.0)
    interp = obs["interpretation"][0]["text"]
    assert "Moderately reduced" in interp


def test_ef_observation_severe():
    obs = build_ef_observation(ef_percent=20.0)
    interp = obs["interpretation"][0]["text"]
    assert "Severely reduced" in interp


def test_ef_observation_has_performer_device():
    obs = build_ef_observation(ef_percent=50.0)
    assert obs["performer"][0]["reference"] == "#ai-performer"
    device = obs["contained"][0]
    assert device["resourceType"] == "Device"
    assert any(n["name"] == "EchoNet-Dynamic AI" for n in device["deviceName"])


def test_segmentation_observation():
    obs = build_segmentation_observation(
        n_frames_segmented=64,
        mean_lv_area_px=1234.5,
        patient_id="P1",
    )
    assert obs["resourceType"] == "Observation"
    components = obs["component"]
    assert any(c["valueQuantity"]["value"] == 64 for c in components)
    assert any(abs(c["valueQuantity"]["value"] - 1234.5) < 0.01 for c in components)
