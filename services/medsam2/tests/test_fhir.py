"""Tests for MedSAM2 FHIR builder (no model required)."""
from app.fhir import build_segmentation_observation


def test_segmentation_observation_basic():
    obs = build_segmentation_observation(
        structure="left ventricle",
        volume_ml=125.5,
        n_slices=12,
        patient_id="P1",
    )
    assert obs["resourceType"] == "Observation"
    assert "left ventricle" in obs["code"]["text"]
    assert obs["status"] == "preliminary"
    # 2 components: volume + n_slices
    assert len(obs["component"]) == 2


def test_segmentation_observation_with_surface_area():
    obs = build_segmentation_observation(
        structure="aorta",
        volume_ml=45.0,
        surface_area_mm2=5400.0,
        n_slices=20,
    )
    components = obs["component"]
    assert len(components) == 3
    values = [c["valueQuantity"]["value"] for c in components]
    assert 45.0 in values
    assert 5400.0 in values
    assert 20 in values
