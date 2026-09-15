"""Tests for nnU-Net CMR FHIR builder (no model required)."""
from app.fhir import build_cmr_bundle, build_cmr_observation


def test_cmr_observation_basic():
    obs = build_cmr_observation(
        structure="lv_blood_pool",
        volume_ml=125.5,
        n_slices=12,
        patient_id="P1",
    )
    assert obs["resourceType"] == "Observation"
    assert "lv_blood_pool" in obs["code"]["text"]
    assert obs["status"] == "preliminary"
    assert len(obs["component"]) == 2  # volume + slices


def test_cmr_observation_with_mass():
    obs = build_cmr_observation(
        structure="lv_myocardium",
        volume_ml=85.0,
        mass_g=120.0,
        n_slices=12,
    )
    assert len(obs["component"]) == 3  # volume + slices + mass


def test_cmr_bundle():
    structures = {
        "lv_blood_pool": {"volume_ml": 125.5, "n_slices": 12},
        "lv_myocardium": {"volume_ml": 85.0, "mass_g": 120.0, "n_slices": 12},
        "rv_blood_pool": {"volume_ml": 95.0, "n_slices": 12},
    }
    bundle = build_cmr_bundle(structures=structures, patient_id="P1")
    assert bundle["resourceType"] == "Bundle"
    assert bundle["total"] == 3
    assert len(bundle["entry"]) == 3


def test_cmr_bundle_with_study_uid():
    bundle = build_cmr_bundle(
        structures={"lv_blood_pool": {"volume_ml": 100.0, "n_slices": 10}},
        study_instance_uid="1.2.3.4.5",
    )
    assert any(i["value"] == "urn:oid:1.2.3.4.5" for i in bundle["identifier"])
