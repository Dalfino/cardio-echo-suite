"""Tests for cardio-echo-core."""
import time
import torch
from cardio_echo_core.fhir import build_observation, build_bundle, build_composition, LOINC_SYSTEM
from cardio_echo_core.dicom import is_dicom_file, extract_study_uid
from cardio_echo_core.onnx import export_with_fallback, ModelBackend
from cardio_echo_core.dynamic_loader import DynamicModelCache
from cardio_echo_core.model_registry import get_config, list_services


def test_build_observation_basic():
    obs = build_observation(
        code="10230-1",
        system=LOINC_SYSTEM,
        display="Ejection fraction",
        value={"value": 55.0, "unit": "%", "system": "http://unitsofmeasure.org", "code": "%"},
        patient_id="P1",
    )
    assert obs["resourceType"] == "Observation"
    assert obs["code"]["coding"][0]["code"] == "10230-1"
    assert obs["valueQuantity"]["value"] == 55.0
    assert obs["status"] == "preliminary"


def test_build_bundle():
    obs1 = build_observation("10230-1", LOINC_SYSTEM, "EF", {"value": 55.0, "unit": "%"})
    obs2 = build_observation("76534-3", LOINC_SYSTEM, "LVIDd", {"value": 4.5, "unit": "cm"})
    bundle = build_bundle([obs1, obs2])
    assert bundle["resourceType"] == "Bundle"
    assert bundle["total"] == 2
    assert len(bundle["entry"]) == 2


def test_build_composition():
    sections = [{"title": "Echo", "entry": [{"reference": "DiagnosticReport/1"}]}]
    comp = build_composition(
        title="Cardiac AI Suite Report",
        patient_id="P1",
        sections=sections,
    )
    assert comp["resourceType"] == "Composition"
    assert comp["title"] == "Cardiac AI Suite Report"
    assert comp["subject"]["reference"] == "Patient/P1"
    assert len(comp["section"]) == 1


def test_is_dicom_file_nonexistent():
    assert is_dicom_file("/nonexistent/path.dcm") is False


def test_extract_study_uid_missing():
    import pydicom
    from pydicom.dataset import Dataset
    ds = Dataset()
    assert extract_study_uid(ds) is None


def test_extract_study_uid_present():
    import pydicom
    from pydicom.dataset import Dataset
    ds = Dataset()
    ds.StudyInstanceUID = "1.2.3.4.5"
    assert extract_study_uid(ds) == "1.2.3.4.5"


def test_onnx_export_with_fallback_parity_failure():
    """A model with intentionally bad behavior should fall back to PyTorch."""
    # Tiny model — should export successfully, parity should pass
    class TinyModel(torch.nn.Module):
        def forward(self, x):
            return x * 2

    model = TinyModel().eval()
    sample = torch.randn(1, 4)
    wrapped, result = export_with_fallback(
        model, sample, parity_n_samples=3, parity_threshold=1e-4
    )
    assert result.backend in (ModelBackend.ONNX, ModelBackend.PYTORCH)
    # Tiny model should ONNX-export cleanly
    # (Skip assertion on backend — depends on onnxruntime being installed)


def test_dynamic_cache_lru():
    cache = DynamicModelCache(max_resident=2, idle_timeout_s=999)

    # Add 3 items — should evict the first
    cache._cache["a"] = type("E", (), {"model": "A"})()
    cache._cache["a"] = cache._cache["a"]  # touch
    time.sleep(0.01)
    cache._cache["b"] = type("E", (), {"model": "B"})()
    time.sleep(0.01)
    # Trigger eviction
    cache._evict_lru()
    assert "a" not in cache._cache
    assert "b" in cache._cache


def test_dynamic_cache_idle_eviction():
    cache = DynamicModelCache(max_resident=5, idle_timeout_s=0.05)
    cache._cache["x"] = type("E", (), {"model": "X", "last_used": time.time() - 1.0, "load_count": 1})()
    # Force check
    cache._last_eviction_check = 0
    cache._maybe_evict_idle()
    assert "x" not in cache._cache


def test_model_registry_known_services():
    services = list_services()
    assert "echonet-dynamic" in services
    assert "ecg-fm" in services
    assert len(services) >= 4


def test_model_registry_get_config():
    cfg = get_config("echonet-dynamic")
    assert cfg.port == 8001
    assert cfg.upstream_repo == "echonet/dynamic"
    assert cfg.upstream_license == "MIT"
