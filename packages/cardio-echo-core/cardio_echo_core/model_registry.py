"""Per-model backend flags and version pinning.

Defines which backend (pytorch / onnx / int8) each service should use, and
pins upstream model versions for reproducibility.

This file is the single source of truth — change a flag here and every
service picks it up at next restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .onnx import ModelBackend


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one model service."""
    name: str
    port: int
    upstream_repo: str
    upstream_ref: str  # git tag / HF revision
    upstream_license: str
    upstream_citation: str
    preferred_backend: ModelBackend  # what we'd LIKE to use
    fallback_backend: ModelBackend   # what we fall back to if export/quantize fails
    quantize: bool = False           # try INT8 quantization?
    max_quantize_mae: float = 1.0    # EF percentage points (or relevant unit)


# Registry — pinned upstream versions
REGISTRY = {
    "echonet-dynamic": ModelConfig(
        name="echonet-dynamic",
        port=8001,
        upstream_repo="echonet/dynamic",
        upstream_ref="master@e01d02947b20604b0ba1a427f3cf6f4890dc4e9c",
        upstream_license="MIT",
        upstream_citation="Ouyang D et al., Nature 2020",
        preferred_backend=ModelBackend.ONNX,
        fallback_backend=ModelBackend.PYTORCH,
        quantize=False,
    ),
    "echoprime": ModelConfig(
        name="echoprime",
        port=8002,
        upstream_repo="digital-echo/EchoPrime",
        upstream_ref="main",
        upstream_license="Research use (see HF model card)",
        upstream_citation="EchoPrime preprint 2024",
        preferred_backend=ModelBackend.INT8,
        fallback_backend=ModelBackend.PYTORCH,
        quantize=True,
        max_quantize_mae=1.0,  # EF percentage points
    ),
    "panecho": ModelConfig(
        name="panecho",
        port=8003,
        upstream_repo="CarDS-Yale/PanEcho",
        upstream_ref="main@836c46ef6bc4eace50b7dc36704ecef2d33affde",
        upstream_license="See LICENSES.md",
        upstream_citation="Holste G et al., JAMA 2025",
        preferred_backend=ModelBackend.ONNX,
        fallback_backend=ModelBackend.PYTORCH,
        quantize=False,
    ),
    "ecg-fm": ModelConfig(
        name="ecg-fm",
        port=8004,
        upstream_repo="bman03/ECG-FM",
        upstream_ref="main",
        upstream_license="See HF model card",
        upstream_citation="McKeen SR et al., ECG-FM 2024",
        preferred_backend=ModelBackend.ONNX,
        fallback_backend=ModelBackend.PYTORCH,
        quantize=False,
    ),
    "medsam2": ModelConfig(
        name="medsam2",
        port=8005,
        upstream_repo="Bowang-lab/MedSAM2",
        upstream_ref="main",
        upstream_license="Apache-2.0",
        upstream_citation="MedSAM2 2024",
        preferred_backend=ModelBackend.ONNX,
        fallback_backend=ModelBackend.PYTORCH,
        quantize=False,
    ),
    "nnunet-cmr": ModelConfig(
        name="nnunet-cmr",
        port=8006,
        upstream_repo="MIC-DKFZ/nnUNet",
        upstream_ref="master",
        upstream_license="Apache-2.0",
        upstream_citation="Isensee F et al., Nature Methods 2021",
        preferred_backend=ModelBackend.ONNX,
        fallback_backend=ModelBackend.PYTORCH,
        quantize=False,
    ),
    "neurokit": ModelConfig(
        name="neurokit",
        port=8007,
        upstream_repo="neuropsychology/NeuroKit",
        upstream_ref="master",
        upstream_license="MIT",
        upstream_citation="Makowski D et al., NeuroKit2 2021",
        preferred_backend=ModelBackend.PYTORCH,  # not a model — pure Python
        fallback_backend=ModelBackend.PYTORCH,
        quantize=False,
    ),
}


def get_config(service_name: str) -> ModelConfig:
    """Get the configuration for a service. Raises KeyError if unknown."""
    if service_name not in REGISTRY:
        raise KeyError(f"Unknown service: {service_name}. Known: {list(REGISTRY.keys())}")
    return REGISTRY[service_name]


def list_services() -> list:
    """List all registered services."""
    return list(REGISTRY.keys())
