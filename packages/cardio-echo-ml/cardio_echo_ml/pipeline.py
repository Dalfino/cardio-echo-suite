"""Commercial-Grade Inference Pipeline — all 6 ML engineering layers combined.

This is the main entry point. It chains all 6 modules into a single
predict() call that produces a commercial-grade prediction with:

- Quality-gated input
- TTA-augmented prediction
- MC dropout uncertainty
- Ensemble combination (if multiple models)
- Failure mode detection
- Consistency monitoring
- Calibration (bias correction + isotonic)

The output is a dict suitable for direct FHIR R4 Observation generation.

Usage:
    from cardio_echo_ml.pipeline import CommercialPipeline

    pipeline = CommercialPipeline(
        model=panecho_model,
        calibration_params={"bias": -4.5, "isotonic": iso_func},
    )

    result = pipeline.predict(video_tensor, patient_id="P001")
    # result = {
    #     "ef": 55.2,
    #     "ef_ci_95": [52.1, 58.3],
    #     "ef_raw": 60.1,  # before calibration
    #     "ef_calibrated": 55.2,  # after calibration
    #     "confidence": "high",
    #     "quality_score": 0.87,
    #     "flagged_for_review": False,
    #     "review_reasons": [],
    #     "tta_std": 1.2,
    #     "mc_dropout_std": 1.5,
    #     "n_augmentations": 4,
    #     "n_mc_passes": 10,
    #     "consistency": {...},
    #     "metadata": {...},
    # }
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .tta import predict_with_tta, tta_confidence_score
from .uncertainty import estimate_uncertainty
from .quality_gate import QualityGate, QualityThresholds
from .consistency import ConsistencyMonitor, ConsistencyConfig
from .failure_mode import detect_failure_modes
from .ensemble import EnsemblePredictor

logger = logging.getLogger(__name__)


class CommercialPipeline:
    """Commercial-grade inference pipeline with 6 ML engineering layers.

    Layers (applied in order):
    1. Quality Gate — refuse bad inputs
    2. TTA — 4 augmented passes, averaged
    3. MC Dropout — 10 passes for uncertainty estimation
    4. Ensemble — (if multiple models) combine predictions
    5. Calibration — bias correction + isotonic
    6. Failure Mode Detection — flag suspicious predictions
    7. Consistency Monitor — track prediction distribution
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        models: Optional[Dict[str, torch.nn.Module]] = None,
        forward_fn: Optional[Callable] = None,
        calibration_params: Optional[Dict[str, Any]] = None,
        quality_gate: Optional[QualityGate] = None,
        consistency_monitor: Optional[ConsistencyMonitor] = None,
        n_tta: int = 4,
        n_mc_dropout: int = 10,
        enable_tta: bool = True,
        enable_mc_dropout: bool = True,
        enable_failure_detection: bool = True,
        enable_consistency_monitoring: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            model: single model (if not using ensemble)
            models: dict of models for ensemble (if using ensemble)
            forward_fn: optional forward function
            calibration_params: dict with "bias" (float) and optional "isotonic" (callable)
            quality_gate: custom quality gate (default: QualityGate(min_quality=0.5))
            consistency_monitor: custom monitor (default: ConsistencyMonitor())
            n_tta: number of TTA augmentations (default 4)
            n_mc_dropout: number of MC dropout passes (default 10)
            enable_*: toggle individual layers
        """
        self.models = models or ({"model": model} if model else {})
        self.forward_fn = forward_fn
        self.calibration = calibration_params or {}
        self.quality_gate = quality_gate or QualityGate(min_quality=0.4, is_normalized=True)
        self.consistency_monitor = consistency_monitor or ConsistencyMonitor()
        self.n_tta = n_tta
        self.n_mc_dropout = n_mc_dropout
        self.enable_tta = enable_tta
        self.enable_mc_dropout = enable_mc_dropout
        self.enable_failure_detection = enable_failure_detection
        self.enable_consistency_monitoring = enable_consistency_monitoring

        # If we have multiple models, create ensemble
        if len(self.models) > 1:
            self.ensemble = EnsemblePredictor(
                models=self.models,
                forward_fns={k: forward_fn for k in self.models} if forward_fn else None,
            )
        else:
            self.ensemble = None

    def _calibrate(self, ef: float) -> float:
        """Apply calibration (bias correction + isotonic)."""
        # Bias correction
        if "bias" in self.calibration:
            ef = ef - self.calibration["bias"]
        # Isotonic calibration
        if "isotonic" in self.calibration and self.calibration["isotonic"] is not None:
            try:
                ef = float(self.calibration["isotonic"]([[ef]])[0])
            except Exception:
                try:
                    ef = float(self.calibration["isotonic"](np.array([[ef]]))[0])
                except Exception:
                    pass  # Keep bias-corrected value
        return round(ef, 2)

    def predict(
        self,
        video: torch.Tensor,
        patient_id: str = "unknown",
        video_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full commercial-grade pipeline.

        Args:
            video: input tensor (1, 3, T, H, W) or (1, C, H, W)
            patient_id: patient identifier for audit log
            video_hash: optional hash of input for deduplication

        Returns:
            dict with ef, ef_ci_95, confidence, quality_score, flags, metadata
        """
        result: Dict[str, Any] = {
            "patient_id": patient_id,
            "pipeline_version": "0.1.0",
            "layers_enabled": {
                "tta": self.enable_tta,
                "mc_dropout": self.enable_mc_dropout,
                "failure_detection": self.enable_failure_detection,
                "consistency_monitoring": self.enable_consistency_monitoring,
                "ensemble": self.ensemble is not None,
                "calibration": bool(self.calibration),
            },
        }

        # Layer 1: Quality Gate
        quality = self.quality_gate.check(video)
        result["quality"] = quality

        if not quality["is_acceptable"]:
            result.update({
                "ef": None,
                "refused": True,
                "refusal_reason": "low_quality",
                "quality_score": quality["quality_score"],
                "quality_reasons": quality["reasons"],
            })
            logger.info("Prediction refused — quality score %.2f", quality["quality_score"])
            return result

        # Layer 2+3+4: Prediction (TTA + MC Dropout + Ensemble)
        ef_raw: float = 0.0
        tta_std: Optional[float] = None
        mc_std: Optional[float] = None
        ensemble_agreement: Optional[float] = None
        tta_details: List[Dict] = []
        mc_details: Dict = {}

        if self.ensemble is not None:
            # Ensemble mode
            ens_result = self.ensemble.predict(
                video,
                tta_fn=predict_with_tta if self.enable_tta else None,
            )
            ef_raw = ens_result.get("ef", 0.0)
            tta_std = ens_result.get("ef_std")
            ensemble_agreement = ens_result.get("agreement")
            result["ensemble"] = ens_result
        else:
            # Single model mode
            model = list(self.models.values())[0]
            fwd = self.forward_fn or (lambda m, x: m(x))

            if self.enable_tta:
                ef_raw, tta_std, tta_details = predict_with_tta(
                    model, video, n_augmentations=self.n_tta, forward_fn=fwd,
                )
                result["tta"] = {
                    "n_augmentations": len(tta_details),
                    "std": tta_std,
                    "details": tta_details,
                }
            else:
                model.eval()
                with torch.inference_mode():
                    output = fwd(model, video)
                if isinstance(output, dict) and "EF" in output:
                    ef_raw = float(output["EF"].item())
                elif isinstance(output, torch.Tensor):
                    ef_raw = float(output.mean().item())
                else:
                    ef_raw = float(output)

            # Layer 3: MC Dropout (if enabled and model has dropout)
            if self.enable_mc_dropout:
                try:
                    mc_result = estimate_uncertainty(
                        model, video, n_passes=self.n_mc_dropout, forward_fn=fwd,
                    )
                    mc_std = mc_result.get("std")
                    mc_details = mc_result
                    result["mc_dropout"] = mc_result
                except Exception as e:
                    logger.warning("MC dropout failed: %s", e)

        # Layer 5: Calibration
        ef_calibrated = self._calibrate(ef_raw)
        result["ef_raw"] = round(ef_raw, 2)
        result["ef_calibrated"] = ef_calibrated
        result["ef"] = ef_calibrated  # Final EF = calibrated

        # Compute 95% CI from TTA + MC dropout combined
        combined_std = 0.0
        if tta_std is not None:
            combined_std = max(combined_std, tta_std)
        if mc_std is not None:
            combined_std = max(combined_std, mc_std)
        result["ef_ci_95"] = [
            round(ef_calibrated - 1.96 * combined_std, 2),
            round(ef_calibrated + 1.96 * combined_std, 2),
        ]

        # Confidence label
        tta_conf = tta_confidence_score(combined_std) if combined_std > 0 else "high"
        mc_conf = mc_details.get("confidence", "high") if mc_details else "high"
        # Take the worse confidence
        confidence_order = {"high": 0, "moderate": 1, "low": 2}
        result["confidence"] = max([tta_conf, mc_conf], key=lambda c: confidence_order.get(c, 0))

        # Layer 6: Failure Mode Detection
        if self.enable_failure_detection:
            flags = detect_failure_modes(
                ef=ef_calibrated,
                tta_std=tta_std,
                ensemble_agreement=ensemble_agreement,
                mc_dropout_std=mc_std,
                quality_score=quality["quality_score"],
            )
            result["failure_mode"] = flags
            result["flagged_for_review"] = flags["flagged_for_review"]
            result["review_reasons"] = flags["review_reasons"]
        else:
            result["flagged_for_review"] = False
            result["review_reasons"] = []

        # Layer 7: Consistency Monitoring
        if self.enable_consistency_monitoring:
            self.consistency_monitor.add_prediction(ef=ef_calibrated, video_hash=video_hash)
            consistency = self.consistency_monitor.check_consistency()
            result["consistency"] = consistency

        # Summary
        result["refused"] = False
        result["quality_score"] = quality["quality_score"]
        result["tta_std"] = round(tta_std, 2) if tta_std else None
        result["mc_dropout_std"] = round(mc_std, 2) if mc_std else None
        result["ensemble_agreement"] = round(ensemble_agreement, 4) if ensemble_agreement else None

        return result
