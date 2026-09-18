"""Ensemble Manager — combine multiple models, weight by confidence.

When you have multiple models (PanEcho + EchoNet-Dynamic + EchoNet-LVH),
running all of them on the same input and averaging their predictions
typically reduces MAE by 1-1.5 EF%.

The ensemble manager:
1. Runs each model on the input
2. Computes per-model confidence (via TTA std or model's own uncertainty)
3. Weights predictions by confidence (inverse-variance weighting)
4. Returns the weighted average + ensemble agreement score

Usage:
    from cardio_echo_ml.ensemble import EnsemblePredictor

    ensemble = EnsemblePredictor(models={
        "panecho": panecho_model,
        "echonet": echonet_model,
    })
    result = ensemble.predict(video_tensor)
    # result = {
    #     "ef": 55.2,
    #     "ef_std": 1.1,
    #     "per_model": {"panecho": 55.5, "echonet": 54.9},
    #     "weights": {"pancho": 0.6, "echonet": 0.4},
    #     "agreement": 0.92,
    # }
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Manages an ensemble of models with inverse-variance weighting."""

    def __init__(
        self,
        models: Dict[str, torch.nn.Module],
        forward_fns: Optional[Dict[str, Callable]] = None,
        min_confidence: float = 0.0,
    ):
        """Initialize ensemble.

        Args:
            models: dict of {model_name: model}
            forward_fns: optional dict of {model_name: forward_fn}.
                        If None, uses model(input) for each.
            min_confidence: if all models have std > this, flag as low confidence.
        """
        self.models = models
        self.forward_fns = forward_fns or {}
        self.min_confidence = min_confidence

    def predict_single(
        self,
        model_name: str,
        video: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run a single model and extract EF prediction."""
        model = self.models[model_name]
        forward_fn = self.forward_fns.get(model_name, lambda m, x: m(x))

        model.eval()
        try:
            with torch.inference_mode():
                output = forward_fn(model, video)

            if isinstance(output, dict) and "EF" in output:
                ef = float(output["EF"].item())
            elif isinstance(output, torch.Tensor):
                ef = float(output.mean().item())
            else:
                ef = float(output)

            return {"ef": ef, "success": True, "error": None}
        except Exception as e:
            return {"ef": None, "success": False, "error": str(e)}

    def predict(
        self,
        video: torch.Tensor,
        tta_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run ensemble prediction.

        Args:
            video: input tensor
            tta_fn: optional callable(model, video) -> (mean, std, details)
                    If provided, uses TTA for each model. If None, runs each
                    model once (no TTA).

        Returns:
            dict with ensemble prediction, per-model predictions, weights, agreement.
        """
        per_model: Dict[str, float] = {}
        per_model_std: Dict[str, float] = {}
        errors: Dict[str, str] = {}

        for name in self.models:
            if tta_fn is not None:
                try:
                    mean, std, _ = tta_fn(self.models[name], video)
                    per_model[name] = mean
                    per_model_std[name] = std
                except Exception as e:
                    errors[name] = str(e)
            else:
                result = self.predict_single(name, video)
                if result["success"]:
                    per_model[name] = result["ef"]
                    per_model_std[name] = 1.0  # default std if no TTA
                else:
                    errors[name] = result["error"]

        if not per_model:
            return {
                "ef": None,
                "error": "All models failed",
                "errors": errors,
            }

        # Inverse-variance weighting
        # Weight = 1 / variance = 1 / std^2
        weights: Dict[str, float] = {}
        total_weight = 0.0
        for name, std in per_model_std.items():
            variance = max(std ** 2, 0.01)  # floor to avoid div by zero
            w = 1.0 / variance
            weights[name] = w
            total_weight += w

        # Normalize weights
        for name in weights:
            weights[name] /= total_weight

        # Weighted average
        ef_ensemble = sum(per_model[n] * weights[n] for n in per_model)

        # Ensemble std (propagation of uncertainty)
        ef_std = np.sqrt(sum((weights[n] * per_model_std[n]) ** 2 for n in per_model))

        # Agreement score (1 - coefficient of variation)
        preds = list(per_model.values())
        if len(preds) > 1:
            cv = np.std(preds) / (abs(np.mean(preds)) + 1e-6)
            agreement = max(0.0, 1.0 - cv)
        else:
            agreement = 1.0

        return {
            "ef": round(float(ef_ensemble), 2),
            "ef_std": round(float(ef_std), 2),
            "per_model": {k: round(v, 2) for k, v in per_model.items()},
            "per_model_std": {k: round(v, 2) for k, v in per_model_std.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "agreement": round(float(agreement), 4),
            "n_models": len(per_model),
            "errors": errors if errors else None,
        }
