"""Uncertainty Estimation — Monte Carlo dropout for confidence intervals.

Commercial AI systems don't just output a point estimate ("EF = 55%").
They output a confidence interval ("EF = 55% ± 3%, 95% CI: 52-58%").

MC Dropout technique:
1. Enable dropout at inference time (normally disabled)
2. Run model N times (typically 5-20) on the same input
3. Each run uses different random dropout → slightly different predictions
4. Compute mean + std of the N predictions
5. 95% CI = mean ± 1.96 × std

This is the standard technique from Gal & Ghahramani (2016) and is used
by DeepMind, Caption Health, and most FDA-cleared AI devices.

Usage:
    from cardio_echo_ml.uncertainty import estimate_uncertainty

    result = estimate_uncertainty(model, video, n_passes=10)
    # result = {
    #     "mean": 55.2,
    #     "std": 1.5,
    #     "ci_95": [52.3, 58.1],
    #     "confidence": "high",  # high if std < 2, moderate if < 4, low if >= 4
    #     "n_passes": 10,
    # }
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def enable_dropout(model: torch.nn.Module) -> int:
    """Enable dropout layers at inference time (for MC dropout).

    Returns the number of dropout layers enabled.
    """
    n_dropout = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()  # Enable dropout
            n_dropout += 1
    return n_dropout


def disable_dropout(model: torch.nn.Module) -> None:
    """Disable dropout layers (normal inference mode)."""
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.eval()


def estimate_uncertainty(
    model: torch.nn.Module,
    video: torch.Tensor,
    n_passes: int = 10,
    forward_fn: Optional[Callable] = None,
    confidence_thresholds: tuple = (2.0, 4.0),
) -> Dict[str, Any]:
    """Estimate prediction uncertainty via MC dropout.

    Args:
        model: PyTorch model (must have dropout layers)
        video: input tensor
        n_passes: number of forward passes (default 10)
        forward_fn: optional callable(model, video) -> output
        confidence_thresholds: (low_std, high_std) for confidence labels

    Returns:
        dict with mean, std, ci_95, confidence label, n_passes
    """
    if forward_fn is None:
        forward_fn = lambda m, x: m(x)

    # Enable dropout
    n_dropout = enable_dropout(model)
    if n_dropout == 0:
        logger.warning("Model has no dropout layers — MC dropout will not add uncertainty.")
        logger.warning("Falling back to single-pass prediction.")
        disable_dropout(model)
        with torch.inference_mode():
            output = forward_fn(model, video)
        if isinstance(output, dict) and "EF" in output:
            ef = float(output["EF"].item())
        elif isinstance(output, torch.Tensor):
            ef = float(output.mean().item())
        else:
            ef = float(output)
        return {
            "mean": ef,
            "std": 0.0,
            "ci_95": [ef, ef],
            "confidence": "high",
            "n_passes": 1,
            "note": "No dropout layers — single-pass prediction",
        }

    predictions: List[float] = []

    for i in range(n_passes):
        try:
            with torch.inference_mode():
                output = forward_fn(model, video)

            if isinstance(output, dict) and "EF" in output:
                pred = float(output["EF"].item())
            elif isinstance(output, torch.Tensor):
                pred = float(output.mean().item())
            else:
                pred = float(output)

            predictions.append(pred)
        except Exception as e:
            logger.warning("MC dropout pass %d failed: %s", i + 1, e)

    # Restore normal eval mode
    disable_dropout(model)

    if not predictions:
        raise RuntimeError("All MC dropout passes failed")

    mean = float(np.mean(predictions))
    std = float(np.std(predictions))
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std

    # Confidence label
    low_thr, high_thr = confidence_thresholds
    if std < low_thr:
        confidence = "high"
    elif std < high_thr:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "confidence": confidence,
        "n_passes": len(predictions),
        "n_dropout_layers": n_dropout,
        "all_predictions": [round(p, 2) for p in predictions],
    }
