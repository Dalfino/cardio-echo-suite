"""Accuracy tier 2 helpers — consistency tests, calibration, quality gating.

Three accuracy defenses that we can build without real clinical data:

1. **Consistency test** — feed the same input N times, assert max variance
   is below a threshold. Catches non-determinism bugs.

2. **Temperature scaling calibration** — fits a single temperature parameter
   on a validation set to make model confidences match observed accuracy.
   Useful for ECG-FM arrhythmia probabilities.

3. **Image quality gating** — basic checks on the input image/video:
   - Mean pixel intensity in expected range
   - Standard deviation above noise floor
   - No all-black / all-white frames
   If quality is low, flag the prediction as "low confidence" without
   blocking it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. Consistency test
# ----------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    n_runs: int
    max_variance: float
    mean_variance: float
    is_consistent: bool
    threshold: float
    outputs: List[Any] = None


def consistency_test(
    predict_fn: Callable[[], Any],
    n_runs: int = 10,
    threshold: float = 0.005,
    reduction: Optional[Callable[[Any], float]] = None,
) -> ConsistencyResult:
    """Run predict_fn N times on the same input and check output variance.

    Args:
        predict_fn: callable that returns a model output (no args).
        n_runs: how many times to run.
        threshold: max allowed variance (in reduction units) for `is_consistent`.
        reduction: callable to reduce the output to a single float. If None,
            tries `output.mean()` (works for tensors).

    Returns:
        ConsistencyResult with n_runs, max_variance, mean_variance, is_consistent.
    """
    outputs: List[Any] = []
    for _ in range(n_runs):
        out = predict_fn()
        outputs.append(out)

    # Reduce each output to a scalar
    scalars: List[float] = []
    for out in outputs:
        if reduction is not None:
            s = float(reduction(out))
        elif isinstance(out, torch.Tensor):
            s = float(out.mean().item())
        elif isinstance(out, dict):
            # Take mean of first tensor value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    s = float(v.mean().item())
                    break
            else:
                s = 0.0
        elif isinstance(out, (int, float)):
            s = float(out)
        else:
            s = 0.0
        scalars.append(s)

    arr = np.array(scalars, dtype=np.float64)
    max_var = float(np.var(arr))
    mean_var = float(np.mean(np.abs(arr - arr.mean())))

    return ConsistencyResult(
        n_runs=n_runs,
        max_variance=max_var,
        mean_variance=mean_var,
        is_consistent=max_var < threshold,
        threshold=threshold,
        outputs=outputs,
    )


# ----------------------------------------------------------------------
# 2. Temperature scaling calibration
# ----------------------------------------------------------------------

@dataclass
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float
    n_samples: int


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.1,
) -> CalibrationResult:
    """Fit a single temperature parameter to minimize NLL on a validation set.

    Args:
        logits: (N, C) pre-softmax logits.
        labels: (N,) integer class labels.
        max_iter: optimization iterations.
        lr: learning rate.

    Returns:
        CalibrationResult with optimal temperature and NLL before/after.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    nll_criterion = torch.nn.CrossEntropyLoss()

    # Before calibration
    with torch.no_grad():
        nll_before = float(nll_criterion(logits, labels).item())

    # Fit temperature
    temperature = torch.ones(1, requires_grad=True, dtype=torch.float32)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        nll_after = float(nll_criterion(logits / temperature, labels).item())

    return CalibrationResult(
        temperature=float(temperature.item()),
        nll_before=nll_before,
        nll_after=nll_after,
        n_samples=int(logits.shape[0]),
    )


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits and return calibrated probabilities."""
    return torch.softmax(logits / temperature, dim=-1)


# ----------------------------------------------------------------------
# 3. Image quality gating
# ----------------------------------------------------------------------

@dataclass
class QualityGate:
    is_acceptable: bool
    quality_score: float  # 0-1, higher is better
    reasons: List[str]


def assess_image_quality(
    image: torch.Tensor,
    min_mean: float = 0.05,
    max_mean: float = 0.95,
    min_std: float = 0.02,
    max_black_ratio: float = 0.8,
    max_white_ratio: float = 0.8,
) -> QualityGate:
    """Assess whether an image (or video frame) is acceptable for AI inference.

    Args:
        image: tensor of shape (..., C, H, W) or (..., H, W), float, range [0, 1].
        min_mean, max_mean: acceptable mean pixel intensity range.
        min_std: minimum standard deviation (rejects flat noise / blank frames).
        max_black_ratio, max_white_ratio: max fraction of pixels at 0 or 1.

    Returns:
        QualityGate with is_acceptable flag, quality_score (0-1), and reasons list.
    """
    if image.dim() < 2:
        return QualityGate(False, 0.0, ["Input too small (dim < 2)"])

    flat = image.flatten().float()
    mean = float(flat.mean().item())
    std = float(flat.std().item())
    black_ratio = float((flat < 0.01).float().mean().item())
    white_ratio = float((flat > 0.99).float().mean().item())

    reasons: List[str] = []
    if mean < min_mean:
        reasons.append(f"Mean too low ({mean:.3f} < {min_mean})")
    if mean > max_mean:
        reasons.append(f"Mean too high ({mean:.3f} > {max_mean})")
    if std < min_std:
        reasons.append(f"Std too low ({std:.3f} < {min_std}) — possibly blank")
    if black_ratio > max_black_ratio:
        reasons.append(f"Too many black pixels ({black_ratio:.1%})")
    if white_ratio > max_white_ratio:
        reasons.append(f"Too many white pixels ({white_ratio:.1%})")

    # Quality score: heuristic combination
    score = 1.0
    if mean < min_mean or mean > max_mean:
        score -= 0.3
    if std < min_std:
        score -= 0.4
    if black_ratio > max_black_ratio or white_ratio > max_white_ratio:
        score -= 0.3
    score = max(0.0, min(1.0, score))

    return QualityGate(
        is_acceptable=len(reasons) == 0,
        quality_score=score,
        reasons=reasons,
    )


def assess_video_quality(
    video: torch.Tensor,
    max_bad_frames_ratio: float = 0.2,
) -> QualityGate:
    """Assess a video by checking each frame.

    Args:
        video: tensor of shape (1, T, C, H, W) or (T, C, H, W), float [0, 1].
        max_bad_frames_ratio: if more than this fraction of frames are bad,
            the whole video is rejected.

    Returns:
        QualityGate aggregated across all frames.
    """
    if video.dim() == 5:
        video = video.squeeze(0)
    if video.dim() == 4:
        # (T, C, H, W) — iterate frames
        pass
    elif video.dim() == 3:
        # Single frame (C, H, W)
        video = video.unsqueeze(0)
    else:
        return QualityGate(False, 0.0, [f"Unexpected video dim: {video.dim()}"])

    n_frames = video.shape[0]
    bad_frames = 0
    quality_scores: List[float] = []
    reasons_all: List[str] = []

    for i in range(n_frames):
        gate = assess_image_quality(video[i])
        quality_scores.append(gate.quality_score)
        if not gate.is_acceptable:
            bad_frames += 1
            reasons_all.extend(f"Frame {i}: {r}" for r in gate.reasons)

    bad_ratio = bad_frames / max(n_frames, 1)
    is_acceptable = bad_ratio <= max_bad_frames_ratio
    avg_score = float(np.mean(quality_scores)) if quality_scores else 0.0

    if bad_ratio > max_bad_frames_ratio:
        reasons_all.append(
            f"Too many bad frames: {bad_frames}/{n_frames} ({bad_ratio:.1%})"
        )

    return QualityGate(
        is_acceptable=is_acceptable,
        quality_score=avg_score,
        reasons=reasons_all[:10],  # cap for log readability
    )
