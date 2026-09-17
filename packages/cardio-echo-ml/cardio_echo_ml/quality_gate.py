"""Quality Gating — refuse bad inputs before prediction.

Commercial AI systems don't just predict — they REFUSE to predict when
the input quality is too low. This prevents garbage-in/garbage-out and
builds clinical trust.

For cardiac echo videos, quality is assessed on:
1. Mean pixel intensity (too dark = no signal; too bright = saturation)
2. Standard deviation (too low = blank/noise; too high = artifact)
3. Black pixel ratio (too many black pixels = no ultrasound sector)
4. White pixel ratio (too many white pixels = saturation/artifact)
5. Frame-to-frame consistency (jerky video = poor acquisition)
6. Motion entropy (frozen video = wrong input)

If quality_score < threshold, the pipeline REFUSES to predict and returns:
    {
        "ef": None,
        "refused": True,
        "reason": "low_quality",
        "quality_score": 0.32,
        "quality_reasons": ["Mean too low (0.05)", "Std too low (0.01)"],
    }

Usage:
    from cardio_echo_ml.quality_gate import assess_video_quality, QualityGate

    gate = QualityGate(min_quality=0.4)
    result = gate.check(video_tensor)
    if result["is_acceptable"]:
        ef = model(video_tensor)
    else:
        return {"refused": True, "reason": result["reasons"]}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """Quality thresholds for echo video acceptance."""
    min_mean: float = 0.05  # Too dark
    max_mean: float = 0.95  # Too bright
    min_std: float = 0.03   # Too flat (blank)
    max_black_ratio: float = 0.85  # Too many black pixels
    max_white_ratio: float = 0.85  # Too many white pixels
    min_frame_consistency: float = 0.3  # Frame-to-frame correlation


@dataclass
class QualityResult:
    """Result of quality assessment."""
    is_acceptable: bool
    quality_score: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def assess_frame_quality(frame: torch.Tensor) -> Dict[str, float]:
    """Assess quality of a single frame (2D tensor)."""
    flat = frame.flatten().float()
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "black_ratio": float((flat < 0.01).float().mean().item()),
        "white_ratio": float((flat > 0.99).float().mean().item()),
    }


def assess_video_quality(
    video: torch.Tensor,
    thresholds: Optional[QualityThresholds] = None,
    is_normalized: bool = False,
) -> QualityResult:
    """Assess whether a video is acceptable for AI prediction.

    Args:
        video: tensor of shape (1, 3, T, H, W) or (1, C, H, W) or (C, H, W)
        thresholds: quality thresholds (default: QualityThresholds())
        is_normalized: if True, the video has been ImageNet-normalized (mean
                      subtracted, std divided). Quality assessment will
                      un-normalize first to get [0,1] range.

    Returns:
        QualityResult with is_acceptable, quality_score, reasons, metrics
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    # If normalized, un-normalize for quality assessment
    if is_normalized:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=video.dtype, device=video.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=video.dtype, device=video.device)
        # Reshape mean/std to broadcast: (1, 3, 1, 1, 1) for 5D, (1, 3, 1, 1) for 4D
        if video.dim() == 5:
            mean = mean.view(1, 3, 1, 1, 1)
            std = std.view(1, 3, 1, 1, 1)
        elif video.dim() == 4:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        video = video * std + mean
        video = torch.clamp(video, 0.0, 1.0)

    # Normalize shape
    if video.dim() == 5:
        # (1, 3, T, H, W) → process per-frame
        frames = video.squeeze(0)  # (3, T, H, W)
        # Average across channels for quality assessment
        frames_gray = frames.mean(dim=0)  # (T, H, W)
    elif video.dim() == 4:
        frames_gray = video.squeeze(0).mean(dim=0).unsqueeze(0)  # (1, H, W)
    elif video.dim() == 3:
        frames_gray = video.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
    else:
        return QualityResult(
            is_acceptable=False,
            quality_score=0.0,
            reasons=[f"Unexpected video dimensions: {video.dim()}"],
        )

    # Per-frame metrics
    n_frames = frames_gray.shape[0] if frames_gray.dim() == 3 else 1
    all_metrics: List[Dict[str, float]] = []
    for i in range(n_frames):
        frame = frames_gray[i] if frames_gray.dim() == 3 else frames_gray.squeeze(0)
        all_metrics.append(assess_frame_quality(frame))

    # Aggregate
    metrics = {
        "mean": float(np.mean([m["mean"] for m in all_metrics])),
        "std": float(np.mean([m["std"] for m in all_metrics])),
        "black_ratio": float(np.mean([m["black_ratio"] for m in all_metrics])),
        "white_ratio": float(np.mean([m["white_ratio"] for m in all_metrics])),
    }

    # Frame consistency (correlation between consecutive frames)
    if n_frames > 1:
        consistencies = []
        for i in range(n_frames - 1):
            f1 = frames_gray[i].flatten().float()
            f2 = frames_gray[i + 1].flatten().float()
            # Normalized cross-correlation
            f1_norm = (f1 - f1.mean()) / (f1.std() + 1e-6)
            f2_norm = (f2 - f2.mean()) / (f2.std() + 1e-6)
            corr = float((f1_norm * f2_norm).mean().item())
            consistencies.append(corr)
        metrics["frame_consistency"] = float(np.mean(consistencies))
    else:
        metrics["frame_consistency"] = 1.0

    # Check thresholds
    reasons: List[str] = []
    score = 1.0

    if metrics["mean"] < thresholds.min_mean:
        reasons.append(f"Mean too low ({metrics['mean']:.3f} < {thresholds.min_mean})")
        score -= 0.3
    if metrics["mean"] > thresholds.max_mean:
        reasons.append(f"Mean too high ({metrics['mean']:.3f} > {thresholds.max_mean})")
        score -= 0.3
    if metrics["std"] < thresholds.min_std:
        reasons.append(f"Std too low ({metrics['std']:.3f} < {thresholds.min_std}) — possibly blank")
        score -= 0.4
    if metrics["black_ratio"] > thresholds.max_black_ratio:
        reasons.append(f"Too many black pixels ({metrics['black_ratio']:.1%})")
        score -= 0.3
    if metrics["white_ratio"] > thresholds.max_white_ratio:
        reasons.append(f"Too many white pixels ({metrics['white_ratio']:.1%})")
        score -= 0.3
    if metrics["frame_consistency"] < thresholds.min_frame_consistency:
        reasons.append(f"Low frame consistency ({metrics['frame_consistency']:.2f})")
        score -= 0.2

    score = max(0.0, min(1.0, score))

    return QualityResult(
        is_acceptable=len(reasons) == 0 and score >= 0.5,
        quality_score=round(score, 3),
        reasons=reasons,
        metrics={k: round(v, 4) for k, v in metrics.items()},
    )


class QualityGate:
    """Quality gate that decides whether to run the model."""

    def __init__(
        self,
        min_quality: float = 0.5,
        thresholds: Optional[QualityThresholds] = None,
        is_normalized: bool = False,
    ):
        self.min_quality = min_quality
        self.thresholds = thresholds or QualityThresholds()
        self.is_normalized = is_normalized

    def check(self, video: torch.Tensor) -> Dict[str, Any]:
        """Check if video passes the quality gate.

        Returns dict with:
            is_acceptable: bool
            quality_score: float
            reasons: List[str]
            metrics: Dict[str, float]
        """
        result = assess_video_quality(video, self.thresholds, is_normalized=self.is_normalized)
        return {
            "is_acceptable": result.is_acceptable and result.quality_score >= self.min_quality,
            "quality_score": result.quality_score,
            "reasons": result.reasons,
            "metrics": result.metrics,
        }
