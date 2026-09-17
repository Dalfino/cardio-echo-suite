"""Test-Time Augmentation (TTA) — run model on augmented versions of input, average.

Commercial AI systems (DeepMind, Caption Health) use TTA to reduce prediction
variance by 30-50%. The idea: instead of predicting once on the original
input, predict on 4-8 augmented versions and average. The averaging cancels
out random errors.

Augmentations used (designed for cardiac echo — preserve clinical content):
1. Original (no augmentation)
2. Horizontal flip (mirror — heart is roughly symmetric for EF purposes)
3. Slight rotation (±5° — sonographers hold probe at slightly different angles)
4. Brightness jitter (±10% — different gain settings)

We do NOT use:
- Vertical flip (heart is NOT vertically symmetric)
- Large rotation (>10° — would distort clinical content)
- Color jitter (echo is grayscale, color is meaningless)
- Random crop (would cut off cardiac structures)

Usage:
    from cardio_echo_ml.tta import predict_with_tta

    ef, std = predict_with_tta(model, video_tensor, n_augmentations=4)
    # ef = 55.2 (averaged prediction)
    # std = 1.3 (standard deviation across augmentations — uncertainty)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def horizontal_flip(video: torch.Tensor) -> torch.Tensor:
    """Flip video horizontally (dim=-1 = width)."""
    return torch.flip(video, dims=[-1])


def rotate_video(video: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """Rotate video by a small angle (preserves cardiac content).

    Uses affine grid sampling. video shape: (1, 3, T, H, W) or (1, 3, H, W).
    """
    angle_rad = torch.tensor(angle_deg * np.pi / 180.0)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # Build rotation matrix
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
    ], dtype=torch.float32)

    if video.dim() == 5:
        # (1, 3, T, H, W) — apply to each frame
        B, C, T, H, W = video.shape
        # Reshape to (T, 3, H, W) for grid_sample
        frames = video.squeeze(0).permute(1, 0, 2, 3)  # (T, 3, H, W)
        grid = F.affine_grid(theta.unsqueeze(0).expand(T, -1, -1),
                             [T, 3, H, W], align_corners=False)
        rotated = F.grid_sample(frames, grid, align_corners=False, padding_mode='border')
        return rotated.permute(1, 0, 2, 3).unsqueeze(0)
    elif video.dim() == 4:
        # (1, 3, H, W)
        B, C, H, W = video.shape
        grid = F.affine_grid(theta.unsqueeze(0), [B, C, H, W], align_corners=False)
        return F.grid_sample(video, grid, align_corners=False, padding_mode='border')
    else:
        raise ValueError(f"Unexpected video dimensions: {video.dim()}")


def brightness_jitter(video: torch.Tensor, factor: float) -> torch.Tensor:
    """Multiply pixel intensities by factor (0.9 = darker, 1.1 = brighter)."""
    return torch.clamp(video * factor, 0.0, 1.0)


def get_augmentations(n: int = 4) -> List[Tuple[str, Callable]]:
    """Return a list of (name, augmentation_fn) tuples."""
    all_augs = [
        ("original", lambda x: x),
        ("hflip", horizontal_flip),
        ("rotate_+5", lambda x: rotate_video(x, 5.0)),
        ("rotate_-5", lambda x: rotate_video(x, -5.0)),
        ("brightness_+10", lambda x: brightness_jitter(x, 1.10)),
        ("brightness_-10", lambda x: brightness_jitter(x, 0.90)),
        ("rotate_+3_hflip", lambda x: horizontal_flip(rotate_video(x, 3.0))),
        ("rotate_-3_hflip", lambda x: horizontal_flip(rotate_video(x, -3.0))),
    ]
    if n >= len(all_augs):
        return all_augs
    return all_augs[:n]


def predict_with_tta(
    model: torch.nn.Module,
    video: torch.Tensor,
    n_augmentations: int = 4,
    forward_fn: Optional[Callable] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Run model with test-time augmentation.

    Args:
        model: PyTorch model (will be set to eval mode)
        video: input tensor (1, 3, T, H, W) or (1, 3, H, W)
        n_augmentations: number of augmented passes (default 4)
        forward_fn: optional callable(model, video) -> output.
                    If None, calls model(video) directly.

    Returns:
        (mean_prediction, std_prediction, augmentation_details)
        For EF regression: mean_prediction is the averaged EF value.
        std_prediction is the standard deviation across augmentations
        (a measure of uncertainty).
    """
    model.eval()
    if forward_fn is None:
        forward_fn = lambda m, x: m(x)

    augmentations = get_augmentations(n_augmentations)
    predictions: List[float] = []
    details: List[Dict[str, Any]] = []

    for aug_name, aug_fn in augmentations:
        try:
            augmented = aug_fn(video)
            with torch.inference_mode():
                output = forward_fn(model, augmented)

            # Extract scalar prediction from output
            if isinstance(output, dict):
                # PanEcho-style: dict of task -> tensor
                if "EF" in output:
                    pred = float(output["EF"].item())
                else:
                    # Take first value
                    first_key = next(iter(output))
                    val = output[first_key]
                    pred = float(val.mean().item())
            elif isinstance(output, torch.Tensor):
                pred = float(output.mean().item())
            else:
                pred = float(output)

            predictions.append(pred)
            details.append({
                "augmentation": aug_name,
                "prediction": pred,
                "success": True,
            })
        except Exception as e:
            logger.warning("Augmentation '%s' failed: %s", aug_name, e)
            details.append({
                "augmentation": aug_name,
                "prediction": None,
                "success": False,
                "error": str(e),
            })

    if not predictions:
        raise RuntimeError("All augmentations failed")

    mean_pred = float(np.mean(predictions))
    std_pred = float(np.std(predictions))

    return mean_pred, std_pred, details


def tta_confidence_score(std: float, threshold_low: float = 1.0, threshold_high: float = 3.0) -> str:
    """Convert TTA std to a confidence label.

    Low std = high confidence (all augmentations agree).
    High std = low confidence (augmentations disagree — model is uncertain).
    """
    if std < threshold_low:
        return "high"
    elif std < threshold_high:
        return "moderate"
    else:
        return "low"
