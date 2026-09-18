"""Consistency Checker — detect stuck/duplicate/suspicious predictions.

Commercial AI systems monitor their own outputs for anomalies. If the model
starts outputting the same prediction every time, that's a bug. If predictions
cluster at the training mean, the model isn't actually learning from the input.

This module tracks a rolling window of predictions and flags:
1. Duplicate predictions (same EF to 2 decimal places → model stuck)
2. Mean reversion (all predictions within 5 EF% of training mean → not learning)
3. Distribution shift (recent predictions are very different from historical)
4. Variance collapse (std of recent predictions < 1.0 → suspiciously low)

Usage:
    from cardio_echo_ml.consistency import ConsistencyMonitor

    monitor = ConsistencyMonitor(window_size=50)
    monitor.add_prediction(ef=55.2, video_hash="abc123")
    report = monitor.check_consistency()
    # report = {
    #     "is_consistent": True,
    #     "duplicate_ratio": 0.02,
    #     "mean_reversion_score": 0.15,
    #     "variance_collapse": False,
    #     "n_predictions": 50,
    #     "alerts": [],
    # }
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyConfig:
    """Configuration for consistency monitoring."""
    window_size: int = 50
    duplicate_threshold: float = 0.10  # >10% duplicates = alert
    duplicate_precision: int = 1  # Round to 1 decimal place for duplicate check
    mean_reversion_range: float = 5.0  # Within 5 EF% of training mean
    mean_reversion_threshold: float = 0.50  # >50% in range = alert
    variance_collapse_threshold: float = 1.0  # std < 1.0 = alert
    distribution_shift_z: float = 3.0  # 3 sigma shift = alert


class ConsistencyMonitor:
    """Monitors prediction consistency over a rolling window."""

    def __init__(
        self,
        config: Optional[ConsistencyConfig] = None,
        training_mean: float = 55.0,  # PanEcho training mean EF
    ):
        self.config = config or ConsistencyConfig()
        self.training_mean = training_mean
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=self.config.window_size)
        self.historical_predictions: List[float] = []

    def add_prediction(self, ef: float, video_hash: Optional[str] = None) -> None:
        """Add a new prediction to the monitor."""
        self.predictions.append({
            "ef": ef,
            "video_hash": video_hash,
            "ef_rounded": round(ef, self.config.duplicate_precision),
        })
        self.historical_predictions.append(ef)
        # Keep historical list bounded
        if len(self.historical_predictions) > 1000:
            self.historical_predictions = self.historical_predictions[-500:]

    def check_consistency(self) -> Dict[str, Any]:
        """Check if recent predictions are consistent.

        Returns dict with:
            is_consistent: True if no alerts
            duplicate_ratio: fraction of duplicate predictions
            mean_reversion_score: fraction near training mean
            variance_collapse: True if variance suspiciously low
            distribution_shift: True if recent predictions shifted
            alerts: list of alert strings
        """
        if len(self.predictions) < 5:
            return {
                "is_consistent": True,
                "n_predictions": len(self.predictions),
                "alerts": ["Insufficient data for consistency check (<5 predictions)"],
            }

        efs = [p["ef"] for p in self.predictions]
        efs_rounded = [p["ef_rounded"] for p in self.predictions]

        # 1. Duplicate check
        unique_rounded = set(efs_rounded)
        duplicate_ratio = 1 - len(unique_rounded) / len(efs_rounded)

        # 2. Mean reversion check
        in_range = sum(1 for e in efs if abs(e - self.training_mean) <= self.config.mean_reversion_range)
        mean_reversion_score = in_range / len(efs)

        # 3. Variance collapse
        current_std = float(np.std(efs))
        variance_collapse = current_std < self.config.variance_collapse_threshold

        # 4. Distribution shift (compare recent to historical)
        distribution_shift = False
        if len(self.historical_predictions) > self.config.window_size:
            recent_mean = float(np.mean(efs))
            historical = self.historical_predictions[:-len(self.predictions)]
            hist_mean = float(np.mean(historical))
            hist_std = float(np.std(historical)) + 1e-6
            z_score = abs(recent_mean - hist_mean) / hist_std
            distribution_shift = z_score > self.config.distribution_shift_z

        # Build alerts
        alerts: List[str] = []
        if duplicate_ratio > self.config.duplicate_threshold:
            alerts.append(
                f"High duplicate ratio: {duplicate_ratio:.1%} (threshold: {self.config.duplicate_threshold:.0%})"
            )
        if mean_reversion_score > self.config.mean_reversion_threshold:
            alerts.append(
                f"Mean reversion: {mean_reversion_score:.1%} of predictions within "
                f"{self.config.mean_reversion_range} EF% of training mean"
            )
        if variance_collapse:
            alerts.append(
                f"Variance collapse: std={current_std:.2f} < {self.config.variance_collapse_threshold}"
            )
        if distribution_shift:
            alerts.append("Distribution shift detected — recent predictions differ from historical")

        return {
            "is_consistent": len(alerts) == 0,
            "duplicate_ratio": round(duplicate_ratio, 4),
            "mean_reversion_score": round(mean_reversion_score, 4),
            "variance_collapse": variance_collapse,
            "current_std": round(current_std, 2),
            "distribution_shift": distribution_shift,
            "n_predictions": len(self.predictions),
            "alerts": alerts,
        }
