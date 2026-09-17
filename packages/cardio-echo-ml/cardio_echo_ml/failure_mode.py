"""Failure Mode Detector — flag suspicious predictions for review.

Even with quality gating, some predictions slip through that are likely
wrong. This module flags predictions that:
1. Are at the distribution edge (EF < 15% or EF > 85%)
2. Have low TTA confidence (std > 3.0)
3. Have low ensemble agreement (agreement < 0.7)
4. Have high MC dropout uncertainty (std > 4.0)
5. Are inconsistent with clinical expectations (EF > 80% with low HR)

When flagged, the API response includes:
    "flagged_for_review": True,
    "review_reasons": ["EF at distribution edge (>85%)", "Low TTA confidence (std=3.5)"],

The cardiologist knows to double-check these.

Usage:
    from cardio_echo_ml.failure_mode import detect_failure_modes

    flags = detect_failure_modes(
        ef=55.2,
        tta_std=1.2,
        ensemble_agreement=0.92,
        mc_dropout_std=1.5,
        quality_score=0.85,
    )
    # flags = {
    #     "flagged_for_review": False,
    #     "review_reasons": [],
    #     "severity": "none",
    # }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Thresholds for failure mode detection
DEFAULT_THRESHOLDS = {
    "ef_min_normal": 15.0,        # EF < 15% is extremely rare → suspicious
    "ef_max_normal": 85.0,        # EF > 85% is extremely rare → suspicious
    "tta_std_high": 3.0,           # TTA std > 3.0 → low confidence
    "tta_std_critical": 5.0,       # TTA std > 5.0 → very low confidence
    "ensemble_agreement_low": 0.7, # Agreement < 0.7 → models disagree
    "mc_dropout_std_high": 4.0,    # MC std > 4.0 → high uncertainty
    "quality_score_low": 0.5,      # Quality < 0.5 → borderline input
}


def detect_failure_modes(
    ef: float,
    tta_std: Optional[float] = None,
    ensemble_agreement: Optional[float] = None,
    mc_dropout_std: Optional[float] = None,
    quality_score: Optional[float] = None,
    heart_rate: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Detect potential failure modes in a prediction.

    Args:
        ef: predicted EF value
        tta_std: TTA standard deviation (uncertainty)
        ensemble_agreement: ensemble agreement score (0-1)
        mc_dropout_std: MC dropout std (uncertainty)
        quality_score: input quality score (0-1)
        heart_rate: optional heart rate (for clinical consistency check)
        thresholds: custom thresholds (default: DEFAULT_THRESHOLDS)

    Returns:
        dict with flagged_for_review, review_reasons, severity
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: List[str] = []
    severity_score = 0  # 0 = none, 1 = low, 2 = moderate, 3 = high

    # 1. Distribution edge check
    if ef < t["ef_min_normal"]:
        reasons.append(f"EF at distribution edge ({ef:.1f}% < {t['ef_min_normal']}%)")
        severity_score = max(severity_score, 2)
    if ef > t["ef_max_normal"]:
        reasons.append(f"EF at distribution edge ({ef:.1f}% > {t['ef_max_normal']}%)")
        severity_score = max(severity_score, 2)

    # 2. TTA confidence check
    if tta_std is not None:
        if tta_std > t["tta_std_critical"]:
            reasons.append(f"Very low TTA confidence (std={tta_std:.2f})")
            severity_score = max(severity_score, 3)
        elif tta_std > t["tta_std_high"]:
            reasons.append(f"Low TTA confidence (std={tta_std:.2f})")
            severity_score = max(severity_score, 1)

    # 3. Ensemble agreement check
    if ensemble_agreement is not None:
        if ensemble_agreement < t["ensemble_agreement_low"]:
            reasons.append(f"Low ensemble agreement ({ensemble_agreement:.2f})")
            severity_score = max(severity_score, 2)

    # 4. MC dropout uncertainty check
    if mc_dropout_std is not None:
        if mc_dropout_std > t["mc_dropout_std_high"]:
            reasons.append(f"High MC dropout uncertainty (std={mc_dropout_std:.2f})")
            severity_score = max(severity_score, 2)

    # 5. Quality score check
    if quality_score is not None:
        if quality_score < t["quality_score_low"]:
            reasons.append(f"Low input quality (score={quality_score:.2f})")
            severity_score = max(severity_score, 1)

    # 6. Clinical consistency check
    if heart_rate is not None:
        # EF > 80% with HR < 50 is clinically unusual (would suggest hyperdynamic
        # LV, but bradycardia usually causes normal EF, not hyperdynamic)
        if ef > 80 and heart_rate < 50:
            reasons.append(f"Clinically inconsistent: EF={ef:.1f}% with HR={heart_rate:.0f} (bradycardia)")
            severity_score = max(severity_score, 2)

    severity = "none"
    if severity_score >= 3:
        severity = "high"
    elif severity_score >= 2:
        severity = "moderate"
    elif severity_score >= 1:
        severity = "low"

    return {
        "flagged_for_review": len(reasons) > 0,
        "review_reasons": reasons,
        "severity": severity,
        "severity_score": severity_score,
    }
