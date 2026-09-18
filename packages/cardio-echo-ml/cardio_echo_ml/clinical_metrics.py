"""Clinical Metrics — sensitivity, specificity, AUROC for classification tasks.

In medicine, not all errors are equal:
- False Negative (missed disease) = patient sent home → potentially fatal
- False Positive (false alarm) = unnecessary procedures → cost + anxiety

This module provides clinical-grade metrics for PanEcho's 26+ classification
tasks, including:
- Sensitivity (Recall) = TP / (TP + FN) — "did we catch all the sick patients?"
- Specificity = TN / (TN + FP) — "did we spare the healthy patients?"
- AUROC = area under ROC curve — overall discrimination
- Cross-Entropy Loss = proper loss for classification training
- Per-class F1, Precision, Recall

Also provides:
- Asymmetric loss weighting (penalize FN 5× more than FP)
- Clinical threshold optimization (find threshold that maximizes sensitivity
  at fixed specificity, e.g., ≥95% sensitivity for STEMI detection)
- Risk-stratified reporting (different thresholds for different risk levels)

Usage:
    from cardio_echo_ml.clinical_metrics import ClinicalMetrics

    metrics = ClinicalMetrics()

    # For a binary classification task (e.g., AFib detection)
    result = metrics.evaluate_binary(
        y_true=[1, 0, 1, 0, 1, 0, 1, 0],
        y_prob=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        task_name="AFib",
        threshold=0.5,
    )
    # result = {
    #     "sensitivity": 1.0,  # caught all 4 AFib cases
    #     "specificity": 1.0,  # correctly identified all 4 healthy
    #     "auroc": 1.0,
    #     "f1": 1.0,
    #     "fn_count": 0,  # NO missed cases
    #     "fp_count": 0,  # NO false alarms
    #     "clinical_risk": "low",
    # }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ClinicalThreshold:
    """Clinical thresholds for different risk levels."""
    task_name: str
    # Threshold for "any abnormality" (high sensitivity, lower specificity)
    screening_threshold: float = 0.3
    # Threshold for "likely abnormal" (balanced)
    diagnostic_threshold: float = 0.5
    # Threshold for "definitely abnormal" (high specificity, lower sensitivity)
    confirmatory_threshold: float = 0.7
    # Minimum acceptable sensitivity (for FN-critical tasks like STEMI)
    min_sensitivity: float = 0.90
    # Minimum acceptable specificity (for FP-costly tasks like cath lab activation)
    min_specificity: float = 0.80


# Clinical threshold presets for PanEcho tasks
CLINICAL_THRESHOLDS: Dict[str, ClinicalThreshold] = {
    # FN-critical: missing these = patient harm or death
    "TamponadePhysiology": ClinicalThreshold(
        task_name="TamponadePhysiology",
        screening_threshold=0.2,  # Very low — never miss tamponade
        min_sensitivity=0.95,  # MUST catch ≥95% of tamponade
    ),
    "STEMI": ClinicalThreshold(
        task_name="STEMI",
        screening_threshold=0.15,  # Very low — never miss STEMI
        min_sensitivity=0.95,
    ),
    "LVThrombus": ClinicalThreshold(
        task_name="LVThrombus",
        screening_threshold=0.2,
        min_sensitivity=0.90,
    ),
    "AorticDissection": ClinicalThreshold(
        task_name="AorticDissection",
        screening_threshold=0.15,
        min_sensitivity=0.95,
    ),
    # FP-costly: false alarm = unnecessary procedures
    "AVStenosis_severe": ClinicalThreshold(
        task_name="AVStenosis_severe",
        confirmatory_threshold=0.8,  # High — don't flag severe AS unless certain
        min_specificity=0.90,
    ),
    # Balanced: standard thresholds
    "AFib": ClinicalThreshold(
        task_name="AFib",
        diagnostic_threshold=0.5,
        min_sensitivity=0.85,
        min_specificity=0.85,
    ),
}


class ClinicalMetrics:
    """Clinical-grade metrics for medical AI classification tasks."""

    def evaluate_binary(
        self,
        y_true: List[int],
        y_prob: List[float],
        task_name: str = "",
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate binary classification with clinical metrics.

        Args:
            y_true: ground truth labels (0 = healthy, 1 = disease)
            y_prob: predicted probabilities (0.0 to 1.0)
            task_name: name of the task (for clinical threshold lookup)
            threshold: decision threshold (default 0.5)

        Returns:
            dict with sensitivity, specificity, AUROC, F1, FN/FP counts,
            clinical risk assessment
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        # AUROC (using rank-based method)
        auroc = self._compute_auroc(y_true, y_prob)

        # Cross-entropy loss
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        cross_entropy = -np.mean(
            y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)
        )

        # Clinical risk assessment
        clinical_thresholds = CLINICAL_THRESHOLDS.get(task_name)
        risk = self._assess_clinical_risk(
            sensitivity, specificity, fn, fp, clinical_thresholds
        )

        return {
            "task_name": task_name,
            "threshold": threshold,
            "n_total": len(y_true),
            "n_positive": int(np.sum(y_true)),
            "n_negative": int(np.sum(1 - y_true)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": round(sensitivity, 4),  # Recall
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "auroc": round(auroc, 4),
            "cross_entropy": round(float(cross_entropy), 4),
            "fn_count": fn,  # False negatives — MISSED DISEASE
            "fp_count": fp,  # False positives — FALSE ALARMS
            "clinical_risk": risk,
        }

    def _compute_auroc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute AUROC using rank-based method."""
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            # Fallback: manual computation
            n_pos = np.sum(y_true == 1)
            n_neg = np.sum(y_true == 0)
            if n_pos == 0 or n_neg == 0:
                return 0.5
            # Rank-based
            order = np.argsort(-y_prob)  # Descending
            ranks = np.zeros_like(y_prob)
            ranks[order] = np.arange(1, len(y_prob) + 1)
            sum_ranks_pos = np.sum(ranks[y_true == 1])
            return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def _assess_clinical_risk(
        self,
        sensitivity: float,
        specificity: float,
        fn: int,
        fp: int,
        thresholds: Optional[ClinicalThreshold] = None,
    ) -> str:
        """Assess clinical risk level based on sensitivity/specificity."""
        if thresholds:
            if sensitivity < thresholds.min_sensitivity:
                return "HIGH_RISK_FN"  # Too many missed cases
            if specificity < thresholds.min_specificity:
                return "HIGH_RISK_FP"  # Too many false alarms

        if fn > 0 and sensitivity < 0.8:
            return "HIGH_RISK_FN"
        if fp > 5 and specificity < 0.7:
            return "MODERATE_RISK_FP"
        if sensitivity >= 0.9 and specificity >= 0.8:
            return "LOW_RISK"
        return "MODERATE_RISK"

    def find_optimal_threshold(
        self,
        y_true: List[int],
        y_prob: List[float],
        min_sensitivity: float = 0.90,
        min_specificity: float = 0.80,
    ) -> Dict[str, Any]:
        """Find the threshold that maximizes sensitivity at min specificity.

        In medicine, we often want: "catch ≥90% of disease (sensitivity)
        while keeping false alarms <20% (specificity ≥80%)."

        This finds the threshold that achieves that goal.
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        best_threshold = 0.5
        best_sensitivity = 0.0
        best_specificity = 0.0

        for t in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            # We want: sensitivity >= min_sensitivity AND specificity >= min_specificity
            if sens >= min_sensitivity and spec >= min_specificity:
                # Prefer higher sensitivity (catch more disease)
                if sens > best_sensitivity:
                    best_threshold = float(t)
                    best_sensitivity = sens
                    best_specificity = spec

        return {
            "optimal_threshold": round(best_threshold, 2),
            "sensitivity_at_optimal": round(best_sensitivity, 4),
            "specificity_at_optimal": round(best_specificity, 4),
            "meets_criteria": best_sensitivity >= min_sensitivity
                            and best_specificity >= min_specificity,
        }

    def evaluate_multiclass(
        self,
        y_true: List[int],
        y_prob: List[List[float]],  # (N, C) probabilities
        class_names: List[str],
        task_name: str = "",
    ) -> Dict[str, Any]:
        """Evaluate multi-class classification (e.g., AVStenosis: none/mild/moderate/severe).

        Returns per-class sensitivity/specificity + macro-averaged metrics.
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = np.argmax(y_prob, axis=1)
        n_classes = len(class_names)

        per_class: Dict[str, Any] = {}
        for i, name in enumerate(class_names):
            y_true_bin = (y_true == i).astype(int)
            y_pred_bin = (y_pred == i).astype(int)

            tp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 1)))
            fp = int(np.sum((y_pred_bin == 1) & (y_true_bin == 0)))
            tn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 0)))
            fn = int(np.sum((y_pred_bin == 0) & (y_true_bin == 1)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            per_class[name] = {
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

        # Macro-averaged
        macro_sens = np.mean([per_class[c]["sensitivity"] for c in class_names])
        macro_spec = np.mean([per_class[c]["specificity"] for c in class_names])

        # Cross-entropy loss
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        cross_entropy = -np.mean(np.log(y_prob_clipped[np.arange(len(y_true)), y_true]))

        return {
            "task_name": task_name,
            "n_classes": n_classes,
            "class_names": class_names,
            "per_class": per_class,
            "macro_sensitivity": round(float(macro_sens), 4),
            "macro_specificity": round(float(macro_spec), 4),
            "cross_entropy": round(float(cross_entropy), 4),
            "accuracy": round(float(np.mean(y_pred == y_true)), 4),
        }


def asymmetric_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> torch.Tensor:
    """Asymmetric loss — penalize false negatives more than false positives.

    In medicine, missing a disease (FN) is typically 5-10× worse than
    a false alarm (FP). This loss function reflects that.

    Args:
        y_true: ground truth (0 or 1)
        y_pred: predicted probabilities (0 to 1)
        fn_weight: weight for false negatives (default 5× — missing disease is 5× worse)
        fp_weight: weight for false positives (default 1×)

    Returns:
        weighted cross-entropy loss
    """
    # Standard binary cross-entropy
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')

    # Apply asymmetric weighting
    # When y_true=1 (disease) and y_pred<0.5 → false negative → weight = fn_weight
    # When y_true=0 (healthy) and y_pred>=0.5 → false positive → weight = fp_weight
    weights = torch.where(
        (y_true == 1) & (y_pred < 0.5),
        torch.tensor(fn_weight, dtype=bce.dtype, device=bce.device),
        torch.where(
            (y_true == 0) & (y_pred >= 0.5),
            torch.tensor(fp_weight, dtype=bce.dtype, device=bce.device),
            torch.tensor(1.0, dtype=bce.dtype, device=bce.device),
        )
    )

    return torch.mean(weights * bce)


def evaluate_panecho_classification(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate PanEcho's 26+ classification tasks with clinical metrics.

    Args:
        predictions: dict of {task_name -> probability or class probabilities}
        ground_truth: dict of {task_name -> true label}

    Returns:
        dict with per-task clinical metrics
    """
    metrics = ClinicalMetrics()
    results: Dict[str, Any] = {}

    # Binary classification tasks (yes/no)
    binary_tasks = [
        "TamponadePhysiology", "ASD", "VSD", "PDA", "LVThrombus", "RVThrombus",
        "VegetationAV", "VegetationMV", "VegetationTV", "AorticDissection",
    ]

    # Multi-class tasks (categories: none/mild/moderate/severe)
    multiclass_tasks = {
        "AVStenosis": ["none", "mild", "moderate", "severe"],
        "AVRegurgitation": ["none", "mild", "moderate", "severe"],
        "MVStenosis": ["none", "mild", "moderate", "severe"],
        "MVRegurgitation": ["none", "mild", "moderate", "severe"],
        "TVRegurgitation": ["none", "mild", "moderate", "severe"],
        "PericardialEffusion": ["none", "small", "moderate", "large"],
        "LVSystolicFunction": ["normal", "mildly_reduced", "moderately_reduced", "severely_reduced"],
        "RVSystolicFunction": ["normal", "mildly_reduced", "reduced"],
        "RVSize": ["normal", "mildly_dilated", "dilated"],
        "LASize": ["normal", "mildly_dilated", "dilated", "severely_dilated"],
        "DiastolicFunction": ["normal", "grade_1", "grade_2", "grade_3"],
    }

    for task in binary_tasks:
        if task in predictions and task in ground_truth:
            prob = predictions[task]
            gt = ground_truth[task]
            if isinstance(prob, (list, np.ndarray)):
                # Probabilities array — take max
                prob_val = float(np.max(prob))
            else:
                prob_val = float(prob)

            results[task] = metrics.evaluate_binary(
                y_true=[int(gt)],
                y_prob=[prob_val],
                task_name=task,
            )

    for task, classes in multiclass_tasks.items():
        if task in predictions and task in ground_truth:
            probs = predictions[task]
            gt = ground_truth[task]
            if isinstance(probs, (list, np.ndarray)):
                results[task] = metrics.evaluate_multiclass(
                    y_true=[int(gt)],
                    y_prob=[list(probs)],
                    class_names=classes,
                    task_name=task,
                )

    return results
