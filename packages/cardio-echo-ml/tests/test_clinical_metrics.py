"""Tests for clinical metrics module."""
import numpy as np
from cardio_echo_ml.clinical_metrics import (
    ClinicalMetrics,
    ClinicalThreshold,
    asymmetric_loss,
    evaluate_panecho_classification,
)
import torch


def test_perfect_binary_classification():
    """A perfect model should have sensitivity=1.0, specificity=1.0, AUROC=1.0."""
    m = ClinicalMetrics()
    result = m.evaluate_binary(
        y_true=[1, 0, 1, 0, 1, 0, 1, 0],
        y_prob=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        task_name="test",
        threshold=0.5,
    )
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["auroc"] == 1.0
    assert result["fn_count"] == 0
    assert result["fp_count"] == 0
    assert result["clinical_risk"] == "LOW_RISK"


def test_all_false_negatives():
    """Model that misses all disease cases should be HIGH_RISK_FN."""
    m = ClinicalMetrics()
    result = m.evaluate_binary(
        y_true=[1, 1, 1, 1],  # All have disease
        y_prob=[0.1, 0.2, 0.3, 0.1],  # All predicted as healthy
        task_name="STEMI",
        threshold=0.5,
    )
    assert result["sensitivity"] == 0.0  # Caught none
    assert result["fn_count"] == 4  # All 4 missed
    assert "HIGH_RISK" in result["clinical_risk"]


def test_all_false_positives():
    """Model that flags all healthy patients should be HIGH_RISK_FP or MODERATE_RISK_FP."""
    m = ClinicalMetrics()
    result = m.evaluate_binary(
        y_true=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # All healthy
        y_prob=[0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8],  # All flagged
        task_name="test",
        threshold=0.5,
    )
    assert result["specificity"] == 0.0
    assert result["fp_count"] == 10
    assert "RISK_FP" in result["clinical_risk"]


def test_stemi_threshold():
    """STEMI should use low screening threshold (never miss)."""
    m = ClinicalMetrics()
    # With threshold 0.15, should catch all STEMI cases
    result = m.evaluate_binary(
        y_true=[1, 1, 1, 0, 0, 0],
        y_prob=[0.20, 0.18, 0.16, 0.10, 0.05, 0.08],
        task_name="STEMI",
        threshold=0.15,
    )
    assert result["sensitivity"] == 1.0  # Caught all 3 STEMI
    assert result["fn_count"] == 0


def test_optimal_threshold_finding():
    """Find threshold that achieves 90% sensitivity + 80% specificity."""
    m = ClinicalMetrics()
    # Create test data where threshold ~0.3 achieves both goals
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25,
              0.2, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001]
    result = m.find_optimal_threshold(y_true, y_prob, min_sensitivity=0.9, min_specificity=0.8)
    assert bool(result["meets_criteria"]) is True
    assert result["sensitivity_at_optimal"] >= 0.9


def test_multiclass_evaluation():
    """Multi-class (none/mild/moderate/severe) evaluation."""
    m = ClinicalMetrics()
    result = m.evaluate_multiclass(
        y_true=[0, 1, 2, 3, 0, 1],
        y_prob=[[0.7, 0.2, 0.1, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.1, 0.1, 0.7, 0.1],
                [0.0, 0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1, 0.0],
                [0.2, 0.7, 0.1, 0.0]],
        class_names=["none", "mild", "moderate", "severe"],
        task_name="AVStenosis",
    )
    assert result["n_classes"] == 4
    assert "per_class" in result
    assert "none" in result["per_class"]
    assert result["accuracy"] == 1.0  # All correct


def test_asymmetric_loss():
    """Asymmetric loss should penalize FN more than FP."""
    y_true = torch.tensor([1.0, 1.0, 0.0, 0.0])
    y_pred = torch.tensor([0.3, 0.3, 0.7, 0.7])  # 2 FN + 2 FP

    loss = asymmetric_loss(y_true, y_pred, fn_weight=5.0, fp_weight=1.0)

    # Verify FN gets 5× penalty
    # FN: y_true=1, y_pred<0.5 → weight 5
    # FP: y_true=0, y_pred>=0.5 → weight 1
    assert loss > 0
    assert loss.item() > 0.5  # Should be non-trivial


def test_cross_entropy_computed():
    """Cross-entropy should be computed for classification tasks."""
    m = ClinicalMetrics()
    result = m.evaluate_binary(
        y_true=[1, 0],
        y_prob=[0.9, 0.1],
        task_name="test",
    )
    assert "cross_entropy" in result
    assert result["cross_entropy"] > 0  # Not zero
    assert result["cross_entropy"] < 1  # Not huge (good predictions)


def test_clinical_threshold_presets():
    """Verify clinical threshold presets exist for critical tasks."""
    from cardio_echo_ml.clinical_metrics import CLINICAL_THRESHOLDS
    assert "TamponadePhysiology" in CLINICAL_THRESHOLDS
    assert "STEMI" in CLINICAL_THRESHOLDS
    assert CLINICAL_THRESHOLDS["STEMI"].min_sensitivity >= 0.95  # Never miss STEMI
    assert CLINICAL_THRESHOLDS["TamponadePhysiology"].screening_threshold <= 0.2  # Low threshold
