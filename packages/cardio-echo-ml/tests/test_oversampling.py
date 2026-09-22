"""Tests for oversampling module."""
import numpy as np
import torch
from cardio_echo_ml.oversampling import (
    EFRangeSampler,
    WeightedMSELoss,
    analyze_ef_distribution,
    compute_class_weights,
    compute_inverse_frequency_weights,
    compute_range_distribution,
    create_balanced_dataset,
    get_ef_range,
)


def test_ef_range_classification():
    assert get_ef_range(20.0) == "severely_reduced"
    assert get_ef_range(35.0) == "moderately_reduced"
    assert get_ef_range(45.0) == "mildly_reduced"
    assert get_ef_range(60.0) == "normal"
    assert get_ef_range(80.0) == "hyperdynamic"


def test_range_distribution():
    studies = [
        {"ef": 20.0}, {"ef": 25.0},  # severely_reduced
        {"ef": 35.0},                 # moderately_reduced
        {"ef": 45.0}, {"ef": 48.0},  # mildly_reduced
        {"ef": 55.0}, {"ef": 60.0}, {"ef": 65.0}, {"ef": 68.0},  # normal
        {"ef": 75.0},                 # hyperdynamic
    ]
    counts = compute_range_distribution(studies)
    assert counts["severely_reduced"] == 2
    assert counts["moderately_reduced"] == 1
    assert counts["mildly_reduced"] == 2
    assert counts["normal"] == 4
    assert counts["hyperdynamic"] == 1


def test_inverse_frequency_weights():
    """Rare cases should get HIGHER weight."""
    studies = [
        {"ef": 20.0},   # severely_reduced (rare)
        {"ef": 60.0},   # normal (common)
        {"ef": 60.0},   # normal (common)
        {"ef": 60.0},   # normal (common)
    ]
    weights = compute_inverse_frequency_weights(studies)
    # The severely reduced case should have higher weight than normal cases
    assert weights[0] > weights[1]  # rare > common
    assert weights[1] == weights[2] == weights[3]  # all normal cases same weight


def test_create_balanced_dataset_oversample():
    """Balanced dataset should have ~equal representation per range."""
    studies = [
        {"ef": 20.0},   # severely_reduced
        {"ef": 60.0}, {"ef": 62.0}, {"ef": 65.0}, {"ef": 68.0},  # normal (4)
    ]
    balanced = create_balanced_dataset(studies, method="oversample")
    counts = compute_range_distribution(balanced)
    # Each range should have ~4 (the max count)
    assert counts["normal"] == 4
    assert counts["severely_reduced"] == 4  # oversampled from 1 to 4


def test_create_balanced_dataset_augment():
    """Augmented oversampling should add _augmented flag."""
    studies = [
        {"ef": 20.0},  # severely_reduced (1 case)
        {"ef": 60.0}, {"ef": 62.0}, {"ef": 65.0}, {"ef": 68.0},  # normal (4)
    ]
    balanced = create_balanced_dataset(studies, method="augment", target_per_range=4)
    augmented = [s for s in balanced if s.get("_augmented")]
    assert len(augmented) > 0  # Should have augmented copies
    assert any(s.get("_augmentation") for s in augmented)  # Should have augmentation type


def test_compute_class_weights():
    studies = [
        {"ef": 20.0},   # severely_reduced (rare → high weight)
        {"ef": 60.0}, {"ef": 62.0}, {"ef": 65.0}, {"ef": 68.0},  # normal (common → low weight)
    ]
    weights = compute_class_weights(studies)
    assert weights[0] > weights[1]  # Rare case has higher weight
    assert len(weights) == len(studies)


def test_weighted_mse_loss():
    """Weighted MSE should penalize rare cases more."""
    pred = torch.tensor([55.0, 55.0])
    target = torch.tensor([20.0, 60.0])  # Case 0 is severely reduced (rare)
    weights = torch.tensor([5.0, 1.0])   # Rare case weighted 5×

    loss_fn = WeightedMSELoss(weights)
    loss = loss_fn(pred, target)

    # Error for case 0: (55-20)^2 = 1225, weighted: 1225 * 5 = 6125
    # Error for case 1: (55-60)^2 = 25, weighted: 25 * 1 = 25
    # Mean: (6125 + 25) / 2 = 3075
    expected = (1225 * 5 + 25 * 1) / 2
    assert abs(loss.item() - expected) < 0.01


def test_analyze_ef_distribution():
    studies = [
        {"ef": 20.0},
        {"ef": 60.0}, {"ef": 62.0}, {"ef": 65.0}, {"ef": 68.0},
    ]
    analysis = analyze_ef_distribution(studies)
    assert analysis["total"] == 5
    assert analysis["ranges"]["normal"]["count"] == 4
    assert analysis["ranges"]["severely_reduced"]["count"] == 1
    assert analysis["imbalance_ratio"] == 4.0  # 4 / 1


def test_ef_range_sampler():
    """Sampler should exist and have correct length."""
    studies = [{"ef": 20.0}, {"ef": 60.0}, {"ef": 62.0}, {"ef": 65.0}]
    sampler = EFRangeSampler(studies)
    assert len(sampler) == len(studies)
