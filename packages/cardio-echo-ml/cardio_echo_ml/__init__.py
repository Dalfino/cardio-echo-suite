"""cardio_echo_ml — Commercial-grade ML engineering layer for cardio-echo-suite.

This package sits BETWEEN the upstream models and the API, adding 6 layers
of ML engineering that transform "it works" into "it's trustworthy":

1. Test-Time Augmentation (TTA) — 4 augmented passes per input, averaged
2. Ensemble Manager — combine multiple models, weight by confidence
3. Uncertainty Estimation — MC dropout for confidence intervals
4. Quality Gating — refuse bad inputs before prediction
5. Consistency Checker — detect stuck/duplicate predictions
6. Failure Mode Detector — flag suspicious predictions for review

Usage:
    from cardio_echo_ml import CommercialPipeline

    pipeline = CommercialPipeline(model=panecho_model)
    result = pipeline.predict(video_tensor)
    # result = {
    #     "ef": 55.2,
    #     "ef_ci_95": [52.1, 58.3],
    #     "confidence": "high",
    #     "quality_score": 0.87,
    #     "flagged_for_review": False,
    #     "augmentations_used": 4,
    #     "consistency_score": 0.95,
    # }

License: MIT (our additions only; upstream models retain their licenses)
"""

__version__ = "0.1.0"
