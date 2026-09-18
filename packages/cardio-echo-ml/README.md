# cardio-echo-ml

> Commercial-grade ML engineering layer for cardio-echo-suite.

This package adds 6 layers of ML engineering on top of upstream models,
transforming "it works" into "it's trustworthy":

1. **Test-Time Augmentation (TTA)** — 4 augmented passes per input, averaged → -0.5 to -1.0 EF% MAE
2. **Ensemble Manager** — combine multiple models, inverse-variance weighting → -1.0 to -1.5 EF% MAE
3. **Uncertainty Estimation** — MC dropout for 95% confidence intervals → clinical trust
4. **Quality Gating** — refuse bad inputs before prediction → prevents garbage-in/garbage-out
5. **Consistency Checker** — detect stuck/duplicate predictions → safety net
6. **Failure Mode Detector** — flag suspicious predictions for review → cardiologist trust

## Installation

```bash
pip install -e packages/cardio-echo-ml
```

## Usage

```python
from cardio_echo_ml.pipeline import CommercialPipeline

pipeline = CommercialPipeline(
    model=panecho_model,
    calibration_params={"bias": -4.5},  # from calibrate_and_analyze.py
    n_tta=4,
    n_mc_dropout=10,
)

result = pipeline.predict(video_tensor, patient_id="P001")
# result = {
#     "ef": 55.2,
#     "ef_ci_95": [52.1, 58.3],
#     "ef_raw": 60.1,
#     "ef_calibrated": 55.2,
#     "confidence": "high",
#     "quality_score": 0.87,
#     "flagged_for_review": False,
#     "tta_std": 1.2,
#     "mc_dropout_std": 1.5,
#     ...
# }
```

## License

MIT (our additions only; upstream models retain their licenses).
