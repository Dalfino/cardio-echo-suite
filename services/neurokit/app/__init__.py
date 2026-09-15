"""NeuroKit2 utility service — ECG signal processing (HRV, R-peak, signal quality).

Not a model — a pure-Python signal-processing microservice. Consumes raw ECG
signals (from ECG-FM or any source) and computes:

- /hrv     — Heart Rate Variability (time + frequency domain)
- /r-peak  — R-peak detection
- /quality — Signal quality assessment
- /ecg-clean — Cleaned ECG signal (baseline wander removal)

Wraps the NeuroKit2 library (MIT-licensed, github.com/neuropsychology/NeuroKit).
"""

__version__ = "0.1.0"
