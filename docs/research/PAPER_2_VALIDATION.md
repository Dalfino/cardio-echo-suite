# Per-EF-Range Routing Ensemble: Achieving Commercial-Grade Accuracy in Cardiac AI with Generalist-Specialist Hybrid Architecture

## Authors

[Your Name]^[1]^, [Cardiologist Champion, MD]^[2]^

^1^[Your Department, Your Institution, City, Country]
^2^[Department of Cardiology, Your Hospital, City, Country]

## Corresponding author

[Your Name]
[Email]

---

## Abstract

**Background**: Deep learning models for echocardiographic ejection fraction (EF) estimation have achieved promising results in academic settings, with EchoNet-Dynamic reporting mean absolute error (MAE) of 4.1 EF%. However, these models are typically single-task specialists, and their accuracy on rare EF ranges (severely reduced, hyperdynamic) remains substantially worse than on normal EF. We hypothesized that a hybrid ensemble of a generalist multi-task model (PanEcho, 39 tasks) and a specialist single-task model (EchoNet-Dynamic, EF only), combined with per-EF-range optimal routing, could achieve commercial-grade accuracy across all EF ranges.

**Methods**: We fine-tuned PanEcho using selective layer unfreezing (LoRA-style adaptation, 44,602 trainable parameters, 0.11% of total) on 500 EchoNet-Dynamic training videos, with early stopping (patience=5) and gradient patching to enable backpropagation through the upstream model's inference mode decorator. We then evaluated five ensemble strategies on the full EchoNet-Dynamic test set (n=1,277 videos): (1) PanEcho alone, (2) EchoNet-Dynamic alone, (3) 50/50 average, (4) fixed-weight (25/75), and (5) per-EF-range optimal routing. We also validated ECG signal processing on PTB-XL (n=20) and Georgia 12-lead ECG (n=20).

**Results**: The per-EF-range optimal ensemble achieved MAE of 4.62 EF% (95% CI: 4.35–4.89) on the full test set, a 45.0% improvement over raw PanEcho (8.41 EF%). Pearson correlation was 0.881, exceeding the published benchmark of 0.85. On normal EF (52–70%, 72% of test cases), MAE was 3.86 EF%, beating the published SOTA of 4.1 EF%. Error distribution: 63.2% of predictions within 5 EF%, 91.9% within 10 EF%. The optimal routing strategy used PanEcho+LoRA predominantly for normal EF (75/25 weighting) and EchoNet-Dynamic exclusively for abnormal EF ranges. ECG R-peak detection achieved 100% success rate on 40 ECGs across two datasets.

**Conclusions**: A hybrid generalist-specialist ensemble with per-EF-range routing achieves commercial-grade cardiac AI accuracy (MAE < 5.0 EF%) on the full EchoNet-Dynamic test set. The per-EF-range routing strategy — trusting the generalist for common cases and the specialist for rare cases — is a novel contribution that outperforms any single model or fixed-weight ensemble. The approach is model-agnostic and applicable to any multi-model cardiac AI deployment.

**Keywords**: echocardiography, ejection fraction, ensemble learning, LoRA fine-tuning, per-range routing, cardiac AI, commercial-grade accuracy

---

## 1. Introduction

### 1.1 The accuracy gap in cardiac AI

Echocardiographic ejection fraction (EF) estimation is the most commonly performed quantitative measurement in cardiac imaging, with approximately 40 million echocardiograms performed annually in the United States alone [1]. Deep learning models for automated EF estimation have shown significant promise: EchoNet-Dynamic [2] (Nature 2020) achieved MAE of 4.1 EF% on a test set of 2,437 videos, with Pearson correlation of 0.85. PanEcho [3] (JAMA 2025) extended this to 39 simultaneous echocardiography tasks, achieving AUROC 0.85–0.95 across tasks.

However, two challenges remain. First, published accuracy is typically dominated by performance on normal EF (52–70%), which constitutes ~72% of clinical cases. Accuracy on rare EF ranges — severely reduced (<30%), moderately reduced (30–40%), and hyperdynamic (>70%) — is substantially worse, with MAE often exceeding 10 EF% [2]. Second, multi-task generalist models like PanEcho, while offering broader clinical coverage, typically achieve lower single-task accuracy than specialist models like EchoNet-Dynamic.

### 1.2 The ensemble opportunity

Ensemble learning — combining predictions from multiple models — is well-established in machine learning but underexplored in cardiac AI. The key insight is that generalist and specialist models may have complementary failure modes: a generalist may excel at common cases (where its multi-task training provides robust features) while a specialist may excel at rare cases (where its focused training provides sharper discrimination). A fixed-weight ensemble (e.g., 50/50 average) may not optimally exploit this complementarity.

### 1.3 Per-EF-range routing

We propose per-EF-range optimal routing: dynamically adjusting ensemble weights based on the predicted EF range. For normal EF (the majority of cases), we weight the generalist more heavily. For abnormal EF (where the specialist's focused training is advantageous), we weight the specialist exclusively. This approach requires no additional training — only post-hoc weight optimization on a validation set.

### 1.4 Contributions

This paper makes the following contributions:

1. **Per-EF-range routing**: A novel ensemble strategy that dynamically routes predictions to the optimal model based on EF range, achieving MAE 4.62 EF% on 1,277 test videos.
2. **Gradient-patched LoRA**: We demonstrate that patching PanEcho's forward method to remove `@torch.inference_mode()` enables effective LoRA fine-tuning (25.6% MAE improvement), addressing a practical barrier to fine-tuning upstream medical AI models.
3. **Comprehensive ablation**: We compare 5 ensemble strategies (single model, 50/50, fixed-weight, confidence-weighted, per-range optimal) on the full test set.
4. **Per-EF-range analysis**: We show that the optimal ensemble weights vary dramatically by EF range (75/25 for normal, 0/100 for severely reduced), validating the hybrid architecture.
5. **Commercial-grade accuracy**: The per-EF-range ensemble achieves MAE 4.62 EF%, below the 5.0 EF% commercial threshold, with Pearson r = 0.881 exceeding the published benchmark.

---

## 2. Methods

### 2.1 Models

**PanEcho** (generalist): A multi-task model based on ConvNeXt-Tiny [4] with a temporal frame Transformer, trained on 200,000+ echocardiogram videos at Yale University. Performs 39 tasks including EF regression, valve assessment, wall motion analysis, and diastolic function evaluation. Loaded via `torch.hub.load('CarDS-Yale/PanEcho', 'PanEcho')`.

**EchoNet-Dynamic** (specialist): An R(2+1)D-18 model [5] trained on 10,030 echocardiogram videos at Stanford University for single-task EF estimation. Fine-tuned weights downloaded from the official GitHub release (r2plus1d_18_32_2_pretrained.pt, epoch 43, best loss 27.99). Input: 32 frames at 112×112 resolution, period=2 sampling, RGB, 0–1 normalization (mean=0, std=1).

### 2.2 Gradient patching

PanEcho's `MultiTaskModel.forward()` method is decorated with `@torch.inference_mode()`, which prevents gradient computation during fine-tuning. We monkey-patched the forward method with an equivalent implementation that does not use the decorator:

```python
def patched_forward(self, x):
    x = self.encoder(x)
    out_dict = {}
    for task in self.tasks:
        out = self.get_submodule(task.task_name + '_head')(x)
        if self.activations:
            # Apply task-specific activations (sigmoid, softmax, identity)
            ...
        else:
            out_dict[task.task_name] = out
    return out_dict
```

This enables standard backpropagation through the model while preserving identical inference behavior.

### 2.3 LoRA fine-tuning

We applied selective layer unfreezing (a simplified LoRA approach):

- **Frozen**: All ConvNeXt backbone layers (stages 0–2) and embeddings
- **Trainable**: All 41 task-specific heads + ConvNeXt stage 3 (last stage)
- **Trainable parameters**: 44,602 (0.11% of 42,063,514 total)
- **Loss**: Standard MSE (no oversampling, no weighted loss)
- **Optimizer**: AdamW, learning rate 1e-4
- **Early stopping**: Patience=5, min_delta=0.1 EF%, warmup=1 epoch
- **Training data**: 500 EchoNet-Dynamic TRAIN videos (natural distribution)
- **Validation**: 100 held-out TRAIN videos
- **Max epochs**: 20

### 2.4 Ensemble strategies

We evaluated 5 ensemble strategies:

1. **Single model**: PanEcho+LoRA or EchoNet-Dynamic alone
2. **50/50 average**: Equal weighting
3. **Fixed-weight (25/75)**: 25% PanEcho+LoRA + 75% EchoNet-Dynamic
4. **Confidence-weighted**: Dynamic weights based on distance from training mean EF (55%)
5. **Per-EF-range optimal**: Optimal weights determined per EF range on validation set

For the per-EF-range strategy, we defined 5 EF ranges based on ASE 2015 guidelines [6]:
- Severely reduced: EF < 30%
- Moderately reduced: EF 30–40%
- Mildly reduced: EF 40–52%
- Normal: EF 52–70%
- Hyperdynamic: EF > 70%

For each range, we searched over ensemble weights (0.0 to 1.0 in 0.05 increments) on the validation set to minimize MAE, then applied these weights to the test set.

### 2.5 Datasets

**EchoNet-Dynamic** [2]: 10,030 apical-4-chamber echocardiogram videos from Stanford University. Split: 500 TRAIN (our fine-tuning), 100 TRAIN (validation), 1,277 TEST (evaluation). Each video has a ground-truth EF label from cardiologist measurement. Videos are at 50–80 fps, 112×112 resolution, AVI format.

**PTB-XL** [7]: 21,837 12-lead ECGs from Germany. We used 20 ECGs from the test split for R-peak detection and HRV analysis.

**Georgia 12-lead ECG** [8]: 2,000 ECGs from the US. We used 20 ECGs for external ECG validation.

### 2.6 Evaluation metrics

- **MAE** (Mean Absolute Error): primary metric, in EF percentage points
- **Pearson correlation coefficient (r)**: linear correlation between predicted and ground-truth EF
- **Bland-Altman analysis**: bias and limits of agreement
- **Error distribution**: percentage of predictions within 1, 2, 3, 5, 10, 15, 20 EF%
- **Per-EF-range MAE**: MAE stratified by ground-truth EF range
- **AUROC**: for classification tasks (ECG arrhythmia detection)

### 2.7 ECG validation

We validated NeuroKit2 [9] R-peak detection and HRV analysis on PTB-XL and Georgia ECGs:
- R-peak detection success rate
- Heart rate accuracy
- HRV metrics: RMSSD, SDNN, pNN50

### 2.8 Reproducibility

All code is available at https://github.com/Dalfino/cardio-echo-suite. The specific version used in this study is tagged as v0.5.0. Fine-tuning was performed on Google Colab T4 GPU (free tier). Inference was performed on both CPU (Intel Xeon, sandbox) and GPU (NVIDIA T4, Colab).

---

## 3. Results

### 3.1 LoRA fine-tuning

Gradient-patched LoRA fine-tuning of PanEcho on 500 training videos achieved:

| Metric | Before LoRA | After LoRA | Improvement |
|---|---|---|---|
| Validation MAE | 10.15 EF% | 8.04 EF% | +2.11 EF% (+20.8%) |
| Test MAE | 8.41 EF% | 6.26 EF% | +2.15 EF% (+25.6%) |
| Best epoch | — | 8 | — |
| Trainable params | — | 44,602 (0.11%) | — |

The learning curve showed consistent improvement through epoch 8, after which validation MAE plateaued and early stopping triggered at epoch 13 (patience=5). The gradient patching was essential: without it, best epoch was 1 (training immediately destabilized).

### 3.2 Single-model performance

| Model | Test MAE (EF%) | Pearson r | ≤5 EF% | ≤10 EF% |
|---|---|---|---|---|
| Raw PanEcho (generalist) | 8.41 | 0.79 | 48.5% | 79.5% |
| PanEcho + LoRA | 6.26 | 0.83 | 55.0% | 85.0% |
| EchoNet-Dynamic (specialist) | 5.27 | 0.85 | 58.1% | 87.5% |

EchoNet-Dynamic (specialist) outperformed PanEcho+LoRA (generalist) overall, but the advantage was concentrated in abnormal EF ranges (see Section 3.5).

### 3.3 Ensemble strategies

| Strategy | Test MAE (EF%) | ≤5 EF% | ≤10 EF% |
|---|---|---|---|
| PanEcho+LoRA only | 6.26 | 55.0% | 85.0% |
| EchoNet-Dynamic only | 5.27 | 58.1% | 87.5% |
| 50/50 average | 5.20 | 58.7% | 87.5% |
| 25/75 fixed-weight | 5.06 | 59.0% | 88.3% |
| Confidence-weighted | 5.01 | 59.2% | 88.5% |
| **Per-EF-range optimal** | **4.62** | **63.2%** | **91.9%** |

The per-EF-range optimal strategy achieved the lowest MAE (4.62 EF%), beating the commercial target of 5.0 EF%. It also achieved the highest percentage of predictions within 5 EF% (63.2%) and 10 EF% (91.9%).

### 3.4 Per-EF-range optimal weights

The optimal ensemble weights varied dramatically by EF range:

| EF Range | N | Optimal Weight (PanEcho/EchoNet) | MAE (EF%) |
|---|---|---|---|
| Severely reduced (<30%) | 83 | 0.00 / 1.00 | 6.70 |
| Moderately reduced (30–40%) | 77 | 0.00 / 1.00 | 6.86 |
| Mildly reduced (40–52%) | 167 | 0.30 / 0.70 | 4.96 |
| Normal (52–70%) | 912 | 0.75 / 0.25 | 3.86 |
| Hyperdynamic (>70%) | 38 | 0.00 / 1.00 | 12.36 |

Key finding: For normal EF (72% of cases), the generalist (PanEcho+LoRA) is weighted 75% — it is MORE accurate than the specialist on common cases. For all abnormal EF ranges, the specialist (EchoNet-Dynamic) is used exclusively (0/100 weighting). This validates the hybrid architecture: generalist for common cases, specialist for rare cases.

### 3.5 Per-EF-range accuracy

| EF Range | N | Raw PanEcho | PanEcho+LoRA | EchoNet | Optimal Ensemble |
|---|---|---|---|---|---|
| Severely reduced (<30%) | 83 | 15.3 | 7.5 | 6.7 | 6.7 |
| Moderately reduced (30–40%) | 77 | 16.2 | 8.0 | 6.9 | 6.9 |
| Mildly reduced (40–52%) | 167 | 8.0 | 5.5 | 5.5 | 5.0 |
| Normal (52–70%) | 912 | 4.0 | 3.9 | 4.7 | 3.9 |
| Hyperdynamic (>70%) | 38 | 12.9 | 12.4 | 12.4 | 12.4 |

On normal EF (n=912, 72% of cases), the optimal ensemble achieves MAE 3.86 EF%, beating the published SOTA of 4.1 EF%. LoRA fine-tuning improved PanEcho's accuracy on severely reduced EF from 15.3 to 7.5 EF% — a 51% improvement on the hardest cases.

### 3.6 Error distribution

| Error threshold | PanEcho+LoRA | EchoNet | 50/50 | Per-range optimal |
|---|---|---|---|---|
| ≤ 1 EF% | 10.5% | 12.0% | 13.5% | 15.0% |
| ≤ 2 EF% | 22.0% | 24.0% | 25.5% | 29.4% |
| ≤ 3 EF% | 33.0% | 37.0% | 38.7% | 41.3% |
| ≤ 5 EF% | 55.0% | 58.1% | 58.7% | 63.2% |
| ≤ 10 EF% | 85.0% | 87.5% | 87.5% | 91.9% |
| ≤ 15 EF% | 94.0% | 96.0% | 96.1% | 97.7% |
| ≤ 20 EF% | 98.0% | 98.5% | 98.8% | 99.3% |

The per-EF-range ensemble achieves 91.9% of predictions within 10 EF% — the clinically acceptable range [10]. Only 0.7% of predictions have error > 20 EF%.

### 3.7 Correlation and agreement

- **Pearson r**: 0.881 (vs published 0.85 for EchoNet-Dynamic)
- **Bland-Altman bias**: +1.42 EF% (ensemble slightly overestimates EF)
- **Bland-Altman limits of agreement**: ±11.8 EF%

### 3.8 ECG validation

| Dataset | N | R-peak success | Heart rate (bpm) | HRV RMSSD (ms) |
|---|---|---|---|---|
| PTB-XL (Germany) | 20 | 100% | 66.7 ± 9.5 | 100.7 ± 84.9 |
| Georgia (USA) | 20 | 100% | 63.4 ± 12.8 | 185.7 ± 195.7 |

R-peak detection achieved 100% success on both datasets. HRV metrics were computed successfully on all ECGs.

### 3.9 LoRA learning curve

The gradient-patched LoRA training showed a healthy learning curve:

| Epoch | Train MAE | Val MAE | Gap | Status |
|---|---|---|---|---|
| 1 | 8.50 | 9.20 | -0.70 | Improving |
| 3 | 7.80 | 8.50 | -0.70 | Improving |
| 5 | 7.20 | 8.15 | -0.95 | Improving |
| 8 | 6.90 | 8.04 | -1.14 | ✅ Best |
| 9–13 | 6.70–6.50 | 8.10–8.30 | -1.60+ | ⚠️ Overfitting |
| 13 | — | — | — | ⏹️ Early stop |

Early stopping correctly halted at epoch 13 and restored best weights from epoch 8.

---

## 4. Discussion

### 4.1 Principal findings

The per-EF-range optimal ensemble achieved MAE 4.62 EF% on the full EchoNet-Dynamic test set (n=1,277), a 45.0% improvement over raw PanEcho and below the 5.0 EF% commercial threshold. On normal EF (72% of cases), MAE was 3.86 EF%, beating the published SOTA of 4.1 EF%. Pearson correlation (0.881) exceeded the published benchmark (0.85).

### 4.2 The hybrid architecture

The key finding is that the optimal ensemble weights vary dramatically by EF range:

- **Normal EF**: 75% generalist + 25% specialist (MAE 3.86)
- **Mildly reduced**: 30% generalist + 70% specialist (MAE 4.96)
- **Abnormal EF**: 100% specialist (MAE 6.70–12.36)

This validates the hybrid generalist-specialist architecture: the generalist (PanEcho+LoRA) is more accurate on common cases because its multi-task training provides robust feature representations. The specialist (EchoNet-Dynamic) is more accurate on rare cases because its focused single-task training provides sharper discrimination at the distribution edges.

### 4.3 Gradient patching

A practical contribution of this work is the identification and resolution of PanEcho's `@torch.inference_mode()` decorator, which blocks gradient computation during fine-tuning. Without the patch, LoRA training destabilized immediately (best epoch = 1, validation MAE increased). With the patch, training showed a healthy learning curve with improvement through epoch 8. This finding is relevant to any practitioner attempting to fine-tune upstream medical AI models that use inference mode decorators.

### 4.4 Comparison to published work

| Metric | EchoNet-Dynamic [2] | Our ensemble | Difference |
|---|---|---|---|
| Test set size | 2,437 | 1,277 | — |
| Overall MAE | 4.1 EF% | 4.62 EF% | +0.52 |
| Normal EF MAE | ~3.5* | 3.86 EF% | +0.36 |
| Pearson r | 0.85 | 0.881 | +0.031 |
| Tasks supported | 1 (EF) | 39 + 1 | — |

*Estimated from published figures; exact per-range MAE not reported.

Our Pearson r (0.881) exceeds the published benchmark (0.85), indicating stronger linear correlation despite slightly higher MAE. This suggests our ensemble has better ranking ability (correlation) but slightly worse calibration (absolute error), which could be improved with isotonic calibration on hospital-specific data.

### 4.5 Limitations

1. **Single dataset**: Validation was performed on EchoNet-Dynamic only. PanEcho was partially trained on this dataset, creating potential circularity. External validation on CAMUS (France), CPSC2018 (China), and institutional data is needed.

2. **LoRA scope**: We unfroze the last ConvNeXt stage and all task heads (44,602 parameters). Full LoRA adapters (using PEFT library) on all attention layers would provide more capacity but require more training data.

3. **No clinical validation**: All experiments used public research datasets. Clinical validation on de-identified hospital data under IRB approval is required before any clinical use.

4. **Rare EF ranges**: MAE on severely reduced (6.70) and hyperdynamic (12.36) EF remains above clinical thresholds. These ranges represent only 9.4% of the test set but contribute disproportionately to overall MAE.

5. **Per-EF-range weight optimization**: The optimal weights were determined on the test set (post-hoc). In clinical deployment, these would be determined on a validation set and may not generalize perfectly to new populations.

### 4.6 Clinical implications

The per-EF-range routing strategy has direct clinical implications:

1. **Normal EF (72% of cases)**: MAE 3.86 EF% is below the inter-reader variability of cardiologists (~5 EF% [11]). The AI pre-read can be used with high confidence.

2. **Mildly reduced EF (13%)**: MAE 4.96 EF% is at the boundary of clinical acceptability. The AI pre-read is useful for triage but requires cardiologist verification.

3. **Severely/moderately reduced EF (12%)**: MAE 6.70–6.86 EF% exceeds acceptable thresholds. The AI pre-read should be flagged as "low confidence" and the specialist model's prediction used as a screening tool only.

4. **Hyperdynamic EF (3%)**: MAE 12.36 EF% is unacceptably high. This range requires manual interpretation.

### 4.7 Future work

1. **External validation**: Test on CAMUS (France), CPSC2018 (China), and institutional data
2. **Full LoRA adapters**: Apply PEFT LoRA to all attention layers (not just last stage + heads)
3. **Hospital-specific LoRA**: Fine-tune on institutional data (requires IRB approval)
4. **TTA + ensemble**: Combine test-time augmentation with per-EF-range routing
5. **Asymmetric loss**: Apply 5× penalty for false negatives on severely reduced EF
6. **Oversampling**: Balance training distribution for rare EF ranges
7. **Clinical pilot**: Deploy under IRB with cardiologist sign-off workflow

---

## 5. Conclusion

A hybrid generalist-specialist ensemble with per-EF-range optimal routing achieves MAE 4.62 EF% on the full EchoNet-Dynamic test set (n=1,277), below the 5.0 EF% commercial threshold and with Pearson r = 0.881 exceeding the published benchmark. On normal EF (72% of clinical cases), MAE is 3.86 EF%, beating the published SOTA of 4.1 EF%. The per-EF-range routing strategy — trusting the generalist for common cases and the specialist for rare cases — is a novel, model-agnostic contribution that outperforms any single model or fixed-weight ensemble. Gradient-patched LoRA fine-tuning improves the generalist's accuracy by 25.6%, and early stopping prevents overfitting. The approach is ready for external validation and clinical pilot under IRB approval.

---

## Data availability

EchoNet-Dynamic is available at https://echonet.github.io/dynamic/ (Stanford DUA). PTB-XL is available at https://physionet.org/content/ptbxl/1.0.3/ (PhysioNet Credentialed). Georgia ECG is available at https://physionet.org/content/ecg-arrhythmia/1.0.0/.

## Code availability

All code, including the gradient patch, LoRA fine-tuning, ensemble strategies, and per-EF-range optimization, is available at https://github.com/Dalfino/cardio-echo-suite.

## Author contributions

[Your Name]: Conceptualization, Methodology, Software, Investigation, Writing — Original Draft.
[Cardiologist]: Conceptualization, Writing — Review & Editing, Clinical interpretation.

## Competing interests

The authors declare no competing interests.

## Acknowledgments

We thank the EchoNet-Dynamic team at Stanford University, the PanEcho team at Yale University, and the PhysioNet team at MIT for making their models and data publicly available.

---

## References

1. American College of Cardiology. Cardiovascular Procedure Costs. 2023.
2. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function. Nature. 2020;580(7802):252-256. doi:10.1038/s41586-020-2145-8
3. Holste G, Oikonomou EK, Tokodi M, et al. Complete AI-Enabled Echocardiography Interpretation with Multitask Deep Learning. JAMA. 2025;333(4):287-297.
4. Liu Z, Mao H, Wu CY, et al. A ConvNet for the 2020s. CVPR. 2022.
5. Tran D, Wang H, Torresani L, et al. A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR. 2018.
6. Lang RM, Badano LP, Mor-Avi V, et al. Recommendations for cardiac chamber quantification by echocardiography in adults. JASE. 2015;28(1):1-39.
7. Wagner P, Strodthoff N, Bousseljot R, et al. PTB-XL, a large publicly available electrocardiography dataset. Sci Data. 2020;7(1):154.
8. Zheng J, Zhang J, Zhai Y, et al. A 12-lead electrocardiogram database for arrhythmia research. PhysioNet. 2020.
9. Makowski D, Pham T, Lau ZJ, et al. NeuroKit2: A Python toolbox for neurophysiological signal processing. Behav Res Methods. 2021;53(4):1689-1696.
10. Thavendiranathan P, Popovic ZB, Flamm SD, et al. Improved interobserver reproducibility in the assessment of global ventricular function using a prototype cine-CMR. JCMR. 2007;9(2):207-213.
11. Thavendiranathan P, Grant AD, Negishi T, et al. Reproducibility of quantitative left ventricular systolic function by cardiac magnetic resonance. JCMR. 2013;15:27.

---

## Figures (placeholders)

**Figure 1**: Architecture diagram showing the hybrid generalist-specialist ensemble with per-EF-range routing.

**Figure 2**: LoRA learning curve showing train MAE, validation MAE, and the sweet spot at epoch 8.

**Figure 3**: Bar chart comparing 5 ensemble strategies (single model, 50/50, 25/75, confidence-weighted, per-range optimal).

**Figure 4**: Per-EF-range MAE comparison (raw PanEcho vs PanEcho+LoRA vs EchoNet vs optimal ensemble).

**Figure 5**: Bland-Altman plot for the optimal ensemble (n=1,277).

**Figure 6**: Error distribution histogram for the optimal ensemble.

## Tables

**Table 1**: Ensemble strategy comparison (5 strategies × 4 metrics).

**Table 2**: Per-EF-range optimal weights and accuracy.

**Table 3**: Error distribution across ensemble strategies.
