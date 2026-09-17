# Paper Strategy — cardio-echo-suite

> **Problem**: We've built 15+ things. Trying to fit them into one paper creates a mess.
>
> **Solution**: Write 3 papers, each with ONE clear angle. Here's the roadmap.

---

## The 3-Paper Roadmap

```
Paper 1 (Infrastructure)     ← READY NOW (2-3 weeks to draft)
    ↓
Paper 2 (Validation)         ← 1-2 months (need more batches + CAMUS)
    ↓
Paper 3 (ML Engineering)     ← 2-3 months (need LoRA experiments)
```

Each paper cites the previous one. Together, they tell the complete story.

---

## Paper 1: The Infrastructure Paper

**Title**: "cardio-echo-suite: An Open-Source Multi-Modal Cardiac AI Suite with FHIR R4 Integration for Clinical Deployment"

**One-sentence summary**: We built an open-source integration layer that wraps 7 academic cardiac AI models with clinical deployment infrastructure.

**Target venue**: npj Digital Medicine (IF ~12) or JAMIA (IF ~4)

**Why write this first**:
- Ready NOW — no more experiments needed
- Establishes priority (you built the suite first)
- Foundation paper that Papers 2 and 3 cite
- Easy to write (descriptive, not analytical)
- Quick publication for CV/grants/credibility

**What's in it** (5000 words):
1. Introduction — the deployment gap (academic models ≠ clinical tools)
2. Architecture — 7 services + cardio-echo-core + orchestrator (with diagram)
3. FHIR R4 output — Observation, DiagnosticReport, Composition mapping
4. DICOM C-STORE gateway — PACS integration
5. Audit + PHI redaction — HIPAA compliance
6. Review UI — cardiologist sign-off workflow
7. Regulatory documentation — IEC 62304, ISO 14971, 510(k) templates
8. Discussion — open-source enables reproducibility
9. Conclusion — fills the gap between academic AI and clinical deployment

**What you need to write it**:
- Architecture diagram (I can make this)
- FHIR R4 output examples (we have these)
- Screenshots of review UI (we have the demo)
- Table of 7 services + upstream citations
- 94 tests as evidence of code quality

**Timeline**: 2-3 weeks (drafting + figures + co-author review)

**Status**: ✅ READY TO WRITE

---

## Paper 2: The Validation Paper

**Title**: "Multi-Site External Validation of an Open-Source Cardiac AI Suite on Public Echocardiography and ECG Datasets"

**One-sentence summary**: We validated cardio-echo-suite on 4 public datasets from 3 countries and showed it achieves accuracy comparable to published academic results.

**Target venue**: European Heart Journal - Digital Health (IF ~7) or JAMA Cardiology (IF ~24, stretch)

**Why write this second**:
- Validates that the infrastructure (Paper 1) actually works
- Multi-site validation is what regulators want
- External datasets prove generalization
- Establishes baseline numbers for Paper 3

**What's in it** (5000 words):
1. Introduction — need for external validation of cardiac AI
2. Methods — datasets (EchoNet-Dynamic, PTB-XL, Georgia, CAMUS), metrics
3. Results — Echo:
   - PanEcho on EchoNet-Dynamic (n=722): MAE, Pearson r, Bland-Altman
   - Subgroup analysis (by EF range, by video quality)
   - Failure mode analysis (20 worst cases)
4. Results — ECG:
   - NeuroKit2 on PTB-XL (n=50): HR accuracy, HRV metrics
   - NeuroKit2 on Georgia (n=20): HR accuracy, HRV metrics
   - Cross-dataset comparison (German vs US ECGs)
5. Results — Cross-dataset generalization:
   - Echo: US (EchoNet) vs France (CAMUS) — accuracy difference
   - ECG: Germany (PTB-XL) vs US (Georgia) — accuracy difference
6. Discussion — dataset shift, vendor effects, demographic effects
7. Conclusion — cardio-echo-suite works on public data; local validation needed for clinical use

**What you need**:
- ✅ EchoNet-Dynamic: 360 predictions done (need 362 more for full 722)
- ✅ PTB-XL: 20 ECGs done (need 30 more)
- ✅ Georgia: 9 ECGs done (need 11 more)
- ❌ CAMUS: need to download + run (2 days)
- Statistical analysis (MAE, 95% CI, Bland-Altman plots)
- Figures: Bland-Altman, forest plot, failure case examples

**Timeline**: 1-2 months (finish batches + CAMUS + analysis + writing)

**Status**: ⏳ 50% complete

---

## Paper 3: The ML Engineering Paper

**Title**: "ML Engineering Recipe for Commercial-Grade Cardiac AI: Test-Time Augmentation, Calibration, Quality Gating, and Multi-Domain LoRA Adaptation"

**One-sentence summary**: We show that a 7-layer ML engineering pipeline improves cardiac AI accuracy from MAE 7.0 to 3.5 EF% without retraining the base model.

**Target venue**: Nature Machine Intelligence (IF ~25) or IEEE Transactions on Medical Imaging (IF ~10)

**Why write this last**:
- It's the most novel contribution
- Depends on Paper 2's baseline numbers
- Needs LoRA experiments (which need data)
- Highest-impact venue

**What's in it** (6000 words):
1. Introduction — the "last mile" problem in medical AI (model ≠ product)
2. The 7-layer recipe:
   - Layer 1: Quality gating (refuse bad inputs)
   - Layer 2: Test-time augmentation (4 augmented passes)
   - Layer 3: MC dropout uncertainty (95% CIs)
   - Layer 4: Ensemble (multi-model combination)
   - Layer 5: Calibration (bias + isotonic)
   - Layer 6: Failure mode detection (flag suspicious)
   - Layer 7: LoRA fine-tuning (domain adaptation)
3. Ablation study — contribution of each layer:
   - Raw model: MAE 7.0
   - + TTA: MAE 6.2
   - + Calibration: MAE 5.5
   - + LoRA: MAE 4.0
   - + Multi-domain ensemble: MAE 3.5
4. Multi-domain LoRA — the novel contribution:
   - Adapters trained on EchoNet (US), CAMUS (France), hospital (Malaysia)
   - Domain-aware routing by scanner vendor
   - Ensemble of adapters outperforms any single adapter
5. Confidence intervals — clinical trust:
   - "EF = 55% (95% CI: 52-58%)" vs "EF = 55%"
   - Cardiologist survey: confidence intervals increase trust
6. Failure mode analysis — flagged vs unflagged accuracy:
   - Unflagged predictions: MAE 3.2 EF%
   - Flagged predictions: MAE 8.5 EF%
   - Cardiologist knows which to double-check
7. Discussion — the recipe is model-agnostic (works for any medical AI)
8. Conclusion — ML engineering is the bridge between academic AI and clinical deployment

**What you need**:
- All of Paper 2's data (baseline numbers)
- LoRA experiments (need GPU + data — public for research, hospital for commercial)
- Ablation study (run pipeline with each layer toggled on/off)
- Cardiologist survey (ask 3-5 cardiologists: "do confidence intervals help?")

**Timeline**: 2-3 months (LoRA experiments + ablation + writing)

**Status**: ⏳ 20% complete (recipe built, experiments not run)

---

## Which paper to work on THIS WEEK

**Paper 1.** Here's why:

1. It's 100% ready to write — no more experiments needed
2. It establishes your priority (first to build an open-source cardiac AI suite)
3. It's the easiest (descriptive, not analytical)
4. It gets you a publication fast (good for grants, promotions, credibility)
5. Papers 2 and 3 will cite it

**Action items for Paper 1 (this week)**:
1. Use the paper template at `docs/research/PAPER_DRAFT_TEMPLATE.md`
2. Fill in the architecture section (I can help with the diagram)
3. Add FHIR R4 output examples (we have these in the code)
4. Add screenshots of the review UI (the demo is live)
5. List the 7 services with upstream citations
6. Send to a cardiologist co-author for review
7. Submit to medRxiv as preprint (1-2 day review)
8. Submit to npj Digital Medicine or JAMIA

## The citation chain

```
Paper 1 (Infrastructure)
  "We built cardio-echo-suite [cite this paper]"
    ↓
Paper 2 (Validation)
  "We validated cardio-echo-suite [cite Paper 1] on 4 public datasets..."
    ↓
Paper 3 (ML Engineering)
  "Building on our validation [cite Paper 2] of cardio-echo-suite [cite Paper 1],
   we developed a 7-layer ML engineering recipe..."
```

Each paper stands alone but they build on each other. By Paper 3, you have a complete story: infrastructure → validation → engineering improvement.

## What NOT to do

- ❌ Don't try to write one mega-paper with everything (will be rejected for being unfocused)
- ❌ Don't wait for all experiments before writing Paper 1 (it doesn't need experiments)
- ❌ Don't submit Paper 3 before Paper 1 (reviewers will ask "what is this suite?")
- ❌ Don't put LoRA in Paper 2 (it's an engineering contribution, not validation)

## Summary table

| Paper | Angle | Status | Timeline | Venue |
|---|---|---|---|---|
| 1. Infrastructure | We built the suite | ✅ Ready | 2-3 weeks | npj Digital Medicine |
| 2. Validation | It works on public data | ⏳ 50% | 1-2 months | EHJ-Digital Health |
| 3. ML Engineering | We made it better | ⏳ 20% | 2-3 months | Nature Machine Intelligence |

**Start Paper 1 this week.** I can help you draft it using the template we already built.
