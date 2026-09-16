# Research Phase Checklist — cardio-echo-suite

> **Goal**: From "I have an idea" to "I have a published preprint" in 6 months.
> **Cost**: $0 (all public data, free tools)
> **Risk**: Zero (no patients, no regulators, no liability)

---

## Month 1: Setup + Credentialing

### Week 1
- [ ] **Mon**: Apply for PhysioNet account at https://physionet.org/register/
- [ ] **Mon**: Start CITI human-subjects training (https://about.citiprogram.org/)
  - 6 hours total, can spread over the week
  - Save the completion certificate PDF
- [ ] **Tue**: Email EchoNet team (echonet@stanford.edu) requesting dataset access
  - Attach CITI certificate
  - Brief research proposal (1 paragraph)
- [ ] **Wed**: Submit PhysioNet credentialed access application
  - Upload CITI certificate
  - Institutional affiliation letter (or supervisor email)
  - Research description: "Validation of open-source cardiac AI suite on public datasets"
- [ ] **Thu**: Read the 3 key papers:
  - Ouyang et al. Nature 2020 (EchoNet-Dynamic)
  - Holste et al. JAMA 2025 (PanEcho)
  - Wagner et al. Sci Data 2020 (PTB-XL)
- [ ] **Fri**: Clone the repo locally:
  ```bash
  git clone https://github.com/[USER]/cardio-echo-suite.git
  cd cardio-echo-suite
  docker compose -f deploy/docker-compose.yml up -d
  curl http://localhost:8080/healthz
  ```

### Week 2
- [ ] **Mon**: PhysioNet credentialing typically approved by now
- [ ] **Mon**: Download PTB-XL (~3 GB):
  ```bash
  wget --user=YOUR_USER --password=YOUR_PASS \
    https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
  unzip ptb-xl-*.zip -d /data/public/ptbxl
  ```
- [ ] **Tue**: EchoNet-Dynamic access typically granted by now
- [ ] **Tue**: Download EchoNet-Dynamic (~6 GB):
  ```bash
  # Link from EchoNet team
  unzip EchoNet-Dynamic.zip -d /data/public/echonet-dynamic
  ```
- [ ] **Wed**: Verify datasets:
  ```bash
  ls /data/public/echonet-dynamic/Videos/ | wc -l   # Should be 10030
  head -5 /data/public/echonet-dynamic/FileList.csv
  ls /data/public/ptbxl/records100/ | wc -l
  head -5 /data/public/ptbxl/ptbxl_database.csv
  ```
- [ ] **Thu**: Test cardio-echo-suite is running:
  ```bash
  # Echo dry-run
  curl -X POST "http://localhost:8080/v1/echo/full?patient_id=TEST&skip=echonet,echoprime" \
    -F "file=@/data/public/echonet-dynamic/Videos/0X10A28877E97DF540.avi" | python -m json.tool

  # ECG dry-run
  curl -X POST "http://localhost:8080/v1/ecg/full?patient_id=TEST" \
    -F "file=@/data/public/ptbxl/records100/00000/00001_lr.dat" | python -m json.tool
  ```
- [ ] **Fri**: Start research notebook (Notion / Overleaf / Word doc):
  - Write 1-paragraph research question
  - Write 1-paragraph methods draft
  - List the metrics you'll report

### Week 3
- [ ] **Mon**: Run first echo shakedown (50 cases):
  ```bash
  python scripts/public_data_shakedown.py \
    --dataset echonet-dynamic \
    --data-dir /data/public/echonet-dynamic/Videos \
    --labels /data/public/echonet-dynamic/FileList.csv \
    --orchestrator-url http://localhost:8080 \
    --output /tmp/echo_shakedown_50.json \
    --max-n 50 \
    --i-understand-this-is-not-validation
  ```
- [ ] **Tue**: Analyze shakedown results — what's the MAE? Is the pipeline working?
- [ ] **Wed**: Run first ECG shakedown (50 cases):
  ```bash
  python scripts/public_data_shakedown.py \
    --dataset ptbxl \
    --data-dir /data/public/ptbxl/records100 \
    --labels /data/public/ptbxl/ptbxl_database.csv \
    --orchestrator-url http://localhost:8080 \
    --output /tmp/ecg_shakedown_50.json \
    --max-n 50 \
    --i-understand-this-is-not-validation
  ```
- [ ] **Thu**: Fix any bugs discovered in shakedown
- [ ] **Fri**: Submit IRB protocol IN PARALLEL (use `docs/irb/PROTOCOL_TEMPLATE.md`)
  - Don't wait for research phase to finish
  - IRB approval takes 2-3 months

### Week 4
- [ ] **Mon**: Run full echo shakedown (500 cases — 20% of test set):
  ```bash
  python scripts/public_data_shakedown.py \
    --dataset echonet-dynamic \
    --data-dir /data/public/echonet-dynamic/Videos \
    --labels /data/public/echonet-dynamic/FileList.csv \
    --orchestrator-url http://localhost:8080 \
    --output /tmp/echo_shakedown_500.json \
    --max-n 500 \
    --i-understand-this-is-not-validation
  ```
- [ ] **Tue**: Run full ECG shakedown (500 cases):
  ```bash
  python scripts/public_data_shakedown.py \
    --dataset ptbxl \
    --data-dir /data/public/ptbxl/records100 \
    --labels /data/public/ptbxl/ptbxl_database.csv \
    --orchestrator-url http://localhost:8080 \
    --output /tmp/ecg_shakedown_500.json \
    --max-n 500 \
    --i-understand-this-is-not-validation
  ```
- [ ] **Wed**: Compute initial statistics (MAE, AUROC, 95% CIs)
- [ ] **Thu**: First subgroup analysis (sex, age where available)
- [ ] **Fri**: End-of-month review — is the pipeline working? Are results reasonable?

---

## Month 2: Scale up + analyze

### Week 5
- [ ] **Mon-Wed**: Run full EchoNet-Dynamic test set (2,437 cases)
  - Will take ~12-24 hours depending on CPU
  - Save results to `/data/results/echo_full.json`
- [ ] **Thu-Fri**: Run full PTB-XL test set (~4,000 cases)
  - Will take ~6-12 hours
  - Save results to `/data/results/ecg_full.json`

### Week 6
- [ ] **Mon-Tue**: Compute final statistics:
  - Echo: MAE, Pearson r, Bland-Altman, subgroup MAE by sex/age/EF category
  - ECG: AUROC per arrhythmia, F1, Brier score, calibration plot, subgroup AUROC
- [ ] **Wed**: Make figures (use matplotlib):
  - Figure 1: Bland-Altman plot
  - Figure 2: Reliability diagram (calibration)
  - Figure 3: Subgroup forest plot
  - Figure 4: Example failure cases (3-5 worst)
- [ ] **Thu**: Manual review of 20 worst echo cases — categorize failure modes
- [ ] **Fri**: Manual review of 20 worst ECG cases — categorize failure modes

### Week 7
- [ ] **Mon-Tue**: System performance measurement:
  - Mean inference latency (echo, ECG)
  - Peak memory
  - Cold start time
  - Error rate
- [ ] **Wed**: Compare to published academic results:
  - EchoNet-Dynamic published MAE: 4.1 EF%
  - ECG-FM published AUROC for AFib: 0.97
  - Are we within 1 EF% / 0.05 AUROC?
- [ ] **Thu**: Draft results section of paper (use template)
- [ ] **Fri**: Draft methods section

### Week 8
- [ ] **Mon**: Draft introduction
- [ ] **Tue**: Draft discussion (strengths, limitations, clinical implications)
- [ ] **Wed**: Draft abstract + conclusions
- [ ] **Thu**: Send to co-authors for feedback
- [ ] **Fri**: Begin addressing feedback

---

## Month 3: Polish paper + submit

### Week 9
- [ ] **Mon-Wed**: Incorporate co-author feedback
- [ ] **Thu**: Finalize figures (high-res, journal-ready)
- [ ] **Fri**: Format references (use Zotero or Mendeley)

### Week 10
- [ ] **Mon**: Final proofread
- [ ] **Tue**: Submit to medRxiv as preprint
  - https://www.medrxiv.org/
  - Free, fast (1-2 day review)
  - Gets you a DOI + citation
- [ ] **Wed**: Submit to peer-reviewed journal (if aiming higher)
  - Suggested: European Heart Journal - Digital Health
  - OR: npj Digital Medicine
  - OR: JAMA Cardiology (stretch goal)
- [ ] **Thu-Fri**: Take a breath. You've published.

### Week 11-12
- [ ] Address journal reviewer comments (typical: 4-8 weeks to first decision)
- [ ] In parallel: start planning Phase V (local validation)
  - Has IRB approved yet? (submitted in week 3, decision usually by week 12-15)
  - If approved: start honest broker data pull
  - If not approved yet: follow up with IRB

---

## Month 4-6: Parallel tracks

### Track A: Wait for journal decision
- Typical review time: 6-12 weeks
- If accepted: revise per reviewer comments
- If rejected: submit to lower-tier journal

### Track B: Start local validation prep
- [ ] Month 4: IRB approval should arrive
- [ ] Month 4-5: Honest broker pulls 100-300 historical cases
- [ ] Month 5: Run scripts/validate.py on local data
- [ ] Month 6: Subgroup analysis + failure mode review

### Track C: Build reputation
- [ ] Submit abstract to conference (ACC, ESC, ASE, ISCE)
- [ ] Tweet thread about the paper (with cardiologist champion)
- [ ] LinkedIn post
- [ ] Add to your CV / portfolio
- [ ] Consider follow-up paper: "cardio-echo-suite: Local validation at [Hospital X]"

---

## Concrete deliverables per month

| Month | Deliverable | Status check |
|---|---|---|
| 1 | Datasets downloaded, shakedown scripts run | Pipeline works end-to-end |
| 2 | Full results on EchoNet-Dynamic + PTB-XL | Numbers ready for paper |
| 3 | Paper submitted to medRxiv + journal | DOI assigned |
| 4 | IRB approval, local data extraction starting | Honest broker engaged |
| 5 | Local validation running | 100+ cases processed |
| 6 | Local validation complete, paper 2 drafted | Go/no-go for clinical pilot |

---

## Weekly time commitment

| Activity | Hours/week |
|---|---|
| Running experiments | 3-5 |
| Analyzing results | 2-3 |
| Writing paper | 3-5 (months 2-3 only) |
| Reading literature | 1-2 |
| Total | ~10-15 hours/week |

**6 months × 12 hours/week = ~300 hours total.** This is achievable as a side project alongside a full-time job.

---

## Red flags (stop and rethink)

- **Week 2 and PhysioNet still not approved** → email PhysioNet support
- **Week 4 and shakedown shows >20% error rate** → debug pipeline before scaling up
- **Week 6 and MAE is >8 EF%** → something is wrong, results won't be publishable
- **Month 3 and IRB hasn't responded** → escalate via your institution's research office
- **Month 4 and co-authors haven't responded** → change co-authors or go solo

---

## Green flags (you're on track)

- ✅ Week 2: Datasets downloaded, pipeline runs
- ✅ Week 4: Shakedown results look reasonable (MAE < 6 EF%, AUROC > 0.85)
- ✅ Week 6: Full results computed, figures made
- ✅ Week 8: Paper draft complete, co-authors reviewing
- ✅ Week 10: Preprint on medRxiv
- ✅ Month 4: IRB approved, local data extraction starting

---

## What success looks like at month 6

1. **One published preprint** on medRxiv (and ideally accepted at a journal)
2. **IRB approval** for local validation
3. **Local validation underway** (100+ cases processed)
4. **A cardiologist champion** who's excited about the project
5. **Conference abstract submitted** (ACC, ESC, ASE)
6. **Decision point**: Should we pursue commercial path? (If yes, start Phase V → C)

---

## What failure looks like (and how to avoid it)

**Failure mode 1**: "I downloaded the data but never ran the analysis"
- Fix: Set weekly deadlines with a colleague who holds you accountable

**Failure mode 2**: "I ran the analysis but never wrote the paper"
- Fix: Use the paper template. Write badly first, edit later. Don't aim for perfection.

**Failure mode 3**: "I wrote the paper but never submitted"
- Fix: Submit to medRxiv first (low bar, fast). Then iterate for journal submission.

**Failure mode 4**: "I submitted but got rejected"
- Fix: Address reviewer comments. Resubmit. If rejected again, go one tier lower. Publication is a numbers game.

**Failure mode 5**: "I published but never did local validation"
- Fix: This is fine if your goal was just research. If your goal is clinical deployment, commit to Phase V next.
