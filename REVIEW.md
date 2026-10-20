# Week 5: peer review and author response

Project: *Reliable Probabilistic Time Series Imputation* (code, results and poster in this repository).
All numbers below are from `results/tables/metrics_{val,test}.csv` and `results/figures/` of the main run.

---

## Part 1. Peer review

### 1. Summary of problem, method and claimed contribution

**Problem.** Imputation methods often report a predictive distribution, not only a point value. The
project asks when these native distributions are reliable, that is calibrated and not needlessly wide,
and whether a forecasting foundation model can be turned into an imputer without losing calibration.

**Method.** Three model families are compared on identical, saved masks:

* a local-level Kalman smoother (analytic Gaussian, variances fitted by MLE on train);
* CSDI via PyPOTS (one univariate model per dataset, 100 draws);
* frozen Chronos-2-small, used two-sidedly: forecast from the left prefix, forecast from the
  reversed right suffix, and a heuristic blend that averages matching quantiles with linear weights.

Data are ETT_h1, Electricity (20 clients) and BeijingAir (PM2.5, 12 stations), split 60/20/20 in time.
In 256-step windows 24 or 128 values are hidden, scattered (`point`) or as one gap (`block`).
Metrics are MAE/RMSE of the median, pinball, CRPS, PCE, CCE, 80% coverage and width, and SIW.

**Claimed contribution.** (H1) contiguous gaps are harder than scattered points; (H2) median accuracy
and calibration can disagree; (H3) blending left and right forecasts changes calibration, and the
direction is not fixed. Plus a reproducible benchmark protocol with shared masks.

### 2. Strengths, with evidence

* **Clean, fair protocol.** All models are scored on the same saved masks
  (`results/masks/<split>/<dataset>.npz`), only hidden targets are scored, scaling and Kalman
  variances use train only, and val and test use different mask seeds. Leakage between splits is
  ruled out by construction because every window lies inside one split.
* **Calibration is read together with sharpness.** Coverage is always reported with width and SIW,
  so an over-wide model cannot look good. This matters here: Kalman reaches cov80 = 0.948 on
  Electricity block 128 (test) only with SIW = 2.39, six times wider than CSDI (0.41).
* **H1 is supported on every dataset and every model.** Example (test, Chronos blend MAE):
  ETT_h1 0.209 → 0.330, Electricity 0.178 → 0.283, BeijingAir 0.114 → 0.422 from point 24 to block 24.
* **The left/right/blend ablation is the right control for H3.** The along-gap plots
  (`figures/along_gap_*.png`) show coverage and width per position, which directly exposes how the
  blend behaves near the edges and in the middle of a gap.
* **Sampling error of CSDI is checked.** 50 vs 100 draws change the 80% width by 1–2% and CRPS in the
  third decimal (`tables/csdi_draws_test.csv`), so CSDI conclusions are not a Monte Carlo artefact.
* **Honest framing.** The README calls the blend a heuristic and the bridge figure an illustration.

### 3. Major concerns (validity or interpretation)

**M1. No uncertainty on the reported differences.** Each cell of the table is 60 windows
(`n_windows = 60`), and windows from the same channel and neighbouring periods are dependent.
Many rankings rest on small gaps: Electricity point 24 test CRPS is 0.127 (blend) vs 0.129 (CSDI);
BeijingAir point 24 is 0.082 (CSDI) vs 0.083 (Kalman). Without paired intervals these cannot be
called wins. The val/test agreement is a useful sanity check but not a substitute.

**M2. CRPS is computed three different ways and the ranking depends on it.** Kalman uses the analytic
Gaussian CRPS, CSDI the sample estimator, Chronos the quantile approximation (2 × mean pinball over
11 levels, tails beyond 0.05/0.95 ignored). On the common pinball metric the winner changes in two of
twelve test cells: Electricity point 24 (CRPS → blend, pinball → CSDI) and BeijingAir block 24
(CRPS → blend, pinball → Kalman). Cross-family CRPS comparisons are therefore not on one scale.

**M3. H3 is shown, but its cause is not identified.** The blend over-covers on scattered points
(CCE −0.13 BeijingAir point 24, −0.06 Electricity point 24) and under-covers on long gaps
(CCE +0.10 BeijingAir block 128, cov80 0.67). This is consistent with "direction is not fixed", but
two explanations are mixed: (a) the averaging rule itself (quantile averaging of two forecasts that
both ignore the other side), and (b) dataset properties such as seasonality and level shifts. Real
data cannot separate them because the true conditional distribution is unknown. The bridge figure is
analytic only and is not compared with Chronos output.

**M4. CSDI is not in its intended regime, so "CSDI under-covers" may be a setup effect.** CSDI is run
univariately (no cross-channel attention, which is its main design feature) and is trained for at most
40 epochs: on Electricity the best epoch is 39 of 40 and validation loss is still falling
(`logs/main_run.log`). CSDI under-covers everywhere (cov80 0.52–0.77, CCE > 0 in all 24 cells). The
draws check rules out sampling noise, but not under-training. The conclusion should be phrased as
"univariate CSDI with this budget", not CSDI in general.

**M5. The Kalman baseline is misspecified in a known way.** A local-level random walk ignores the
daily cycle. In the along-gap plot for ETT_h1 block 24 its coverage drops to 0.55 mid-gap while width
is the largest of all models; for block 128 it covers ≈1.0 with width ≈5. So its numbers on blocks
measure the misspecification, not "what a classical probabilistic method can do". A seasonal
state-space model would be a fairer classical reference.

### 4. Minor concerns (clarity or reproducibility)

* **m1.** CSDI training and sampling do not set a torch seed, and MPS is non-deterministic; reruns
  will not reproduce the CSDI numbers bit-for-bit. Only numpy seeds for masks and training windows
  are fixed.
* **m2.** `results/csdi/log.json` training times for ETT_h1 and Electricity are reconstructed from log
  timestamps after an interrupted run; this is stated in the file but should be in the README.
* **m3.** NLL is reported only for Kalman; the column is NaN for the others. Either drop it from the
  main table or explain it is not comparable.
* **m4.** Processed data (`data/`) are not committed; `python -m tsi.data` downloads via `tsdb`.
  The README should say this explicitly and give the expected download size and time.
* **m5.** Mask configs are not ordered in the CSV (point/block × 24/128 in mixed order), which makes
  the raw table hard to read; add a summary table with best model per cell.
* **m6.** Possible overlap of ETT/Electricity with Chronos pretraining data is listed as a limitation
  but not quantified; at least mark which datasets are known to be in the Chronos training corpus.
* **m7.** "Individual contributions" in the README is still empty.

### 5. Do the experiments support the claims?

| Claim | Verdict | Evidence |
|---|---|---|
| H1 blocks harder than points | **Supported** | MAE and CRPS higher for block at equal count for every model and dataset, val and test |
| H2 accuracy and calibration disagree | **Supported** | Kalman best MAE on BeijingAir point 24 (0.094) with the worst PCE (0.099); CSDI best pinball on Electricity point 128 with cov80 0.72 |
| H3 blend changes calibration in either direction | **Partly supported** | Direction differs between points (over-coverage) and long BeijingAir gaps (under-coverage), but the cause and the significance are not established (M1, M3) |
| "Model X is best on cell Y" | **Weak** | Margins often below plausible noise (M1) and sensitive to the CRPS estimator (M2) |

### 6. One decisive, feasible extra experiment

**Synthetic oracle test for the blend (addresses M3, and partly M1).**

Generate series from a Gaussian local-level model (and a second set with a daily seasonal component)
with known variances, so the exact conditional distribution of a gap is available from the Kalman
smoother with true parameters. Use the same windows and masks (block 24 and block 128, 60+ windows ×
several noise ratios). Run Chronos left, right and blend, and the oracle.

Measure, per gap position: 80% coverage, width ratio to the oracle, and PIT histograms; summarise with
PCE and CCE, with paired bootstrap intervals over windows.

**Why decisive.** If the blend is over-wide in the middle of the gap relative to the oracle (as the
analytic bridge suggests, about 1.4×) on synthetic data, the effect is caused by the averaging rule
and H3 becomes a mechanistic claim. If it is calibrated on synthetic data, the real-data
miscalibration comes from the data and H3 should be restated.

**Feasibility.** No training: Chronos is frozen and the oracle is a closed-form smoother; the synthetic
set is no larger than one real dataset. Estimated 1–2 hours including bootstrap, using the
existing `tsi.masks`, `tsi.chronos_imputer` and `tsi.metrics` code.

### 7. Reproducibility assessment

**Good.** One config file, `run_all.sh`, fixed numpy seeds, saved masks and predictions, pinned
`requirements.txt`, and `evaluate` rebuilds every table and figure from saved predictions without
rerunning models. Trained CSDI checkpoints (`results/csdi/<dataset>/model.pypots`) are committed.

**Gaps.** No torch seed (m1), data downloaded at run time (m4), Apple MPS as the tested device, and
CSDI training time per dataset is 0.35–0.54 h on an M4, so a full rerun takes several hours.
Tables can be reproduced exactly from the committed predictions; model outputs can be reproduced only
approximately for CSDI.

### 8. Overall recommendation

**Accept with major revisions.** The protocol is careful and the two simpler hypotheses are well
supported by consistent val/test evidence. The central and most interesting claim (H3) is only
partly supported: its direction is shown but its cause is not, and several model rankings are within
noise and depend on the CRPS estimator. Paired intervals (M1), a common scoring rule (M2) and the
synthetic oracle test (M3) are all cheap and would make the conclusions defensible.

---

## Part 2. Author response

| # | Agree? | What will change | Cannot be finished in time |
|---|---|---|---|
| M1 no uncertainty | **Agree** | Paired block bootstrap over windows, resampling by channel, for every model difference in the summary table; differences whose 95% interval covers zero will be reported as ties | More windows per cell (needs new CSDI sampling, ~0.5 h per split and dataset) |
| M2 three CRPS estimators | **Agree** | Use pinball (same 11 levels for all models) as the primary score; compute CRPS for Kalman and CSDI also by the same quantile approximation so all three are on one scale; keep exact CRPS as a secondary column | — |
| M3 cause of H3 | **Agree** | Run the synthetic oracle test from Section 6 and add along-gap width ratio to the oracle | Seasonal synthetic variant if time is short; the plain local-level set is done first |
| M4 CSDI regime | **Partly agree** | Restate the claim as "univariate CSDI, 40 epochs"; report the validation curve; retrain Electricity with a larger epoch budget and check whether coverage moves | Multivariate CSDI on all channels: needs a different data pipeline and several GPU hours |
| M5 Kalman misspecified | **Partly agree** | Make the role explicit: it is a deliberately simple analytic baseline. Add a note and the along-gap evidence | Seasonal state-space (local level + daily cycle) baseline: feasible later, not before the deadline |

Minor points: m1 add `torch.manual_seed` and document MPS non-determinism; m2, m4, m7 fix the README;
m3 move NLL to a Kalman-only note; m5 add a sorted best-per-cell table; m6 mark datasets known to be
in the Chronos corpus.
