# Reliable Probabilistic Time Series Imputation

Code for the Models of Sequential Data project (Skoltech, 2026).
Authors: Vladislav Voronin, Kirill Lalayants, Mikhail Bezrukov, Fedor Zhirkov. Supervisor: Vitaliy Pozdnyakov.

**Question.** When are native predictive distributions reliable, and how does a two-sided
forecasting adaptation affect calibration?

**Hypotheses.** (1) Contiguous blocks are harder than scattered points. (2) Median accuracy and
calibration can disagree. (3) Blending left and right forecasts can change calibration in either
direction.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Tested on macOS (Apple M4, 24 GB) with Python 3.13. Chronos-2 runs on CPU (faster than MPS at this
size); CSDI trains on MPS. On a CUDA machine set `csdi.device: cuda` in the config.

## Run

```bash
./run_all.sh configs/pilot.yaml   # small pilot: checks the pipeline end to end
./run_all.sh                      # main experiment (configs/main.yaml)
```

Steps (each can be run on its own; all read the same config):

| Step | Command | Output |
|---|---|---|
| Data | `python -m tsi.data` | `data/<dataset>.npz`: scaled series, split bounds, train-only mean/std |
| Masks | `python -m tsi.masks` | `results/masks/<split>/<dataset>.npz`: windows and hidden indices |
| Kalman | `python -m tsi.kalman` | `results/predictions/<split>/<dataset>/kalman.npz` |
| Chronos-2 | `python -m tsi.chronos_imputer` | `chronos_left.npz`, `chronos_right.npz`, `chronos_blend.npz` |
| CSDI | `python -m tsi.csdi_imputer [--skip-train]` | `csdi.npz` (100 draws), checkpoint in `results/csdi/` |
| Evaluate | `python -m tsi.evaluate` | `results/tables/*.csv|md`, `results/figures/*.png` |

`evaluate` rebuilds every table and figure from saved predictions, without rerunning models.

## Protocol

* **Data.** ETT_h1 (7 channels, hourly); Electricity (2012–2014, hourly means, the first 20 clients
  by id without zero runs longer than 24 h in the train part); BeijingAir (PM2.5 at 12 stations,
  hourly). Loaded with `tsdb`; raw data are not committed.
* **Split.** Chronological 60/20/20 train/validation/test per dataset. Scaling, Kalman variances and
  CSDI use train only. Every window lies fully inside its split.
* **Windows and masks.** Windows of 256 steps without native missing values ("eligible complete
  windows"). In each window we hide 24 or 128 values, either scattered (`point`) or as one
  contiguous gap (`block`). The first and last 24 points stay observed. All four mask configs use
  the same windows, and all models use the same saved masks. Validation and test use different
  seeds. Only hidden targets are scored.
* **Kalman smoother.** Local-level model; process and observation variances fitted per channel by
  maximum likelihood on the last 6000 train points. The predictive distribution of a missing
  measurement includes the observation variance. Parameter uncertainty is excluded.
* **CSDI (PyPOTS).** One model per dataset, pooled over channels (univariate windows), ~0.41M
  parameters. PyPOTS `mix` masking strategy: half of the training masks are our point/block masks
  (1–128 hidden values), half are PyPOTS random-ratio masks. Early stopping on masked validation
  windows. 100 draws per window; the median and quantiles are empirical.
* **Chronos-2-small** (`autogluon/chronos-2-small`, frozen). For every run of hidden values:
  forecast from the left prefix, and from the reversed right suffix; other hidden runs stay NaN in
  the context. The blend averages matching quantiles with linear weights
  `w_left(j) = (k - j) / (k + 1)`. This is a heuristic, not conditional inference.

## Metrics

Point: MAE, MSE, RMSE of the median (train-scaled units), MRE in original units (guarded against a
near-zero denominator). Probabilistic: pinball loss averaged over q = 0.1…0.9; CRPS (Kalman:
analytic; CSDI: from samples; Chronos: quantile-score approximation, 2 × mean pinball over all
levels); NLL (Kalman only). Calibration: PCE = mean_q |q − c(q)|; CCE = mean_s [s − coverage(s)]
over central intervals s ∈ {0.2, 0.4, 0.6, 0.8, 0.9} (positive = undercoverage); 80% coverage and
width; SIW = mean_s mean(Width_s) / IQR_s(y). Read coverage together with width.

Quantile levels: 0.05, 0.1, 0.2, …, 0.9, 0.95 (all native Chronos-2 levels).

## Checks and ablations

* Left-only vs right-only vs blended Chronos quantiles on identical targets
  (`figures/along_gap_*.png`: 80% coverage and width along the gap position).
* CSDI 50 vs 100 draws (`tables/csdi_draws_*.csv`).
* Scattered vs contiguous at equal hidden counts (24, 128).
* `figures/bridge.png`: analytical Gaussian random walk with noiseless endpoints and a 24-step gap;
  at the midpoint the blended quantiles are about 1.4 times wider than the exact bridge. This is an
  illustration, not a measured result.

## Limitations

Synthetic masks; only eligible complete windows; channels and windows are dependent, so cases are
not independent repeats; possible overlap with Chronos pretraining data; finite diffusion samples;
Gaussian misspecification of the local-level model (seasonality is not modelled).

## References

Bergmeir (2023); Du et al. (2024), TSI-Bench; Adler et al. (2026); Tashiro et al. (2021), CSDI;
Gneiting & Raftery (2007); PyPOTS and Chronos documentation.

## Individual contributions

To be filled in by the team.
