# Phase 3 multi-day predictive-edge audit

> **STALE / INVALIDATED 2026-06-13:** Inputs were built with the former
> `shift(-1)` outlier filter. Do not cite the edge or confidence intervals.

Structural QRW parameters are selected on June 1-4 and frozen.
June 5-7 are evaluated prequentially; only the regime bias is updated
after each completed day. Baselines use the same OBI and tick-direction
features and are refit only from past data.

| Day | Events | QRW Brier | Affine Brier | Logistic Brier | alpha_obi | alpha_direction | bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-06-05 | 2488424 | 0.071124 | 0.071969 | 0.071187 | 0.940121 | 0.672770 | -0.024187 |
| 2026-06-06 | 1089936 | 0.069820 | 0.069762 | 0.069035 | 0.940121 | 0.672770 | -0.019511 |
| 2026-06-07 | 1008840 | 0.081882 | 0.081533 | 0.081172 | 0.940121 | 0.672770 | -0.016726 |

## Block bootstrap

- QRW minus affine Brier: `-0.000368`, 95% CI `[-0.000443, -0.000291]`
- QRW minus logistic Brier: `0.000308`, 95% CI `[0.000255, 0.000359]`

## Verdict

QRW shows a statistically significant multi-day Brier edge over the fair affine baseline, but not over the nonlinear logistic baseline. This is predictive edge for the QRW link relative to the registered affine benchmark, not evidence of a uniquely quantum edge.
