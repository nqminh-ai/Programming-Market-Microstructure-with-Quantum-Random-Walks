# Phase 4 Checkpoint

**Status:** PASS

Feature source: `data\features\features_BTCUSDT_2026-06-12.parquet`

## Protocol

- Chronological train rows: `1144`
- Later test rows: `501`
- Simulated paths per model: `5000`
- Steps: `500`
- Tick size: `0.01`
- Event move probability: `0.1321`
- Forecast protocol: `fixed_origin_ex_ante_last_observed_features`
- Random seed: `2026`

## Acceptance

| Check | Status | Observed |
|---|---|---:|
| At least 4 models x 5 metrics | PASS | 6 models x 7 metrics |
| Simple CRW variance ratio near move probability | PASS | 0.127716 vs 0.132108 |
| GARCH optimizer converged | PASS | flag=0 |
| Coherent QRW variance ratio > 1.3 x CRW | PASS | 58.581154 vs 1.000000 |

## Observed Test Metrics

| Model | Wasserstein path MAE | Hit@1 | Hit@5 | Hit@10 | Mean direction log likelihood |
|---|---:|---:|---:|---:|---:|
| CRW Biased | 6.701165 | 24.80% | 20.26% | 18.95% | -1.061685 |
| CRW Correlated | 6.564672 | 52.00% | 53.30% | 54.04% | -0.688296 |
| CRW Simple | 6.558474 | 53.60% | 52.42% | 56.14% | -0.690501 |
| GARCH(1,1) | 8.107861 | 55.20% | 70.93% | 81.75% | -0.686804 |
| GBM | 10.119822 | 24.80% | 20.26% | 18.95% | -0.729498 |
| QRW Adaptive | 6.325979 | 75.20% | 79.74% | 81.05% | -0.618696 |

Lower Wasserstein error and less-negative direction log likelihood are
better. These are single-window descriptive results.

## Interpretation Notes

- Paths use the empirical event move probability, so a symmetric
  zero-inflated CRW targets `Var(X_T)/T = P(move)`.
- QRW path forecasts use only the last feature vector observed before
  the holdout; later holdout OBI/intensity values are not inputs.
- `wasserstein_path_mae` is the mean one-dimensional Wasserstein
  distance between each forecast cross-section and the realized price.
- Directional log likelihood is Bernoulli and comparable across models.
  AIC/BIC should only be compared within the same `likelihood_type` in
  `model_comparison_table.csv`.
- QRW superiority is not inferred from this engineering checkpoint.
  Statistical claims require Phase 5 and fresh multi-day holdout data.

## Artifacts

- `results/benchmark_results.csv`
- `results/model_comparison_table.csv`
- `results/garch_params.json`
- `reports/phase4_diagnostics.json`
