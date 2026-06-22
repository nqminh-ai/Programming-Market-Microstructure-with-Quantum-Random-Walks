# Phase 5 Checkpoint

**Engineering status:** PASS

Feature source: `data\features\features_BTCUSDT_2026-06-12.parquet`

## Protocol

- Chronological train rows: `1869214`
- Later holdout rows: `1246144`
- Simulated paths per model: `3000`
- Simulated steps: `500`
- Bootstrap iterations: `500`
- Random seed: `2026`
- Distribution and tail tests use matched empirical/simulated sample
  sizes. Variance scaling uses the same comparison horizon for both.
- Scaling intervals use moving-block bootstrap for empirical returns
  and whole-path resampling for simulations.
- Benjamini-Hochberg adjusted p-values are included for each test
  family.

## Acceptance

| Check | Status | Observed |
|---|---|---:|
| Six test categories persisted | PASS | 6/6 CSV files |
| QRW KS p-value computed | PASS | raw=0.000545877, BH=0.000955286 |
| QRW variance beta has 95% CI | PASS | 1.0026 [0.9841, 1.0183] |
| QRW and empirical tail indices computed | PASS | 1597808.9782 vs 2.4748 |
| Scorecard compiled | PASS | top=CRW Correlated |

## Scorecard

| Rank | Model | Mean metric rank |
|---:|---|---:|
| 1 | CRW Correlated | 2.143 |
| 2 | CRW Biased | 3.571 |
| 3 | GARCH(1,1) | 3.857 |
| 4 | CRW Simple | 4.143 |
| 4 | GBM | 4.143 |
| 6 | QRW Heavy-Tail | 4.714 |
| 7 | QRW Adaptive | 5.286 |

## Interpretation Guardrail

This checkpoint validates implementation completeness on one
June 12, 2026 window. It is exploratory and does not establish
QRW superiority. A confirmatory claim still requires the frozen
protocol and fresh multi-day holdout described in the audit.

## Artifacts

- `results/distribution_tests.csv`
- `results/variance_scaling_results.csv`
- `results/autocorrelation_tests.csv`
- `results/tail_analysis.csv`
- `results/diebold_mariano_tests.csv`
- `results/scorecard_bootstrap_ci.csv`
- `results/model_aic_bic_comparison.csv`
- `results/final_comparison_table.csv`
- `results/scorecard.csv`
- `figures/variance_scaling.png`
- `figures/acf_comparison.png`
- `reports/phase5_diagnostics.json`
