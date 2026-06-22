# Phase 5 Checkpoint

**Engineering status:** PASS

Feature source: `data\features\features_BTCUSDT_2026-06-12.parquet`

## Protocol

- Chronological train rows: `1144`
- Later holdout rows: `764`
- Simulated paths per model: `5000`
- Simulated steps: `500`
- Bootstrap iterations: `1000`
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
| Four test categories persisted | PASS | 4/4 CSV files |
| QRW KS p-value computed | PASS | raw=0.00583208, BH=0.00583208 |
| QRW variance beta has 95% CI | PASS | 1.0023 [0.9853, 1.0192] |
| QRW and empirical tail indices computed | PASS | 1656334.3823 vs 1.2886 |
| Scorecard compiled | PASS | top=CRW Correlated |

## Scorecard

| Rank | Model | Mean metric rank |
|---:|---|---:|
| 1 | CRW Correlated | 1.857 |
| 2 | CRW Simple | 2.571 |
| 3 | QRW Adaptive | 2.857 |
| 4 | CRW Biased | 3.571 |
| 5 | GARCH(1,1) | 4.857 |
| 6 | GBM | 5.000 |

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
- `results/final_comparison_table.csv`
- `results/scorecard.csv`
- `figures/variance_scaling.png`
- `figures/acf_comparison.png`
- `reports/phase5_diagnostics.json`
