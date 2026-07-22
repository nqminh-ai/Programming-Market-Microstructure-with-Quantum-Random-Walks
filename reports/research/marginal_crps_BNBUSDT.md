# Marginal-CRPS comparison — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `features_BNBUSDT_multiday.parquet` (4,000,000 rows)
- Windows: 5 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `a9fe43483d072246012488a721940e86b77a4966` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | CRW Correlated | 0.0555 | 2/5 | 0.054, 0.031, 0.041, 0.111, 0.041 |
| 2 | GBM | 0.0573 | 1/5 | 0.053, 0.032, 0.041, 0.119, 0.042 |
| 3 | CRW Simple | 0.0610 | 0/5 | 0.058, 0.033, 0.043, 0.127, 0.044 |
| 4 | QRW Adaptive **(QRW)** | 0.0614 | 1/5 | 0.061, 0.052, 0.055, 0.081, 0.058 |
| 5 | CRW Biased | 0.0623 | 0/5 | 0.055, 0.035, 0.047, 0.131, 0.044 |
| 6 | GARCH(1,1) | 0.0632 | 1/5 | 0.062, 0.038, 0.037, 0.128, 0.051 |

## Verdict

The QRW marginal wins the CRPS endpoint in 1/5 windows; overall best is 'CRW Correlated'.
