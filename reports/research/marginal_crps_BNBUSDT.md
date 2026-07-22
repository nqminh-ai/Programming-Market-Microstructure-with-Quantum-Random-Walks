# Marginal-CRPS comparison — BNBUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `bnb_combined.parquet` (4,000,000 rows)
- Windows: 5 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `0198737f783718c545bcb2b212563e8a488ba2c8` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | GARCH(1,1) | 0.0927 | 2/5 | 0.098, 0.095, 0.087, 0.150, 0.034 |
| 2 | QRW Adaptive **(QRW)** | 0.0958 | 3/5 | 0.091, 0.085, 0.098, 0.126, 0.080 |
| 3 | CRW Correlated | 0.0967 | 0/5 | 0.098, 0.088, 0.089, 0.172, 0.037 |
| 4 | GBM | 0.0982 | 0/5 | 0.096, 0.095, 0.091, 0.174, 0.036 |
| 5 | CRW Biased | 0.1036 | 0/5 | 0.100, 0.098, 0.100, 0.179, 0.042 |
| 6 | CRW Simple | 0.1082 | 0/5 | 0.111, 0.105, 0.099, 0.191, 0.036 |

## Verdict

The QRW marginal wins the CRPS endpoint in 3/5 windows; overall best is 'GARCH(1,1)'.
