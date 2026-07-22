# Marginal-CRPS comparison — ETHUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `features_ETHUSDT_2026-06-12.parquet` (2,872,918 rows)
- Windows: 5 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `0198737f783718c545bcb2b212563e8a488ba2c8` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | QRW Adaptive **(QRW)** | 0.0962 | 3/5 | 0.084, 0.130, 0.101, 0.068, 0.098 |
| 2 | CRW Correlated | 0.1116 | 1/5 | 0.101, 0.157, 0.100, 0.070, 0.129 |
| 3 | GARCH(1,1) | 0.1235 | 1/5 | 0.077, 0.177, 0.112, 0.089, 0.163 |
| 4 | GBM | 0.1277 | 0/5 | 0.119, 0.180, 0.114, 0.087, 0.139 |
| 5 | CRW Biased | 0.1372 | 0/5 | 0.121, 0.192, 0.123, 0.097, 0.154 |
| 6 | CRW Simple | 0.1375 | 0/5 | 0.129, 0.180, 0.123, 0.101, 0.155 |

## Verdict

The QRW marginal wins the CRPS endpoint in 3/5 windows; overall best is 'QRW Adaptive'.
