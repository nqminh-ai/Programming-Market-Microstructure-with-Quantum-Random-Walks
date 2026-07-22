# Marginal-CRPS comparison — BTCUSDT

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `features_BTCUSDT_recent_subset.parquet` (4,000,000 rows)
- Windows: 5 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `0198737f783718c545bcb2b212563e8a488ba2c8` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | GBM | 1.8576 | 3/5 | 2.796, 0.715, 3.694, 0.742, 1.341 |
| 2 | GARCH(1,1) | 2.0308 | 2/5 | 3.700, 0.948, 3.273, 0.893, 1.340 |
| 3 | QRW Adaptive **(QRW)** | 2.3470 | 0/5 | 3.913, 0.973, 4.234, 0.877, 1.739 |
| 4 | CRW Correlated | 2.3845 | 0/5 | 3.955, 1.014, 4.265, 0.914, 1.775 |
| 5 | CRW Biased | 2.3946 | 0/5 | 3.970, 1.024, 4.271, 0.927, 1.782 |
| 6 | CRW Simple | 2.3987 | 0/5 | 3.975, 1.022, 4.278, 0.922, 1.796 |

## Verdict

On the registered primary endpoint (mean marginal CRPS), the QRW density-matrix marginal is NOT the best model in any of 5 windows; 'GBM' wins overall. The QRW offers no advantage on the distributional endpoint.
