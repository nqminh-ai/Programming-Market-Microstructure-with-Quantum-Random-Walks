# Marginal-CRPS comparison — BTCUSDT_69d_daycluster

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `features_BTCUSDT_69d.parquet` (25,000,000 rows)
- Windows: 10 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `81737c3dddf94218d4e096f659241ae3c80ef880` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | GARCH(1,1) | 2.5744 | 6/10 | 0.944, 1.218, 3.226, 0.649, 4.533, 1.081, 2.243, 1.195, 2.571, 8.084 |
| 2 | GBM | 2.8197 | 3/10 | 0.776, 0.983, 3.806, 0.388, 5.198, 0.910, 2.740, 1.335, 3.040, 9.021 |
| 3 | QRW Adaptive **(QRW)** | 3.3199 | 0/10 | 0.840, 1.376, 4.821, 0.068, 5.922, 1.427, 3.570, 1.651, 3.733, 9.792 |
| 4 | CRW Correlated | 3.3478 | 0/10 | 0.855, 1.403, 4.854, 0.035, 5.963, 1.447, 3.611, 1.681, 3.790, 9.838 |
| 5 | CRW Biased | 3.3645 | 1/10 | 0.863, 1.411, 4.863, 0.031, 5.985, 1.474, 3.632, 1.707, 3.812, 9.868 |
| 6 | CRW Simple | 3.3648 | 0/10 | 0.859, 1.417, 4.879, 0.039, 5.979, 1.477, 3.634, 1.701, 3.810, 9.853 |

## Verdict

On the registered primary endpoint (mean marginal CRPS), the QRW density-matrix marginal is NOT the best model in any of 10 windows; 'GARCH(1,1)' wins overall. The QRW offers no advantage on the distributional endpoint. The volatility story is NOT supported at this sample size: Spearman between realised volatility and the QRW's relative CRPS gap is -0.37 (p=0.293, 10 windows).
