# Marginal-CRPS comparison — ETHUSDT_69d_daycluster

**Status:** `EXPLORATORY_ONLY_NOT_CONFIRMATORY` — exploratory. Registered PRIMARY endpoint: mean fixed-origin marginal CRPS.

- Protocol: `fixed_origin_marginal_density_matrix_ar1_obi_v4`
- Feature file: `features_ETHUSDT_69d.parquet` (25,000,000 rows)
- Windows: 10 non-overlapping chronological splits · n_steps=200 · n_paths=2000
- Git commit: `81737c3dddf94218d4e096f659241ae3c80ef880` · Python 3.14.5

## Mean marginal CRPS across windows (lower = better)

| Rank | Model | mean CRPS | windows best | per-window CRPS |
|---:|---|---:|---:|---|
| 1 | QRW Adaptive **(QRW)** | 0.0938 | 10/10 | 0.088, 0.145, 0.165, 0.078, 0.049, 0.096, 0.097, 0.066, 0.059, 0.094 |
| 2 | CRW Correlated | 0.1056 | 0/10 | 0.104, 0.158, 0.184, 0.081, 0.050, 0.107, 0.117, 0.080, 0.067, 0.109 |
| 3 | GBM | 0.1140 | 0/10 | 0.110, 0.170, 0.202, 0.079, 0.053, 0.123, 0.134, 0.085, 0.078, 0.105 |
| 4 | GARCH(1,1) | 0.1234 | 0/10 | 0.105, 0.197, 0.212, 0.099, 0.056, 0.135, 0.126, 0.093, 0.072, 0.138 |
| 5 | CRW Simple | 0.1288 | 0/10 | 0.126, 0.185, 0.212, 0.101, 0.060, 0.139, 0.145, 0.099, 0.086, 0.135 |
| 6 | CRW Biased | 0.1289 | 0/10 | 0.125, 0.186, 0.219, 0.096, 0.062, 0.137, 0.146, 0.098, 0.087, 0.132 |

## Verdict

The QRW marginal has the lowest mean marginal CRPS in every window. The QRW does fall further behind as realised volatility rises (Spearman +0.70, p=0.025, 10 windows) -- consistent with it not modelling volatility dynamics.
